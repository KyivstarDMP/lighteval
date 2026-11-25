# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Token-alignment engine for loglikelihood scoring over OpenAI-compatible APIs.

Used by the vLLM OpenAI-endpoint backend (``/v1/completions`` echo route +
vLLM's ``prompt_logprobs`` chat-completions extension). Continuation isolation
works on what the server returns — character offsets or decoded token piece
strings — because API clients have no local tokenizer to replay the server's
tokenisation with.
"""

import logging
import re
from bisect import bisect_left


logger = logging.getLogger(__name__)


def find_continuation_start(text_offset: list[int] | None, context_str: str) -> int:
    """Return the index in the token sequence where the continuation begins.

    Character-offset alignment via ``text_offset`` (exact, uses the API's own
    tokenisation — vLLM always returns it on ``/v1/completions``).

    ``context_str`` must already have trailing whitespace stripped by the
    caller: like ``LightevalModel.tok_encode_pair``, whitespace between
    context and continuation is scored as part of the continuation.
    """
    if not text_offset:
        logger.warning(
            "Could not align continuation tokens: the provider returned no "
            "text_offset. Logprob results may be inaccurate."
        )
        return 0

    # First token at or past the context's end; past-the-end == empty continuation.
    return bisect_left(text_offset, len(context_str))


def check_argmax(
    tokens: list[str],
    top_logprobs: list,
    cont_start: int,
) -> bool:
    """Return True if every continuation token was the model's top-1 prediction.

    Mirrors vLLM's ``rank == 1`` check. The last token in ``tokens`` is the
    newly generated token (from ``max_tokens=1``) and is excluded from the
    check.
    """
    if not top_logprobs:
        return False

    cont_end = len(tokens) - 1  # exclude the single generated token at the end
    if cont_start >= cont_end:
        return True  # empty continuation trivially matches

    for i in range(cont_start, cont_end):
        if i >= len(top_logprobs):
            return False
        top_dict = top_logprobs[i]
        if not top_dict:
            return False
        # The top-1 token is the highest-logprob entry (the actual token may
        # be present as an extra entry on some servers, e.g. vLLM).
        top_token = max(top_dict, key=top_dict.get)
        if i >= len(tokens) or tokens[i] != top_token:
            return False

    return True


def gpt2_byte_decoder() -> dict[str, int]:
    """Fixed GPT-2 byte<->unicode bijection, used by every byte-level BPE tokenizer.

    (GPT-2/3, Llama-3, Qwen, ...). Server-side token strings for such models
    are byte-alphabet pieces ("Ġ" = space, "Ċ" = newline, multi-piece UTF-8
    for non-ASCII); mapping them back through this table is what turns a piece
    suffix into comparable text.

    Deliberately self-contained (the inverse of ``transformers``'
    ``bytes_to_unicode``): this module must import in harness environments
    whose transformers version does not expose that helper.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = gpt2_byte_decoder()
_SP_BYTE_PIECE_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def pieces_to_text(pieces: list[str], byte_mode: bool = True) -> str:
    """Best-effort decode of consecutive tokenizer piece strings into text.

    Handles the three piece flavours seen in server token strings:
    SentencePiece pieces ("▁" for space, "<0xHH>" byte fallbacks),
    byte-level BPE pieces (GPT-2 byte alphabet), and already-plain text
    (identity). Pieces are decoded as one byte stream so UTF-8 characters
    split across pieces survive.

    ``byte_mode=False`` disables the GPT-2 byte-alphabet interpretation:
    the byte alphabet overlaps printable Latin-1 (``«``, ``é``, …), so a
    literal SentencePiece token like ``"«"`` is indistinguishable from a
    byte piece by inspection — callers try both modes and keep whichever
    aligns (see ``align_continuation_suffix``).
    """
    buffer = bytearray()
    for piece in pieces:
        byte_match = _SP_BYTE_PIECE_RE.fullmatch(piece)
        if byte_match:
            buffer.append(int(byte_match.group(1), 16))
        elif byte_mode and piece and all(char in _BYTE_DECODER for char in piece):
            buffer.extend(_BYTE_DECODER[char] for char in piece)
        else:
            buffer.extend(piece.replace("▁", " ").encode("utf-8"))
    return buffer.decode("utf-8", errors="replace")


def align_continuation_suffix(pieces: list[str], continuation: str) -> int | None:
    """Return the index of the first token belonging to the continuation.

    Walks the token pieces from the end until their decoded text covers the
    continuation string, then extends backwards over whitespace-only pieces
    (mirroring ``tok_encode_pair``, which moves whitespace at the
    context/continuation boundary into the continuation). Alignment from
    the end sidesteps image-token expansion earlier in the prompt.

    Many chat templates whitespace-trim message content (e.g. Gemma's
    ``| trim``), so a continuation like ``" A"`` renders as ``"A"`` — when
    the verbatim continuation is not the prompt's suffix, alignment retries
    against its stripped form; the boundary whitespace the template emits
    instead is then absorbed by the whitespace walk.

    Both piece interpretations are tried (see ``pieces_to_text``): byte-BPE
    models need the GPT-2 byte alphabet, while SentencePiece models emit
    literal text whose Latin-1 punctuation (``«``, ``»``, …) would be
    mis-decoded as bytes.

    Returns ``None`` when the decoded suffix ends with no (mode, form)
    combination (token strings not reconstructible for this model).
    """
    stripped = continuation.strip()
    targets = (continuation,) if stripped == continuation else (continuation, stripped)
    for byte_mode in (True, False):
        for target in targets:
            idx = len(pieces)
            acc: list[str] = []
            while idx > 0 and len(pieces_to_text(acc, byte_mode)) < len(target):
                idx -= 1
                acc.insert(0, pieces[idx])
            if not pieces_to_text(acc, byte_mode).endswith(target):
                continue
            while idx > 0:
                piece_text = pieces_to_text([pieces[idx - 1]], byte_mode)
                if piece_text == "" or piece_text.strip() != "":
                    break
                idx -= 1
                acc.insert(0, pieces[idx])
            return idx
    return None


def extract_prompt_logprob_entries(prompt_logprobs: list | None) -> list[tuple[str, float, int]] | None:
    """Flatten a vLLM ``prompt_logprobs`` positions list into (piece, logprob, rank) rows.

    Each position holds a mapping ``token_id -> {logprob, rank,
    decoded_token}`` covering the top-1 token and, when different, the
    actual prompt token — the actual token is the entry with the highest
    rank. The leading ``None`` (first prompt token, no context) is skipped.
    """
    if not prompt_logprobs:
        return None

    entries: list[tuple[str, float, int]] = []
    for position in prompt_logprobs:
        if not position:
            continue
        actual = max(position.values(), key=lambda info: info.get("rank") or 0)
        entries.append(
            (
                actual.get("decoded_token") or "",
                float(actual.get("logprob", float("-inf"))),
                int(actual.get("rank") or 0),
            )
        )
    return entries
