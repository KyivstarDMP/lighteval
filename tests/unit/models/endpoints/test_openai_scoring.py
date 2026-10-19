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

"""Unit tests for the token-alignment engine (``openai_scoring``).

Pure functions — no mocking, no network. Used by the vLLM OpenAI-endpoint
backend's loglikelihood routes.
"""

from lighteval.models.endpoints.openai_scoring import (
    align_continuation_suffix,
    check_argmax,
    extract_prompt_logprob_entries,
    find_continuation_start,
    gpt2_byte_decoder,
    pieces_to_text,
)


# ---------------------------------------------------------------------------
# 1. find_continuation_start
# ---------------------------------------------------------------------------


class TestFindContinuationStart:
    def test_text_offset_exact_boundary(self):
        """Continuation starts exactly at len(context_str) characters."""
        # context = "Q:" (2 chars), continuation = " A"
        result = find_continuation_start([0, 1, 2, 4], "Q:")
        assert result == 2

    def test_text_offset_midpoint(self):
        """Works when the context ends mid-word and text_offset values are larger."""
        result = find_continuation_start([0, 5, 11, 12], "Hello world ")
        assert result == 3

    def test_text_offset_all_context_empty_continuation(self):
        """All tokens belong to context (empty continuation) → returns len(text_offset)."""
        result = find_continuation_start([0, 1, 2, 3], "ABCD")
        assert result == 4

    def test_no_offset_returns_zero(self):
        assert find_continuation_start(None, "ctx") == 0
        assert find_continuation_start([], "ctx") == 0


# ---------------------------------------------------------------------------
# 2. check_argmax
# ---------------------------------------------------------------------------


class TestCheckArgmax:
    def test_all_continuation_tokens_match_top1(self):
        tokens = ["ctx", " A", " B", "_gen"]
        top_logprobs = [None, {" A": -0.5}, {" B": -0.3}, {"x": -0.1}]
        assert check_argmax(tokens, top_logprobs, 1) is True

    def test_first_continuation_token_does_not_match(self):
        tokens = ["ctx", " A", "_gen"]
        top_logprobs = [None, {" Z": -0.1}, {"x": -0.1}]
        assert check_argmax(tokens, top_logprobs, 1) is False

    def test_actual_token_present_as_extra_entry(self):
        # vLLM includes the actual token alongside the top-1 entry; the top-1
        # is the highest-logprob entry, not the first key.
        tokens = ["ctx", " A", "_gen"]
        top_logprobs = [None, {" A": -2.0, " Z": -0.1}, {"x": -0.1}]
        assert check_argmax(tokens, top_logprobs, 1) is False

    def test_empty_continuation_returns_true(self):
        assert check_argmax(["a", "b"], [None, {}], 1) is True

    def test_empty_top_logprobs_returns_false(self):
        assert check_argmax(["a", "b", "c"], [], 0) is False

    def test_none_top_dict_at_position_returns_false(self):
        tokens = ["ctx", " A", "_gen"]
        assert check_argmax(tokens, [None, None, {"x": -0.1}], 1) is False


# ---------------------------------------------------------------------------
# 3. pieces_to_text
# ---------------------------------------------------------------------------


class TestPiecesToText:
    def test_plain_ascii_identity(self):
        assert pieces_to_text(["Hello", " world"]) == "Hello world"

    def test_sentencepiece_underscore_becomes_space(self):
        assert pieces_to_text(["▁Привіт", "е"]) == " Привіте"

    def test_sentencepiece_byte_fallback_pieces(self):
        # "<0xD0><0x90>" is the UTF-8 byte pair for Cyrillic "А"
        assert pieces_to_text(["<0xD0>", "<0x90>"]) == "А"

    def test_byte_level_bpe_space_and_newline_markers(self):
        # GPT-2 byte alphabet: "Ġ" == space, "Ċ" == newline
        assert pieces_to_text(["Ġyes", "Ċ"]) == " yes\n"

    def test_byte_level_bpe_utf8_split_across_pieces(self):
        # Cyrillic "Ї" (U+0407) is 0xD0 0x87 in UTF-8; byte-level BPE can put
        # those bytes in separate pieces — they must decode as one character.
        byte_decoder = gpt2_byte_decoder()
        byte_encoder = {b: c for c, b in byte_decoder.items()}
        pieces = [byte_encoder[0xD0], byte_encoder[0x87]]
        assert pieces_to_text(pieces) == "Ї"

    def test_empty_pieces(self):
        assert pieces_to_text([]) == ""


# ---------------------------------------------------------------------------
# 4. align_continuation_suffix
# ---------------------------------------------------------------------------


class TestAlignContinuationSuffix:
    def test_exact_suffix(self):
        pieces = ["<start>", "user", "\n", "Question", "<end>", "▁A"]
        assert align_continuation_suffix(pieces, " A") == 5

    def test_multi_token_continuation(self):
        pieces = ["ctx", "▁The", "▁answer", "▁is", "▁B"]
        assert align_continuation_suffix(pieces, " answer is B") == 2

    def test_whitespace_absorbed_into_continuation(self):
        # Mirrors tok_encode_pair: whitespace between context and continuation
        # belongs to the continuation (e.g. the newline after "<start_of_turn>model").
        pieces = ["<start_of_turn>", "model", "\n", "▁A"]
        assert align_continuation_suffix(pieces, " A") == 2

    def test_no_whitespace_absorption_past_non_whitespace(self):
        pieces = ["Answer:", "▁B"]
        assert align_continuation_suffix(pieces, " B") == 1

    def test_mismatched_suffix_returns_none(self):
        pieces = ["ctx", "▁X"]
        assert align_continuation_suffix(pieces, "completely different") is None

    def test_empty_continuation_scores_nothing(self):
        pieces = ["ctx", "tail"]
        assert align_continuation_suffix(pieces, "") == 2

    def test_continuation_spanning_boundary_token(self):
        # The whole continuation is inside the last token together with
        # context characters — the spanning token is scored.
        pieces = ["Answer", ":▁A"]
        assert align_continuation_suffix(pieces, " A") == 1

    def test_byte_level_bpe_cyrillic_continuation(self):
        byte_decoder = gpt2_byte_decoder()
        byte_encoder = {b: c for c, b in byte_decoder.items()}

        def encode_piece(text: str) -> str:
            return "".join(byte_encoder[b] for b in text.encode("utf-8"))

        pieces = ["context", encode_piece(" Від"), encode_piece("повідь")]
        assert align_continuation_suffix(pieces, " Відповідь") == 1


class TestAlignContinuationSuffixTrimmedTemplates:
    def test_template_trimmed_leading_space(self):
        # Gemma-style templates `| trim` the assistant content: " A" renders as
        # "A" after the template's own "\n". The boundary newline is absorbed,
        # the trimmed choice is matched.
        pieces = ["<start_of_turn>", "model", "\n", "A"]
        assert align_continuation_suffix(pieces, " A") == 2

    def test_verbatim_match_preferred_over_stripped(self):
        # When the verbatim continuation IS the suffix, the stripped retry
        # must not shrink the match.
        pieces = ["Answer:", "▁A"]
        assert align_continuation_suffix(pieces, " A") == 1

    def test_trimmed_multiword_cyrillic(self):
        pieces = ["model", "\n", "ґрунт", ",", "▁вода"]
        assert align_continuation_suffix(pieces, " ґрунт, вода") == 1


# ---------------------------------------------------------------------------
# 5. extract_prompt_logprob_entries
# ---------------------------------------------------------------------------


def make_prompt_logprobs_payload(entries, first_none=True):
    """Build the ``prompt_logprobs`` positions list as vLLM serializes it."""
    positions = [None] if first_none else []
    for token_id, (decoded, logprob, rank) in enumerate(entries, start=1000):
        position = {str(token_id): {"logprob": logprob, "rank": rank, "decoded_token": decoded}}
        if rank != 1:
            # vLLM also includes the top-1 entry when the actual token is not top-1.
            position[str(token_id + 100000)] = {"logprob": logprob + 0.5, "rank": 1, "decoded_token": "«top»"}
        positions.append(position)
    return positions


class TestExtractPromptLogprobEntries:
    def test_leading_none_skipped(self):
        positions = make_prompt_logprobs_payload([("▁A", -0.5, 1)])
        assert extract_prompt_logprob_entries(positions) == [("▁A", -0.5, 1)]

    def test_actual_token_is_max_rank_entry(self):
        # rank 3 actual token + rank 1 top entry: the actual token must win.
        positions = make_prompt_logprobs_payload([("▁B", -2.0, 3)])
        assert extract_prompt_logprob_entries(positions) == [("▁B", -2.0, 3)]

    def test_missing_prompt_logprobs_returns_none(self):
        assert extract_prompt_logprob_entries(None) is None
        assert extract_prompt_logprob_entries([]) is None


class TestAlignContinuationSuffixLatin1Punctuation:
    def test_sentencepiece_guillemets_align_via_literal_mode(self):
        # "«" (U+00AB) is inside the GPT-2 byte alphabet, so a literal
        # SentencePiece "«" piece is indistinguishable from a byte piece by
        # inspection — byte-mode decodes it to a lone invalid byte. The
        # literal-mode retry must align it.
        pieces = ["model", "\n", "▁Не", "▁«", "отримайте", "»", "▁це"]
        # index 1: the boundary "\n" is absorbed by the whitespace walk,
        # exactly like the plain-ASCII cases.
        assert align_continuation_suffix(pieces, " Не «отримайте» це") == 1

    def test_byte_bpe_still_aligns_in_byte_mode(self):
        byte_decoder = gpt2_byte_decoder()
        byte_encoder = {b: c for c, b in byte_decoder.items()}

        def encode_piece(text: str) -> str:
            return "".join(byte_encoder[b] for b in text.encode("utf-8"))

        pieces = ["ctx", encode_piece(" «так»")]
        assert align_continuation_suffix(pieces, " «так»") == 1

    def test_pieces_to_text_literal_mode_keeps_guillemets(self):
        assert pieces_to_text(["▁«", "слово", "»"], byte_mode=False) == " «слово»"
