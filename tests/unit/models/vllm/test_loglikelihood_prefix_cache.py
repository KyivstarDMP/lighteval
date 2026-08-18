# MIT License

# Copyright (c) 2026 The HuggingFace Team

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

"""Prefix-KV-cache reuse for log-likelihood scoring (``VLLMModel``): validity boundary,
sibling recovery, targeted re-issue, preemption fallback and the end-to-end request path — all
against synthetic ``RequestOutput``-like objects (no GPU, no vLLM engine)."""

import logging
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import lighteval.models.vllm.vllm_model as vllm_model_module
from lighteval.models.vllm.vllm_model import (
    LoglikelihoodOutput,
    VLLMModel,
    _resolve_prefix_cached_prompt_logprobs,
    _score_loglikelihood_requests,
    _valid_prompt_logprob_positions,
)
from lighteval.tasks.requests import Doc


# --------------------------------------------------------------------------------------------------
# Synthetic vLLM objects
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeLogprob:
    logprob: float
    rank: int


GARBAGE = FakeLogprob(logprob=-1234.5, rank=1)


def true_logprob(ids: list[int], pos: int) -> float:
    """Deterministic 'model': the logprob of token ``ids[pos]`` given ``ids[:pos]`` depends on the
    whole prefix, so it is only equal across requests whose prefixes are equal up to ``pos``."""
    return -float((sum((i + 1) * t for i, t in enumerate(ids[: pos + 1])) % 97) + 1) / 8.0


def true_rank(ids: list[int], pos: int) -> int:
    return 1 if (sum(ids[: pos + 1]) % 3) else 2


class FakeOutput:
    """Mimics ``vllm.RequestOutput`` for a prompt-logprobs request.

    Positions ``<= num_cached_tokens`` are *garbage* (as vLLM's uninitialised buffer would be): the
    actual token is absent everywhere except, when ``poison_boundary`` is set, at position
    ``num_cached_tokens`` itself, where the actual token IS present but with a wrong value — a
    stale-buffer scenario that a token-presence check alone would not catch.
    """

    def __init__(self, ids: list[int], num_cached_tokens: int | None, poison_boundary: bool = True):
        self.prompt_token_ids = list(ids)
        self.num_cached_tokens = num_cached_tokens
        cached = num_cached_tokens or 0
        self.prompt_logprobs: list = [None]
        for pos in range(1, len(ids)):
            if pos <= cached:
                entry = {10_000_000 + pos: GARBAGE}
                if poison_boundary and pos == cached:
                    entry[ids[pos]] = FakeLogprob(logprob=-99.0, rank=1)  # present but WRONG
                self.prompt_logprobs.append(entry)
            else:
                self.prompt_logprobs.append(
                    {
                        ids[pos]: FakeLogprob(true_logprob(ids, pos), true_rank(ids, pos)),
                        20_000_000 + pos: FakeLogprob(-0.01, 1),  # the "top-1" token
                    }
                )


def continuation_values(resolved: list, ids: list[int], cont_len: int) -> list[float]:
    return [resolved[pos][ids[pos]].logprob for pos in range(len(ids) - cont_len, len(ids))]


def expected_values(ids: list[int], cont_len: int) -> list[float]:
    return [true_logprob(ids, pos) for pos in range(len(ids) - cont_len, len(ids))]


CTX = list(range(100, 116))  # 16 context tokens


# --------------------------------------------------------------------------------------------------
# Validity boundary
# --------------------------------------------------------------------------------------------------


class TestValidPositions:
    def test_zero_cached_everything_but_position_zero_is_valid(self):
        out = FakeOutput(CTX + [1, 2], num_cached_tokens=0)
        assert _valid_prompt_logprob_positions(out) == range(1, 18)

    def test_position_equal_to_num_cached_tokens_is_invalid(self):
        """The row for position p is produced by forwarding token p-1; token ``cached-1`` is the last
        cached one, so row ``cached`` (position == num_cached_tokens) is never written."""
        out = FakeOutput(CTX + [1, 2], num_cached_tokens=16)
        valid = _valid_prompt_logprob_positions(out)
        assert 16 not in valid
        assert 17 in valid
        assert valid == range(17, 18)

    def test_missing_metadata_is_untrusted(self):
        out = FakeOutput(CTX + [1], num_cached_tokens=None)
        assert _valid_prompt_logprob_positions(out) is None
        out = FakeOutput(CTX + [1], num_cached_tokens=0)
        out.prompt_logprobs = out.prompt_logprobs[:-1]  # length mismatch
        assert _valid_prompt_logprob_positions(out) is None
        out = FakeOutput(CTX + [1], num_cached_tokens=0)
        out.prompt_logprobs = None
        assert _valid_prompt_logprob_positions(out) is None


# --------------------------------------------------------------------------------------------------
# Resolution: own values / sibling recovery / re-issue
# --------------------------------------------------------------------------------------------------


class TestResolvePrefixCachedPromptLogprobs:
    def test_zero_cached_uses_own_values_verbatim(self):
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 0)]
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert continuation_values(resolved[0], ids_a, 2) == expected_values(ids_a, 2)
        assert continuation_values(resolved[1], ids_b, 2) == expected_values(ids_b, 2)
        # exact same objects as vLLM returned, no arithmetic in between
        assert resolved[0][16] is outs[0].prompt_logprobs[16]
        assert stats["recovered_positions"] == 0 and stats["reissued_requests"] == 0
        assert stats["prompt_tokens"] == 36 and stats["cached_tokens"] == 0

    def test_cached_context_only_all_continuation_positions_valid(self):
        """Cache hit ends inside the context (cached < continuation start): nothing to repair."""
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 12)]  # 12 <= 15 < cont_start=16
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert continuation_values(resolved[1], ids_b, 2) == expected_values(ids_b, 2)
        assert stats["recovered_positions"] == 0 and stats["reissued_requests"] == 0
        assert stats["cached_tokens"] == 12

    def test_shared_head_of_continuation_is_recovered_from_sibling(self):
        """Cached range covers the shared first continuation token: take the sibling's value."""
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 16)]  # position 16 ('7') invalid for b
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert resolved[1] is not None
        assert continuation_values(resolved[1], ids_b, 2) == expected_values(ids_b, 2)
        assert resolved[1][16] is outs[0].prompt_logprobs[16]  # the sibling's dict object
        assert resolved[1][17] is outs[1].prompt_logprobs[17]  # own value where forwarded
        assert stats == {
            "recovered_positions": 1,
            "recovered_requests": 1,
            "recovered_docs": 1,
            "reissued_requests": 0,
            "reissued_docs": 0,
            "prompt_tokens": 36,
            "cached_tokens": 16,
        }

    def test_off_by_one_boundary_position_never_uses_own_poisoned_value(self):
        """Position == num_cached_tokens carries the actual token with a wrong value in the fake
        (stale buffer); it must be treated as invalid and recovered, not trusted."""
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 16, poison_boundary=True)]
        assert outs[1].prompt_logprobs[16][7].logprob == -99.0  # the trap is armed
        resolved, _ = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert resolved[1][16][7].logprob == true_logprob(ids_b, 16)
        assert resolved[1][16][7].logprob != -99.0

    def test_boundary_at_first_differing_token_must_reissue(self):
        """Cache hit ends exactly on the option-specific token: no sibling shares that position."""
        ids_a, ids_b = CTX + [1], CTX + [2]  # differ at position 16 (block-aligned)
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 16)]
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [1, 1])
        assert resolved[0] is not None
        assert resolved[1] is None
        assert stats["reissued_requests"] == 1 and stats["reissued_docs"] == 1
        assert stats["recovered_positions"] == 0

    def test_option_specific_tail_below_boundary_must_reissue_even_if_head_recoverable(self):
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 17)]  # 16 recoverable, 17 (own '2') is not
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert resolved[1] is None
        assert stats["reissued_requests"] == 1 and stats["recovered_positions"] == 0

    def test_all_siblings_cached_at_shared_position_must_reissue(self):
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 16), FakeOutput(ids_b, 16)]  # nobody forwarded position 16
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert resolved == [None, None]
        assert stats["reissued_requests"] == 2 and stats["reissued_docs"] == 1

    def test_missing_actual_token_at_valid_position_falls_back_instead_of_keyerror(self):
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 0)]
        del outs[1].prompt_logprobs[17][2]  # actual token gone from a 'valid' position
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert resolved[0] is not None and resolved[1] is None
        assert stats["reissued_requests"] == 1

    def test_missing_actual_token_in_donor_is_not_used(self):
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 16)]
        del outs[0].prompt_logprobs[16][7]  # the would-be donor lacks the token at position 16
        # a scores only position 17 (cont_len 1), so a itself stays valid; b needs 16 from a
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [1, 2])
        assert resolved[0] is not None
        assert resolved[1] is None and stats["reissued_requests"] == 1

    def test_untrusted_metadata_reissues(self):
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, None)]
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 2])
        assert resolved[1] is None and stats["reissued_requests"] == 1

    def test_option_that_is_a_prefix_of_another_option(self):
        """Choices 'X' and 'X Y': the short one can be fully served from the long one's blocks
        (cached == n-1, i.e. its only continuation position is invalid) and is recovered from it."""
        ids_long, ids_short = CTX + [7, 8], CTX + [7]
        outs = [FakeOutput(ids_long, 0), FakeOutput(ids_short, 16)]
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [2, 1])
        assert continuation_values(resolved[1], ids_short, 1) == expected_values(ids_short, 1)
        assert resolved[1][16] is outs[0].prompt_logprobs[16]
        assert stats["recovered_positions"] == 1 and stats["reissued_requests"] == 0
        # the other way round: the long one hits the short one's blocks -> its head is recovered,
        # its tail (position 17) was forwarded (cached=16 < 17)
        outs = [FakeOutput(ids_short, 0), FakeOutput(ids_long, 16)]
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2)], [1, 2])
        assert continuation_values(resolved[1], ids_long, 2) == expected_values(ids_long, 2)
        assert stats["recovered_positions"] == 1

    def test_pairwise_prefix_beats_all_sibling_prefix(self):
        """Three options; two share a longer head than the third: recovery uses the pairwise prefix."""
        ids_a, ids_b, ids_c = CTX + [7, 8, 1], CTX + [7, 8, 2], CTX + [3]
        outs = [FakeOutput(ids_c, 0), FakeOutput(ids_a, 0), FakeOutput(ids_b, 17)]  # b: 16, 17 invalid
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 3)], [1, 3, 3])
        assert continuation_values(resolved[2], ids_b, 3) == expected_values(ids_b, 3)
        assert resolved[2][16] is outs[1].prompt_logprobs[16] and resolved[2][17] is outs[1].prompt_logprobs[17]
        assert stats["recovered_positions"] == 2 and stats["reissued_requests"] == 0

    def test_recovery_never_crosses_documents(self):
        """Two docs with token-identical prompts (think: same text, different images): the second
        doc's cached option must be re-issued, never filled from the first doc."""
        ids_a, ids_b = CTX + [7, 1], CTX + [7, 2]
        outs = [FakeOutput(ids_a, 0), FakeOutput(ids_b, 0), FakeOutput(ids_a, 16), FakeOutput(ids_b, 16)]
        resolved, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 2), (2, 4)], [2, 2, 2, 2])
        assert resolved[0] is not None and resolved[1] is not None
        assert resolved[2] is None and resolved[3] is None
        assert stats["reissued_docs"] == 1 and stats["reissued_requests"] == 2

    def test_stats_count_docs_and_requests(self):
        ids = [CTX + [7, 1], CTX + [7, 2], CTX + [7, 3]]
        outs = [FakeOutput(ids[0], 0), FakeOutput(ids[1], 16), FakeOutput(ids[2], 16)]  # doc 1: 2 recovered
        outs += [FakeOutput(CTX + [1], 0), FakeOutput(CTX + [2], 16)]  # doc 2: 1 reissue
        _, stats = _resolve_prefix_cached_prompt_logprobs(outs, [(0, 3), (3, 5)], [2, 2, 2, 1, 1])
        assert stats["recovered_requests"] == 2 and stats["recovered_docs"] == 1
        assert stats["reissued_requests"] == 1 and stats["reissued_docs"] == 1


# --------------------------------------------------------------------------------------------------
# Two-pass driver: first pass with cache reading, targeted re-issue, preemption fallback
# --------------------------------------------------------------------------------------------------


class TestScoreLoglikelihoodRequests:
    @staticmethod
    def _engine(cached_by_request: dict[int, int], calls: list):
        """A ``generate`` stand-in; when reading the cache, request i has ``cached_by_request[i]``
        cached tokens (garbage below), otherwise everything is computed."""

        def generate(inputs, read_prefix_cache):
            calls.append((list(inputs), read_prefix_cache))
            if read_prefix_cache:
                return [FakeOutput(ids, cached_by_request.get(i, 0)) for i, ids in enumerate(inputs)]
            return [FakeOutput(ids, 0) for ids in inputs]

        return generate

    def test_reissues_only_unresolved_requests_and_results_match_safe_path(self):
        inputs = [CTX + [7, 1], CTX + [7, 2], CTX + [1], CTX + [2]]
        doc_slices, cont_lens = [(0, 2), (2, 4)], [2, 2, 1, 1]
        calls = []
        generate = self._engine({1: 16, 3: 16}, calls)  # doc 0: recoverable; doc 1: must re-issue

        results, stats = _score_loglikelihood_requests(generate, inputs, doc_slices, cont_lens, lambda: 0)

        assert [c[1] for c in calls] == [True, False]
        assert calls[1][0] == [CTX + [2]]  # only the unresolved request was re-issued
        assert all(isinstance(r, LoglikelihoodOutput) for r in results)
        for r, ids, n in zip(results, inputs, cont_lens):
            assert r.prompt_token_ids == ids
            assert continuation_values(r.prompt_logprobs, ids, n) == expected_values(ids, n)
        assert stats["recovered_requests"] == 1 and stats["reissued_requests"] == 1
        assert stats["preemptions"] == 0 and stats["fallback"] is None

    def test_no_second_pass_when_everything_resolves(self):
        inputs = [CTX + [7, 1], CTX + [7, 2]]
        calls = []
        results, stats = _score_loglikelihood_requests(
            self._engine({1: 16}, calls), inputs, [(0, 2)], [2, 2], lambda: 5
        )
        assert [c[1] for c in calls] == [True]
        assert stats["reissued_requests"] == 0 and len(results) == 2

    def test_preemption_during_first_pass_reissues_everything(self):
        inputs = [CTX + [7, 1], CTX + [7, 2]]
        calls = []
        counter = iter([3, 4])  # one preemption happened between the two reads
        results, stats = _score_loglikelihood_requests(
            self._engine({1: 16}, calls), inputs, [(0, 2)], [2, 2], lambda: next(counter)
        )
        assert [c[1] for c in calls] == [True, False]
        assert calls[1][0] == inputs
        assert stats["preemptions"] == 1 and "preemption" in stats["fallback"]
        assert stats["reissued_requests"] == 2 and stats["recovered_requests"] == 0
        for r, ids in zip(results, inputs):
            assert continuation_values(r.prompt_logprobs, ids, 2) == expected_values(ids, 2)

    def test_unavailable_preemption_counter_reissues_everything(self):
        inputs = [CTX + [7, 1], CTX + [7, 2]]
        calls = []
        _, stats = _score_loglikelihood_requests(self._engine({}, calls), inputs, [(0, 2)], [2, 2], lambda: None)
        assert [c[1] for c in calls] == [True, False]
        assert stats["fallback"] == "preemption counter unavailable" and stats["reissued_requests"] == 2

    def test_empty_input(self):
        results, stats = _score_loglikelihood_requests(lambda *_: [], [], [], [], lambda: 0)
        assert results == [] and stats["requests"] == 0


# --------------------------------------------------------------------------------------------------
# End-to-end through ``VLLMModel._loglikelihood_tokens`` with a fake engine that models vLLM's
# block-aligned prefix cache (blocks written by every request, read only when asked, garbage rows
# at positions <= num_cached_tokens). Cache-reading path must reproduce the safe path exactly.
# --------------------------------------------------------------------------------------------------


class CharTokenizer:
    """One token per character: keeps prompt construction deterministic and offline."""

    eos_token_id = -1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.skip_reading_prefix_cache = kwargs.get("skip_reading_prefix_cache", True)


class FakeCachingLLM:
    """Simulates the parts of vLLM that matter: full-block prefix hashes are cached as soon as a
    request is scheduled (so later requests in the same call can hit them), reads are block-aligned
    and capped at n-1, and prompt logprobs exist only for forwarded positions."""

    def __init__(self, block_size: int, preemptions: int = 0):
        self.block_size = block_size
        self.blocks: set = set()
        self.calls: list = []
        self._preemptions = preemptions
        self.metrics_calls = 0

    def get_metrics(self):
        self.metrics_calls += 1
        return [SimpleNamespace(name="vllm:num_preemptions", value=self._preemptions)]

    def _hit(self, ids: list[int]) -> int:
        hit = 0
        for end in range(self.block_size, len(ids), self.block_size):  # full blocks, <= n-1
            if tuple(ids[:end]) in self.blocks:
                hit = end
            else:
                break
        return hit

    def _write(self, ids: list[int]) -> None:
        for end in range(self.block_size, len(ids) + 1, self.block_size):
            self.blocks.add(tuple(ids[:end]))

    def generate(self, prompts, sampling_params, use_tqdm=True):
        read = not sampling_params.skip_reading_prefix_cache
        self.calls.append(("read" if read else "safe", len(prompts)))
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            cached = self._hit(ids) if read else 0
            self._write(ids)
            outputs.append(FakeOutput(ids, cached))
        return outputs


def _fake_vllm_modules():
    fake_inputs = ModuleType("vllm.inputs")
    fake_inputs.TokensPrompt = lambda *, prompt_token_ids: {"prompt_token_ids": prompt_token_ids}  # noqa: E731
    fake_vllm = ModuleType("vllm")
    fake_vllm.inputs = fake_inputs
    return {"vllm": fake_vllm, "vllm.inputs": fake_inputs}


def _bare_model(llm, prefix_cache: bool) -> VLLMModel:
    model = VLLMModel.__new__(VLLMModel)
    model.DATASET_SPLITS = 1
    model.use_chat_template = True
    model.prompt_manager = Mock()
    model.prompt_manager.prepare_prompt.side_effect = lambda doc: doc.query
    model._tokenizer = CharTokenizer()
    model._add_special_tokens = False
    model.pairwise_tokenization = False
    model.config = Mock()
    model.config.loglikelihood_prefix_cache = prefix_cache
    model.config.loglikelihood_multimodal_token_prompts = False
    model.data_parallel_size = 1
    model.model = llm
    return model


def _docs() -> list[Doc]:
    # Block size 4. Contexts start with different characters so no cross-doc block is ever shared
    # (as in real data, where documents diverge long before their continuations); lengths are chosen
    # so that the doc set exercises every branch:
    #  - "abcdefgh" (8 = aligned): the shared '\n' head of the continuation is served from the cache
    #    for options 2 and 3 -> sibling recovery
    #  - "ijklmno" (7): the hit ends inside the context (4 < 7) -> everything valid, no repair
    #  - "pqrstuvwxyz0" (12 = aligned): options differ at position 12 = cache boundary -> re-issue
    #  - "ABCDEFG" (7) with choices " XYZ0" and " X": the short option (9 tokens) is a prefix of the
    #    long one and hits its blocks up to position 8 = n-1 -> both scored positions of the short
    #    option are recovered from the long sibling
    return [
        Doc(query="abcdefgh", choices=["\nA", "\nB", "\nC"], gold_index=0),
        Doc(query="ijklmno", choices=["A", "B"], gold_index=1),
        Doc(query="pqrstuvwxyz0", choices=["A", "B", "C", "D"], gold_index=2),
        Doc(query="ABCDEFG", choices=[" XYZ0", " X"], gold_index=0),
    ]


class TestLoglikelihoodTokensEndToEnd:
    def _run(self, prefix_cache: bool, block_size: int = 4, preemptions: int = 0):
        llm = FakeCachingLLM(block_size, preemptions)
        model = _bare_model(llm, prefix_cache)
        with (
            patch.dict(sys.modules, _fake_vllm_modules()),
            patch.object(vllm_model_module, "SamplingParams", FakeSamplingParams),
        ):
            responses = model._loglikelihood_tokens(_docs())
        return responses, llm

    def test_cache_reading_path_reproduces_safe_path_exactly(self, caplog):
        caplog.set_level(logging.INFO, logger=vllm_model_module.__name__)
        safe, safe_llm = self._run(prefix_cache=False)
        fast, fast_llm = self._run(prefix_cache=True)

        assert safe_llm.calls == [("safe", 11)]
        assert fast_llm.calls[0] == ("read", 11)
        # every branch was exercised: some tokens came from cache, one doc needed re-issue
        assert fast_llm.calls[1][0] == "safe" and 0 < fast_llm.calls[1][1] < 11
        assert fast_llm.metrics_calls == 2

        assert len(safe) == len(fast) == 4
        for s, f in zip(safe, fast):
            assert f.logprobs == s.logprobs  # bit-identical sums (same objects, same order)
            assert f.argmax_logits_eq_gold == s.argmax_logits_eq_gold
            assert f.input_tokens == s.input_tokens and f.output_tokens == s.output_tokens

        # ground truth from the deterministic 'model'
        for response, doc in zip(safe, _docs()):
            for logprob, choice in zip(response.logprobs, doc.choices):
                ids = [ord(c) for c in doc.query + choice]
                assert logprob == sum(expected_values(ids, len(choice)))

        summary = [r for r in caplog.records if "Loglikelihood prefix-cache reuse" in r.getMessage()]
        assert len(summary) == 1
        message = summary[0].getMessage()
        assert "11 requests / 4 docs" in message
        assert "sibling recovery on 3 requests / 2 docs (4 positions)" in message  # doc 0: 2 opts, doc 3: 1
        assert "re-issued 3 requests / 1 docs" in message  # doc 2: options B, C, D
        assert "0 preemption(s)" in message

    def test_disabled_flag_uses_legacy_generate_path(self):
        model = _bare_model(FakeCachingLLM(4), prefix_cache=False)
        captured = {}

        def fake_generate(inputs, generate=True):
            captured["generate"] = generate
            return [FakeOutput(ids, 0) for ids in inputs]

        model._generate = fake_generate
        with patch.dict(sys.modules, _fake_vllm_modules()):
            model._loglikelihood_tokens(_docs()[:1])
        assert captured["generate"] is False

    def test_preemption_forces_full_safe_reissue(self, caplog):
        caplog.set_level(logging.WARNING, logger=vllm_model_module.__name__)
        llm = FakeCachingLLM(4)
        counter = iter([0, 2, 2, 2])
        llm.get_metrics = lambda: [SimpleNamespace(name="vllm:num_preemptions", value=next(counter))]
        model = _bare_model(llm, prefix_cache=True)
        with (
            patch.dict(sys.modules, _fake_vllm_modules()),
            patch.object(vllm_model_module, "SamplingParams", FakeSamplingParams),
        ):
            fast = model._loglikelihood_tokens(_docs())
        assert llm.calls == [("read", 11), ("safe", 11)]
        assert any("Fell back to the safe path: 2 preemption(s)" in r.getMessage() for r in caplog.records)
        for response, doc in zip(fast, _docs()):
            for logprob, choice in zip(response.logprobs, doc.choices):
                ids = [ord(c) for c in doc.query + choice]
                assert logprob == sum(expected_values(ids, len(choice)))


class FakeRay:
    """Runs ``@ray.remote`` functions synchronously in-process; records the per-worker payloads."""

    def __init__(self):
        self.payloads: list = []

    def remote(self, num_gpus=None):
        outer = self

        def decorator(fn):
            class _Remote:
                def remote(self, *args):
                    outer.payloads.append(args)
                    return fn(*args)

            return _Remote()

        return decorator

    @staticmethod
    def get(refs):
        return list(refs)

    @staticmethod
    def shutdown():
        pass


class TestDataParallelSharding:
    def test_docs_are_dealt_whole_to_workers_and_results_reassembled(self, caplog):
        caplog.set_level(logging.INFO, logger=vllm_model_module.__name__)
        docs = _docs()
        engines: list[FakeCachingLLM] = []

        def fake_llm(**model_args):
            engine = FakeCachingLLM(4)
            engines.append(engine)
            return engine

        fake_ray = FakeRay()
        model = _bare_model(llm=None, prefix_cache=True)
        model.data_parallel_size = 3
        model.tensor_parallel_size = 1
        model.model_args = {"model": "x"}
        with (
            patch.dict(sys.modules, _fake_vllm_modules()),
            patch.object(vllm_model_module, "SamplingParams", FakeSamplingParams),
            patch.object(vllm_model_module, "ray", fake_ray),
            patch.object(vllm_model_module, "LLM", fake_llm),
        ):
            dp = model._loglikelihood_tokens(docs)

        single = _bare_model(FakeCachingLLM(4), prefix_cache=True)
        with (
            patch.dict(sys.modules, _fake_vllm_modules()),
            patch.object(vllm_model_module, "SamplingParams", FakeSamplingParams),
        ):
            reference = single._loglikelihood_tokens(docs)

        # 4 docs over 3 workers -> workers get docs {0,3}, {1}, {2} (dataset order), each doc whole
        assert len(fake_ray.payloads) == 3 and len(engines) == 3
        request_counts = sorted(len(payload[1]) for payload in fake_ray.payloads)
        assert sum(request_counts) == 11
        for payload in fake_ray.payloads:
            _, requests, slices, cont_lens = payload
            assert len(cont_lens) == len(requests)
            assert [end - start for start, end in slices] and slices[0][0] == 0
            assert slices[-1][1] == len(requests)
        # same numbers as the single-engine path, in the original doc order
        assert [r.logprobs for r in dp] == [r.logprobs for r in reference]
        assert [r.argmax_logits_eq_gold for r in dp] == [r.argmax_logits_eq_gold for r in reference]
        summary = [r.getMessage() for r in caplog.records if "Loglikelihood prefix-cache reuse" in r.getMessage()]
        assert any("11 requests / 4 docs" in m for m in summary)


# --------------------------------------------------------------------------------------------------
# Config / sampling params plumbing
# --------------------------------------------------------------------------------------------------


class TestPlumbing:
    def test_config_field_default_and_engine_arg(self):
        from lighteval.models.vllm.vllm_model import VLLMModelConfig

        assert VLLMModelConfig(model_name="x").loglikelihood_prefix_cache is True
        assert VLLMModelConfig(model_name="x").loglikelihood_multimodal_token_prompts is False
        assert VLLMModelConfig(model_name="x", loglikelihood_prefix_cache=False).loglikelihood_prefix_cache is False
        with pytest.raises(Exception):  # extra=forbid: typos are rejected
            VLLMModelConfig(model_name="x", loglikelihood_prefix_cach=False)

    def test_sampling_params_toggle(self):
        with patch.object(vllm_model_module, "SamplingParams", FakeSamplingParams):
            fast = VLLMModel._loglikelihood_sampling_params(read_prefix_cache=True)
            safe = VLLMModel._loglikelihood_sampling_params(read_prefix_cache=False)
        assert fast.skip_reading_prefix_cache is False
        assert safe.skip_reading_prefix_cache is True  # vLLM's default for prompt_logprobs requests
        for params in (fast, safe):
            assert (params.temperature, params.prompt_logprobs, params.max_tokens, params.detokenize) == (
                0.0,
                1,
                1,
                False,
            )

    def test_sampling_params_without_knob_degrades_to_safe(self, caplog):
        class OldSamplingParams(FakeSamplingParams):
            def __init__(self, **kwargs):
                if "skip_reading_prefix_cache" in kwargs:
                    raise TypeError("unexpected keyword argument")
                super().__init__(**kwargs)

        caplog.set_level(logging.WARNING, logger=vllm_model_module.__name__)
        with patch.object(vllm_model_module, "SamplingParams", OldSamplingParams):
            params = VLLMModel._loglikelihood_sampling_params(read_prefix_cache=True)
        assert params.skip_reading_prefix_cache is True
        assert any("without prefix-cache reuse" in r.getMessage() for r in caplog.records)

    def test_count_preemptions(self):
        llm = Mock()
        llm.get_metrics.return_value = [
            SimpleNamespace(name="vllm:num_preemptions", value=2),  # one sample per engine (DP)
            SimpleNamespace(name="vllm:num_preemptions", value=3),
            SimpleNamespace(name="vllm:prompt_tokens", value=100),
        ]
        assert VLLMModel._count_preemptions(llm) == 5
        llm.get_metrics.return_value = []
        assert VLLMModel._count_preemptions(llm) is None
        llm.get_metrics.side_effect = AssertionError("Stat logging disabled")
        assert VLLMModel._count_preemptions(llm) is None

    def test_engine_args_enable_stat_logging_only_with_prefix_cache(self):
        from lighteval.models.vllm.vllm_model import VLLMModelConfig

        for flag in (True, False):
            config = VLLMModelConfig(model_name="x", loglikelihood_prefix_cache=flag, data_parallel_size=2)
            model = VLLMModel.__new__(VLLMModel)
            model._max_length = 16
            model._create_auto_model(config)  # data_parallel_size > 1 -> records args, builds no engine
            assert model.model_args["disable_log_stats"] is (not flag)
