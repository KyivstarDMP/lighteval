# MIT License
#
# Copyright (c) 2026 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lighteval.data import GenerativeTaskDataset
from lighteval.models.endpoints.litellm_model import LiteLLMClient
from lighteval.models.model_input import GenerationParameters
from lighteval.tasks.requests import Doc


pytest.importorskip("litellm")


def _build_client(model_name: str, *, concurrent_requests: int = 10, api_max_retry: int = 3) -> LiteLLMClient:
    client = LiteLLMClient.__new__(LiteLLMClient)
    client.model = model_name
    client.provider = "openai"
    client.base_url = None
    client.api_key = None
    client.generation_parameters = GenerationParameters()
    client._max_length = 10_000
    client.API_MAX_RETRY = api_max_retry
    client.API_RETRY_SLEEP = 0.0
    client.API_RETRY_MULTIPLIER = 1.0
    client.timeout = None
    client.concurrent_requests = concurrent_requests
    return client


def _make_doc(query: str, stop_sequences: list[str]) -> Doc:
    return Doc(
        query=query,
        choices=[""],
        gold_index=0,
        generation_size=8,
        stop_sequences=stop_sequences,
        use_logits=False,
        num_samples=1,
    )


def _mock_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=None))],
    )


def test_greedy_until_uses_split_local_contexts():
    docs = [
        _make_doc("alpha", ["A"]),
        _make_doc("beta", ["A"]),
        _make_doc("gamma", ["B"]),
    ]

    expected_dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=LiteLLMClient.DATASET_SPLITS)
    expected_contexts_by_split = [
        [f"ctx:{doc.query}" for doc in split] for split in expected_dataset.splits_iterator()
    ]

    model = LiteLLMClient.__new__(LiteLLMClient)
    model._cache = None
    model.generation_parameters = SimpleNamespace(temperature=1, max_new_tokens=None)
    model.prompt_manager = SimpleNamespace(prepare_prompt_api=lambda doc: f"ctx:{doc.query}")

    observed_contexts_by_split: list[list[str]] = []

    async def fake_call_api_parallel(contexts, return_logits, max_new_tokens, num_samples, stop_sequence):
        observed_contexts_by_split.append(list(contexts))
        return [_mock_response(f"out-{index}") for index, _ in enumerate(contexts)]

    model._LiteLLMClient__call_api_parallel = fake_call_api_parallel

    results = model.greedy_until(docs)

    assert observed_contexts_by_split == expected_contexts_by_split
    assert sum(len(contexts) for contexts in observed_contexts_by_split) == len(docs)
    assert len(results) == len(docs)


def test_greedy_until_uses_global_event_loop():
    """greedy_until must drive every dataset split through ONE asyncio.run so litellm's
    per-loop HTTP client pool is reused rather than rebuilt per split. It must also restore
    the original doc order after the internal length/param sort."""
    client = _build_client("openai/gpt-4.1-nano", concurrent_requests=4)
    client._cache = None  # None disables the @cached wrapper so the real body runs
    client.prompt_manager = SimpleNamespace(
        prepare_prompt_api=lambda doc: [{"role": "user", "content": doc.query}]
    )

    # Differing generation_size groups docs into separate generative splits. Input order is
    # shuffled relative to the sort key so get_original_order has real work to undo.
    docs = [
        Doc(query="long a", choices=[], gold_index=0, generation_size=16, task_name="t"),
        Doc(query="short a", choices=[], gold_index=0, generation_size=8, task_name="t"),
        Doc(query="long b", choices=[], gold_index=0, generation_size=16, task_name="t"),
        Doc(query="short b", choices=[], gold_index=0, generation_size=8, task_name="t"),
    ]
    # Guard: the scenario must genuinely span >1 split, else the invariant is vacuous.
    dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=LiteLLMClient.DATASET_SPLITS)
    assert dataset.num_dataset_splits >= 2

    loop_ids = set()

    async def fake_acompletion(**kwargs):
        loop_ids.add(id(asyncio.get_running_loop()))
        return _mock_response(kwargs["messages"][-1]["content"])  # echo the query back

    run_calls = []
    real_asyncio_run = asyncio.run

    def counting_run(coro):
        run_calls.append(1)
        return real_asyncio_run(coro)

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.acompletion", side_effect=fake_acompletion):
            with patch("lighteval.models.endpoints.litellm_model.asyncio.run", side_effect=counting_run):
                responses = client.greedy_until(docs)

    # Exactly one event loop for all splits (a loop-per-split refactor would make this >1).
    assert run_calls == [1]
    assert len(loop_ids) == 1
    # Original order preserved despite the internal sort that grouped by generation_size.
    assert [response.text[0] for response in responses] == ["long a", "short a", "long b", "short b"]
