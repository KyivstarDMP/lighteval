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
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


pytest.importorskip("litellm")

import litellm  # noqa: E402

from lighteval.models.endpoints.litellm_model import LiteLLMClient  # noqa: E402
from lighteval.models.model_input import GenerationParameters  # noqa: E402


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


def _mock_response(content: str = "ok"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_call_api_parallel_runs_concurrently_bounded_by_semaphore():
    """Requests must genuinely overlap in flight (proving real async concurrency, not
    disguised serial execution), while never exceeding `concurrent_requests` at once."""
    concurrency = 3
    num_prompts = 9
    client = _build_client("openai/gpt-4.1-nano", concurrent_requests=concurrency)

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def fake_acompletion(**kwargs):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return _mock_response()

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.acompletion", side_effect=fake_acompletion):
            start = time.monotonic()
            results = asyncio.run(
                client._LiteLLMClient__call_api_parallel(
                    prompts=[[{"role": "user", "content": f"q{i}"}] for i in range(num_prompts)],
                    return_logits=False,
                    max_new_tokens=None,
                    num_samples=1,
                    stop_sequence=None,
                )
            )
            elapsed = time.monotonic() - start

    assert len(results) == num_prompts
    # Bounded: the semaphore must cap in-flight calls at `concurrency`.
    assert max_in_flight == concurrency
    # Genuinely concurrent: 9 requests at 0.05s each, 3 at a time, is ~3 batches (~0.15s
    # total). Fully serial execution would take ~9 batches (~0.45s). Generous margin for
    # scheduler jitter, but well below what serial execution would need.
    assert elapsed < 0.3


def test_call_api_parallel_uses_async_coroutines_not_threads():
    """Requests must run as asyncio coroutines on a single event loop in a single thread.
    The old implementation used a ThreadPoolExecutor, which would spread calls across
    worker threads and leave no event loop running inside a call. This pins that the async
    rewrite did not silently regress to thread-based parallelism."""
    client = _build_client("openai/gpt-4.1-nano", concurrent_requests=4)

    thread_ids = set()
    loop_ids = set()

    async def fake_acompletion(**kwargs):
        thread_ids.add(threading.get_ident())
        # get_running_loop() only succeeds inside a running loop; a ThreadPoolExecutor
        # worker would have none and this would raise.
        loop_ids.add(id(asyncio.get_running_loop()))
        await asyncio.sleep(0.01)
        return _mock_response()

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.acompletion", side_effect=fake_acompletion):
            results = asyncio.run(
                client._LiteLLMClient__call_api_parallel(
                    prompts=[[{"role": "user", "content": f"q{i}"}] for i in range(8)],
                    return_logits=False,
                    max_new_tokens=None,
                    num_samples=1,
                    stop_sequence=None,
                )
            )

    assert len(results) == 8
    # One thread => coroutines, not a thread pool.
    assert len(thread_ids) == 1
    # One event loop shared by every request.
    assert len(loop_ids) == 1


def test_call_api_retries_transient_errors_then_succeeds():
    client = _build_client("openai/gpt-4.1-nano", api_max_retry=4)
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient upstream error")
        return _mock_response("recovered")

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.acompletion", side_effect=fake_acompletion):
            response = asyncio.run(
                client._LiteLLMClient__call_api(
                    prompt=[{"role": "user", "content": "hi"}],
                    return_logits=False,
                    max_new_tokens=None,
                    num_samples=1,
                    stop_sequence=None,
                )
            )

    assert len(attempts) == 3
    assert response.choices[0].message.content == "recovered"


def test_call_api_exhausts_retries_and_returns_empty_response():
    client = _build_client("openai/gpt-4.1-nano", api_max_retry=3)

    async def always_fails(**kwargs):
        raise RuntimeError("persistent upstream error")

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.acompletion", side_effect=always_fails):
            response = asyncio.run(
                client._LiteLLMClient__call_api(
                    prompt=[{"role": "user", "content": "hi"}],
                    return_logits=False,
                    max_new_tokens=None,
                    num_samples=1,
                    stop_sequence=None,
                )
            )

    # LitellmModelResponse() default: one choice with empty content, not a crash.
    assert response.choices[0].message.content is None


def test_call_api_retries_without_caching_on_empty_response():
    """An empty first response (a cache hit that returned nothing) triggers exactly one
    retry with caching disabled, and the non-empty retry result is returned. This is a
    successful call, not a tenacity retry, so acompletion is hit exactly twice."""
    client = _build_client("openai/gpt-4.1-nano")
    caching_flags = []

    async def fake_acompletion(**kwargs):
        caching_flags.append(kwargs.get("caching"))
        # First attempt: cached + empty content -> must retry without caching.
        # Second attempt: uncached + real content -> returned as-is.
        return _mock_response("") if len(caching_flags) == 1 else _mock_response("filled")

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch("lighteval.models.endpoints.litellm_model.litellm.acompletion", side_effect=fake_acompletion):
            response = asyncio.run(
                client._LiteLLMClient__call_api(
                    prompt=[{"role": "user", "content": "hi"}],
                    return_logits=False,
                    max_new_tokens=None,
                    num_samples=1,
                    stop_sequence=None,
                )
            )

    # First call cached, second call retried with caching turned off.
    assert caching_flags == [True, False]
    assert response.choices[0].message.content == "filled"


def test_tenacity_wait_honors_retry_after_header_over_backoff():
    client = _build_client("openai/gpt-4.1-nano")
    client.API_RETRY_SLEEP = 1.0
    client.API_RETRY_MULTIPLIER = 2.0

    error = litellm.RateLimitError(message="rate limited", llm_provider="openai", model="gpt-4.1-nano")
    error.headers = {"retry-after": "67"}
    retry_state = SimpleNamespace(outcome=SimpleNamespace(exception=lambda: error), attempt_number=5)

    # Server-specified Retry-After (67s) wins over what backoff would otherwise compute
    assert client._tenacity_wait(retry_state) == 67.0
