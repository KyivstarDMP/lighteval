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

"""Unit tests for the vLLM OpenAI-endpoint backend.

All SDK calls are mocked — no network requests. Covers request payloads
(standard vs extra-body sampling params, vLLM LL extensions), retry
classification, chat/plain LL scoring, message construction (incl. multi-image
docs), config parsing and the cache-key exclusion of operational fields.
"""

import asyncio
import resource
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from test_openai_scoring import make_prompt_logprobs_payload

from lighteval.models.endpoints.vllm_openai_model import (
    VLLMOpenAIClient,
    VLLMOpenAIModelConfig,
    _AdaptiveLimiter,
    _collect_chat_stream,
    _fit_open_files,
    _metrics_url,
    _next_limit,
    _parse_server_load,
    _ProgressClock,
    _ServerLoad,
    _steer,
)
from lighteval.models.model_input import GenerationParameters
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.cache_management import SampleCache


def make_doc(query, choices, gold_index=0, task_name="test_task", doc_id="0", images=None, instruction=None):
    doc = Doc(query=query, choices=choices, gold_index=gold_index, task_name=task_name, instruction=instruction)
    doc.id = doc_id
    if images is not None:
        doc.images = images
    return doc


def make_client(
    model="org/model-it",
    base_url="http://localhost:8000/v1",
    use_chat_template=True,
    api_max_retry=2,
    generation_parameters=None,
    client_tokenization=False,
):
    """Construct a VLLMOpenAIClient bypassing __init__ (no cache, no SDK client)."""
    client = object.__new__(VLLMOpenAIClient)
    client.client_tokenization = client_tokenization
    client.model = model
    client.base_url = base_url
    client.api_key = None
    client.generation_parameters = generation_parameters or GenerationParameters()
    client.concurrent_requests = 4
    client.timeout = None
    client._max_length = 4096
    client.API_MAX_RETRY = api_max_retry
    client.API_RETRY_SLEEP = 0.0
    client.API_RETRY_MULTIPLIER = 1.0

    client.prompt_manager = PromptManager(use_chat_template=use_chat_template, tokenizer=None, system_prompt=None)

    client._cache = None
    return client


def make_sdk_chat_response(prompt_logprobs=None, content="ok"):
    """SimpleNamespace standing in for an SDK ChatCompletion (extra fields as attrs)."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=None))
    response = SimpleNamespace(choices=[choice])
    if prompt_logprobs is not None:
        response.prompt_logprobs = prompt_logprobs
    return response


def make_sdk_chat_stream(pieces, reasoning_pieces=None, n=1, finish_reason="stop"):
    """Async iterator standing in for an SDK chat AsyncStream: one chunk per piece, per sample index."""

    async def chunks():
        for index in range(n):
            for piece in reasoning_pieces or []:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=index,
                            finish_reason=None,
                            delta=SimpleNamespace(content=None, reasoning_content=piece),
                        )
                    ]
                )
            for piece in pieces:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=index,
                            finish_reason=None,
                            delta=SimpleNamespace(content=piece, reasoning_content=None),
                        )
                    ]
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=index,
                        finish_reason=finish_reason,
                        delta=SimpleNamespace(content=None, reasoning_content=None),
                    )
                ]
            )

    return chunks()


def make_sdk_completion_stream(pieces, n=1, finish_reason="stop"):
    """Async iterator standing in for an SDK completion AsyncStream."""

    async def chunks():
        for index in range(n):
            for piece in pieces:
                yield SimpleNamespace(choices=[SimpleNamespace(index=index, finish_reason=None, text=piece)])
            yield SimpleNamespace(choices=[SimpleNamespace(index=index, finish_reason=finish_reason, text=None)])

    return chunks()


def make_sdk_completion_response(tokens, token_logprobs, top_logprobs=None, text_offset=None, text="t"):
    """SimpleNamespace standing in for an SDK Completion (echo route)."""
    logprobs = SimpleNamespace(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs if top_logprobs is not None else [None] * len(tokens),
        text_offset=text_offset,
    )
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=logprobs, text=text)])


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Config parsing and cache key
# ---------------------------------------------------------------------------


class TestConfig:
    def test_base_url_required(self):
        with pytest.raises(Exception):
            VLLMOpenAIModelConfig(model_name="org/m")

    def test_from_args_round_trip_benchcore_shape(self):
        # The exact inline shape benchcore emits for local models.
        args = (
            "model_name=org/model-it,use_chat_template=true,concurrent_requests=64,timeout=600.0,"
            "max_model_length=5632,base_url=http://localhost:41321/v1,"
            "chat_template_kwargs={enable_thinking:false},generation_parameters={temperature:0,top_p:1}"
        )
        config = VLLMOpenAIModelConfig.from_args(args)
        assert config.model_name == "org/model-it"
        assert config.base_url == "http://localhost:41321/v1"
        assert config.use_chat_template is True
        assert config.concurrent_requests == 64
        assert config.timeout == 600.0
        assert config.max_model_length == 5632
        assert config.chat_template_kwargs == {"enable_thinking": False}
        assert config.generation_parameters.temperature == 0
        assert config.generation_parameters.top_p == 1

    def test_unknown_field_rejected(self):
        with pytest.raises(Exception):
            VLLMOpenAIModelConfig(model_name="org/m", base_url="http://x", provider="hosted_vllm")

    def test_cache_key_ignores_operational_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            a = VLLMOpenAIModelConfig(model_name="org/m", base_url="http://localhost:1111/v1", cache_dir=temp_dir)
            b = VLLMOpenAIModelConfig(
                model_name="org/m",
                base_url="http://localhost:2222/v1",
                timeout=5,
                concurrent_requests=99,
                api_max_retry=1,
                cache_dir=temp_dir,
            )
            cache = SampleCache(a)
            assert cache.get_model_hash(a) == cache.get_model_hash(b)

    def test_cache_key_tracks_identity_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            a = VLLMOpenAIModelConfig(model_name="org/m", base_url="http://x/v1", cache_dir=temp_dir)
            b = VLLMOpenAIModelConfig(
                model_name="org/m",
                base_url="http://x/v1",
                cache_dir=temp_dir,
                generation_parameters=GenerationParameters(temperature=0.7),
            )
            c = VLLMOpenAIModelConfig(
                model_name="org/m", base_url="http://x/v1", use_chat_template=False, cache_dir=temp_dir
            )
            cache = SampleCache(a)
            assert cache.get_model_hash(a) != cache.get_model_hash(b)
            assert cache.get_model_hash(a) != cache.get_model_hash(c)


# ---------------------------------------------------------------------------
# 2. Payload mapping
# ---------------------------------------------------------------------------


class TestPayloadMapping:
    def test_sampling_params_split_standard_vs_extra(self):
        client = make_client(
            generation_parameters=GenerationParameters(
                temperature=0.8,
                top_p=0.95,
                seed=7,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                top_k=64,
                min_p=0.05,
                repetition_penalty=1.05,
            )
        )
        standard, extra = client._sampling_params()
        assert standard == {
            "temperature": 0.8,
            "top_p": 0.95,
            "seed": 7,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        }
        assert extra == {"top_k": 64, "min_p": 0.05, "repetition_penalty": 1.05}

    def test_skip_special_tokens_false_reaches_extra_body(self):
        """False is falsy: the payload filter must key on `is not None`."""
        client = make_client(
            generation_parameters=GenerationParameters(temperature=1.0, top_k=64, skip_special_tokens=False)
        )
        _, extra = client._sampling_params()
        assert extra["skip_special_tokens"] is False

    def test_min_new_tokens_maps_to_min_tokens(self):
        client = make_client(generation_parameters=GenerationParameters(min_new_tokens=16))
        _, extra = client._sampling_params()
        assert extra["min_tokens"] == 16

    def test_unset_vllm_knobs_are_omitted(self):
        _, extra = make_client(generation_parameters=GenerationParameters(temperature=0.5))._sampling_params()
        assert "skip_special_tokens" not in extra
        assert "min_tokens" not in extra

    def test_skip_special_tokens_reaches_chat_completion_call(self):
        client = make_client(generation_parameters=GenerationParameters(temperature=1.0, skip_special_tokens=False))
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_stream(["ok"]))
        run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "q"}], 32, 1))
        assert sdk.chat.completions.create.call_args.kwargs["extra_body"]["skip_special_tokens"] is False

    def test_default_temperature_zero_is_sent(self):
        standard, extra = make_client()._sampling_params()
        assert standard == {"temperature": 0}
        assert extra == {}

    def test_chat_generative_payload(self):
        client = make_client(generation_parameters=GenerationParameters(temperature=1.0, top_k=64))
        client.prompt_manager.chat_template_kwargs = {"enable_thinking": False}
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_stream(["ok"]))

        run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 128, 1))

        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "org/model-it"
        assert kwargs["max_tokens"] == 128
        assert kwargs["stream"] is True  # the timeout bounds chunk gaps, not the whole generation
        # No stop sequences on the chat route: chat models terminate via EOS
        # (in-process VLLMModel parity; a reasoning model opening "<think>\n"
        # must not be cut by a task's "\n" stop marker).
        assert "stop" not in kwargs
        assert kwargs["temperature"] == 1.0
        # vLLM-only knobs + template kwargs ride in extra_body.
        assert kwargs["extra_body"] == {"top_k": 64, "chat_template_kwargs": {"enable_thinking": False}}

    def test_task_stop_sequences_win_over_config_stop_tokens(self):
        # Text-completion route only: the chat route never sends stop sequences.
        client = make_client(generation_parameters=GenerationParameters(stop_tokens=["cfg"]))
        assert client._stop_for(["task"]) == ["task"]
        assert client._stop_for(None) == ["cfg"]
        assert make_client()._stop_for(None) is None

    def test_text_generative_payload_uses_prompt(self):
        client = make_client(use_chat_template=False)
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(return_value=make_sdk_completion_stream(["t"]))

        run(client._call_api_text_generative(sdk, "plain prompt", 64, 1, None))

        kwargs = sdk.completions.create.call_args.kwargs
        assert kwargs["prompt"] == "plain prompt"
        assert kwargs["max_tokens"] == 64
        assert kwargs["stream"] is True
        assert "messages" not in kwargs

    def test_echo_payload_is_deterministic_scoring(self):
        client = make_client()
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(return_value=make_sdk_completion_response(["t"], [None]))

        run(client._call_api_text_completion_async(sdk, "ctx A"))

        kwargs = sdk.completions.create.call_args.kwargs
        assert kwargs["echo"] is True
        assert kwargs["logprobs"] == 1
        assert kwargs["max_tokens"] == 1
        assert kwargs["temperature"] == 0.0
        assert kwargs["prompt"] == "ctx A"

    def test_chat_ll_payload_carries_vllm_extensions(self):
        client = make_client()
        client.prompt_manager.chat_template_kwargs = {"enable_thinking": False}
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_response(prompt_logprobs=[]))

        messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": " A"}]
        run(client._call_api_chat_prompt_logprobs_async(sdk, messages))

        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "org/model-it"  # bare name — no provider prefix
        assert kwargs["max_tokens"] == 1
        assert kwargs["temperature"] == 0.0
        assert kwargs["extra_body"] == {
            "add_generation_prompt": False,
            "continue_final_message": True,
            "prompt_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        }


# ---------------------------------------------------------------------------
# 3. Retry classification
# ---------------------------------------------------------------------------


def _status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


def _connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    return openai.APIConnectionError(request=request)


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"))


class TestRetryClassification:
    def test_connect_refused_raises_after_quick_retries(self):
        client = make_client()
        call = AsyncMock(side_effect=_connection_error())
        with pytest.raises(RuntimeError, match="unreachable"):
            run(client._request(call, label="test"))
        # initial + 2 quick retries
        assert call.await_count == 3

    def test_bad_request_degrades_without_retry(self):
        client = make_client()
        call = AsyncMock(side_effect=_status_error(400))
        assert run(client._request(call, label="test")) is None
        assert call.await_count == 1

    def test_server_error_backs_off_then_degrades(self):
        client = make_client(api_max_retry=3)
        call = AsyncMock(side_effect=_status_error(500))
        assert run(client._request(call, label="test")) is None
        assert call.await_count >= 3

    def test_rate_limit_backs_off_then_succeeds(self):
        client = make_client(api_max_retry=3)
        good = make_sdk_chat_response()
        call = AsyncMock(side_effect=[_status_error(429), good])
        assert run(client._request(call, label="test")) is good

    def test_timeout_is_a_stall_degraded_without_retry(self):
        client = make_client(api_max_retry=3)
        call = AsyncMock(side_effect=_timeout_error())
        assert run(client._request(call, label="test")) is None
        assert call.await_count == 1

    def test_consecutive_stalls_abort_the_run_and_a_success_resets_the_count(self):
        client = make_client()
        stalled = AsyncMock(side_effect=_timeout_error())
        for _ in range(7):
            assert run(client._request(stalled, label="test")) is None
        good = make_sdk_chat_response()
        assert run(client._request(AsyncMock(return_value=good), label="test")) is good  # resets
        for _ in range(7):
            assert run(client._request(stalled, label="test")) is None
        with pytest.raises(RuntimeError, match="stalled on 8 consecutive"):
            run(client._request(stalled, label="test"))


# ---------------------------------------------------------------------------
# 3b. Streamed generation
# ---------------------------------------------------------------------------


class TestStreamedGeneration:
    def test_chat_stream_folds_content_and_reasoning_per_sample(self):
        client = make_client()
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(
            return_value=make_sdk_chat_stream(
                ["Hel", "lo"], reasoning_pieces=["thi", "nk"], n=2, finish_reason="length"
            )
        )
        response = run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 2))
        assert [choice.message.content for choice in response.choices] == ["Hello", "Hello"]
        assert [choice.message.reasoning_content for choice in response.choices] == ["think", "think"]
        assert [choice.finish_reason for choice in response.choices] == ["length", "length"]

    def test_chat_stream_reads_vllms_reasoning_field(self):
        """vLLM streams the parsed channel as `reasoning`; `reasoning_content` is the older spelling."""

        async def chunks():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(index=0, finish_reason=None, delta=SimpleNamespace(content=None, reasoning="why "))
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(index=0, finish_reason=None, delta=SimpleNamespace(content=None, reasoning="not"))
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(index=0, finish_reason="stop", delta=SimpleNamespace(content="42", reasoning=None))
                ]
            )

        client = make_client()
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=chunks())
        response = run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 1))
        assert response.choices[0].message.reasoning_content == "why not"
        assert response.choices[0].message.content == "42"

    def test_chat_stream_without_reasoning_reports_none(self):
        client = make_client()
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_stream(["A"]))
        response = run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 1))
        assert response.choices[0].message.content == "A"
        assert response.choices[0].message.reasoning_content is None

    def test_text_stream_folds_text(self):
        client = make_client(use_chat_template=False)
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(return_value=make_sdk_completion_stream(["a", "b", "c"]))
        response = run(client._call_api_text_generative(sdk, "p", 8, 1, None))
        assert response.choices[0].text == "abc"

    def test_empty_stream_yields_empty_choices_not_none(self):
        """An empty generation must reach greedy_until as text [""], not as a degraded request."""

        async def nothing():
            return
            yield

        client = make_client()
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=nothing())
        response = run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 1))
        assert [choice.message.content for choice in response.choices] == [""]

    def test_greedy_until_reads_the_folded_stream(self):
        client = make_client()
        doc = Doc(query="Q?", choices=[], gold_index=0, task_name="t", generation_size=8)
        doc.id = "0"
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(
            return_value=make_sdk_chat_stream(["fin", "al"], reasoning_pieces=["r"])
        )
        sdk.__aenter__ = AsyncMock(return_value=sdk)
        sdk.__aexit__ = AsyncMock(return_value=False)
        with patch.object(VLLMOpenAIClient, "_make_client", lambda self: sdk):
            (result,) = client.greedy_until([doc])
        assert result.text == ["final"]
        assert result.reasonings == ["r"]

    def test_generation_prompts_are_held_to_the_context_budget_except_images(self):
        pil = pytest.importorskip("PIL.Image")

        def sent_extra_body(budget, images=None):
            client = make_client()
            client._context_budget = budget
            doc = Doc(query="Q?", choices=[], gold_index=0, task_name="t", generation_size=30, images=images)
            doc.id = "0"
            sdk = MagicMock()
            sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_stream(["ok"]))
            sdk.__aenter__ = AsyncMock(return_value=sdk)
            sdk.__aexit__ = AsyncMock(return_value=False)
            with patch.object(VLLMOpenAIClient, "_make_client", lambda self: sdk):
                client.greedy_until([doc])
            return sdk.chat.completions.create.call_args.kwargs["extra_body"] or {}

        assert sent_extra_body(100)["truncate_prompt_tokens"] == 70  # budget minus the answer's max_new_tokens
        assert "truncate_prompt_tokens" not in sent_extra_body(None)
        assert "truncate_prompt_tokens" not in sent_extra_body(20)  # no room left: send as is
        assert "truncate_prompt_tokens" not in sent_extra_body(100, images=[pil.new("RGB", (2, 2))])

    def test_text_generation_route_sends_the_bos_choice(self):
        def sent(add_special_tokens):
            client = make_client(use_chat_template=False)
            client._add_special_tokens = add_special_tokens
            sdk = MagicMock()
            sdk.completions.create = AsyncMock(return_value=make_sdk_completion_stream(["ok"]))
            run(client._call_api_text_generative(sdk, "<bos>rendered template", 8, 1, None))
            return sdk.completions.create.call_args.kwargs["extra_body"]["add_special_tokens"]

        assert sent(None) is True
        assert sent(True) is True
        assert sent(False) is False

    def test_text_generation_route_carries_the_prompt_room(self):
        client = make_client(use_chat_template=False)
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(return_value=make_sdk_completion_stream(["ok"]))
        run(client._call_api_text_generative(sdk, "p", 8, 1, None, prompt_room=5))
        assert sdk.completions.create.call_args.kwargs["extra_body"]["truncate_prompt_tokens"] == 5


# ---------------------------------------------------------------------------
# 3b'. Stall detection: queued behind others is not stuck
# ---------------------------------------------------------------------------


def _chat_chunk(piece):
    delta = SimpleNamespace(content=piece, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, finish_reason=None, delta=delta)])


# A generating stream: a chunk every `delay` seconds, then opens `gate` for the requests queued behind it.
async def _busy_stream(gate, chunks=12, delay=0.05):
    for _ in range(chunks):
        await asyncio.sleep(delay)
        yield _chat_chunk("x")
    gate.set()


# A request the server has accepted but not scheduled: silent until `gate` opens.
async def _queued_stream(gate, piece="late"):
    await gate.wait()
    yield _chat_chunk(piece)


class TestStallDetection:
    STALL_S = 0.3

    def collect(self, stream, progress):
        return _collect_chat_stream(stream, 1, progress=progress, stall_s=self.STALL_S)

    def test_a_queued_request_waits_while_other_streams_progress(self):
        async def scenario():
            progress, gate = _ProgressClock(), asyncio.Event()
            started = asyncio.get_running_loop().time()
            busy, queued = await asyncio.gather(
                self.collect(_busy_stream(gate), progress), self.collect(_queued_stream(gate), progress)
            )
            return queued, asyncio.get_running_loop().time() - started

        queued, waited = run(scenario())
        assert queued.choices[0].message.content == "late"
        assert waited > 2 * self.STALL_S  # it waited far longer than the stall bound and was not stalled

    def test_a_queued_request_stalls_when_no_stream_progresses(self):
        with pytest.raises(openai.APITimeoutError, match="any stream"):
            run(self.collect(_queued_stream(asyncio.Event()), _ProgressClock()))

    def test_a_started_stream_stalls_on_its_own_gap_while_others_progress(self):
        async def stuck():
            yield _chat_chunk("first")
            await asyncio.sleep(30)

        async def scenario():
            progress, gate = _ProgressClock(), asyncio.Event()
            return await asyncio.gather(
                self.collect(stuck(), progress),
                self.collect(_busy_stream(gate, chunks=20), progress),
                return_exceptions=True,
            )

        stalled, busy = run(scenario())
        assert isinstance(stalled, openai.APITimeoutError) and "this stream" in str(stalled)
        assert busy.choices[0].message.content == "x" * 20

    def test_a_stalled_stream_is_closed(self):
        class Hanging:
            closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(30)

            async def close(self):
                self.closed = True

        stream = Hanging()
        with pytest.raises(openai.APITimeoutError):
            run(self.collect(stream, _ProgressClock()))
        assert stream.closed

    def test_opening_a_stream_stalls_when_no_stream_progresses(self):
        client = make_client()
        client.timeout = 0.2
        sdk = MagicMock()

        async def never_opens(**kwargs):
            await asyncio.sleep(30)

        sdk.chat.completions.create = never_opens
        assert run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 1)) is None
        assert client._consecutive_stalls == 1

    def test_opening_a_stream_waits_while_other_streams_progress(self):
        client = make_client()
        client.timeout = 0.2
        sdk = MagicMock()

        async def opens_late(**kwargs):
            await asyncio.sleep(0.6)
            return make_sdk_chat_stream(["ok"])

        sdk.chat.completions.create = opens_late

        async def scenario():
            async def elsewhere():  # another stream receiving chunks meanwhile
                for _ in range(14):
                    await asyncio.sleep(0.05)
                    client._clock().tick()

            response, _ = await asyncio.gather(
                client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 1), elsewhere()
            )
            return response

        assert run(scenario()).choices[0].message.content == "ok"
        assert client._consecutive_stalls == 0

    def test_streamed_calls_have_no_sdk_read_timeout(self):
        client = make_client(use_chat_template=True)
        client.timeout = 42.0
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_stream(["ok"]))
        sdk.completions.create = AsyncMock(return_value=make_sdk_completion_stream(["ok"]))
        run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 1))
        run(client._call_api_text_generative(sdk, "Q", 8, 1, None))
        for create in (sdk.chat.completions.create, sdk.completions.create):
            timeout = create.call_args.kwargs["timeout"]
            assert isinstance(timeout, httpx.Timeout) and timeout.read is None and timeout.connect == 42.0

    def test_an_httpx_timeout_is_a_stall_not_a_crash(self):
        client = make_client()
        timed_out = AsyncMock(side_effect=httpx.ReadTimeout("no bytes"))
        for _ in range(7):
            assert run(client._request(timed_out, label="test")) is None
        with pytest.raises(RuntimeError, match="stalled on 8 consecutive"):
            run(client._request(timed_out, label="test"))


# ---------------------------------------------------------------------------
# 3b''. Adaptive concurrency
# ---------------------------------------------------------------------------

METRICS_SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="org/m"} 11.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="org/m"} 719.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="org/m",reason="capacity"} 719.0
vllm:kv_cache_usage_perc{engine="0",model_name="org/m"} 0.231717933836332
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="org/m"} 353.0
"""


def _load(running=100, waiting=0, kv=0.5, preemptions=0.0):
    return _ServerLoad(running=running, waiting=waiting, kv_usage=kv, preemptions=preemptions)


class TestServerLoad:
    def test_reads_the_series_the_limiter_steers_by(self):
        assert _parse_server_load(METRICS_SAMPLE) == _ServerLoad(
            running=11.0, waiting=719.0, kv_usage=0.231717933836332, preemptions=353.0
        )  # `waiting_by_reason` is a different series and is not added to `waiting`

    def test_engines_add_up_and_the_fullest_kv_cache_counts(self):
        two = METRICS_SAMPLE + (
            'vllm:num_requests_running{engine="1",model_name="org/m"} 4.0\n'
            'vllm:num_requests_waiting{engine="1",model_name="org/m"} 1.0\n'
            'vllm:kv_cache_usage_perc{engine="1",model_name="org/m"} 0.9\n'
            'vllm:num_preemptions_total{engine="1",model_name="org/m"} 2.0\n'
        )
        assert _parse_server_load(two) == _ServerLoad(running=15.0, waiting=720.0, kv_usage=0.9, preemptions=355.0)

    def test_a_server_missing_a_series_reports_no_load(self):
        assert _parse_server_load(METRICS_SAMPLE.replace("vllm:kv_cache_usage_perc", "vllm:other")) is None
        assert _parse_server_load("") is None

    def test_metrics_live_at_the_server_root(self):
        assert _metrics_url("http://localhost:8000/v1") == "http://localhost:8000/metrics"
        assert _metrics_url("http://localhost:8000/v1/") == "http://localhost:8000/metrics"


class TestNextLimit:
    @pytest.mark.parametrize(
        ("limit", "in_flight", "load", "preempted", "expected"),
        [
            pytest.param(64, 64, _load(kv=0.3), 0, 96, id="grows_while_the_server_takes_everything"),
            pytest.param(900, 900, _load(running=900, kv=0.3), 0, 1000, id="growth_stops_at_the_ceiling"),
            pytest.param(
                64, 64, _load(running=58, kv=0.3), 0, 96, id="grows_once_the_server_runs_nearly_all_in_flight"
            ),
            pytest.param(
                64, 64, _load(running=0, kv=0.0), 0, 64, id="holds_while_what_was_sent_has_not_been_scheduled"
            ),
            pytest.param(100, 100, _load(running=85, kv=0.3), 0, 100, id="holds_while_a_tenth_is_still_in_transit"),
            pytest.param(14, 14, _load(running=14, kv=0.5), 0, 21, id="regrows_after_a_shed_once_kv_frees_up"),
            pytest.param(64, 10, _load(kv=0.3), 0, 64, id="holds_when_the_client_is_not_saturating_its_limit"),
            pytest.param(64, 64, _load(waiting=5, kv=0.3), 0, 64, id="holds_while_requests_wait_at_the_server"),
            pytest.param(64, 64, _load(kv=0.9), 0, 64, id="holds_in_the_band_between_grow_and_shed"),
            pytest.param(100, 100, _load(running=80, kv=0.97), 0, 72, id="sheds_below_what_the_server_runs"),
            pytest.param(300, 300, _load(running=120, kv=0.97), 0, 150, id="sheds_by_at_most_half"),
            pytest.param(300, 300, _load(running=200, kv=0.5), 3, 180, id="sheds_on_preemption_even_with_kv_to_spare"),
            pytest.param(
                50, 120, _load(running=40, kv=0.99), 4, 50, id="does_not_shed_again_before_a_shed_takes_effect"
            ),
            pytest.param(14, 100, _load(running=100, kv=0.5), 0, 14, id="does_not_grow_before_a_shed_takes_effect"),
            pytest.param(50, 50, _load(running=200, kv=0.99), 0, 50, id="never_raises_the_limit_when_shedding"),
            pytest.param(1, 1, _load(running=0, kv=0.99), 0, 1, id="never_sheds_below_one"),
        ],
    )
    def test_control_step(self, limit, in_flight, load, preempted, expected):
        assert _next_limit(limit, ceiling=1000, in_flight=in_flight, load=load, preempted=preempted) == expected


class TestAdaptiveLimiter:
    def test_waiters_are_served_in_order_and_never_exceed_the_limit(self):
        async def scenario():
            limiter, order, peak = _AdaptiveLimiter(2), [], 0

            async def worker(name):
                nonlocal peak
                async with limiter:
                    peak = max(peak, limiter.in_flight)
                    order.append(name)
                    await asyncio.sleep(0.01)

            await asyncio.gather(*[worker(i) for i in range(6)])
            return order, peak, limiter.in_flight

        order, peak, left = run(scenario())
        assert order == [0, 1, 2, 3, 4, 5] and peak == 2 and left == 0

    def test_raising_the_limit_admits_waiters_at_once(self):
        async def scenario():
            limiter, gate, entered = _AdaptiveLimiter(1), asyncio.Event(), []

            async def worker(name):
                async with limiter:
                    entered.append(name)
                    await gate.wait()

            tasks = [asyncio.create_task(worker(i)) for i in range(3)]
            await asyncio.sleep(0.02)
            before = list(entered)
            limiter.set_limit(3)
            await asyncio.sleep(0.02)
            after = list(entered)
            gate.set()
            await asyncio.gather(*tasks)
            return before, after

        assert run(scenario()) == ([0], [0, 1, 2])

    def test_lowering_the_limit_evicts_nothing_and_holds_back_new_requests(self):
        async def scenario():
            limiter, gate, entered = _AdaptiveLimiter(3), asyncio.Event(), []

            async def worker(name):
                async with limiter:
                    entered.append(name)
                    await gate.wait()

            first = [asyncio.create_task(worker(i)) for i in range(3)]
            await asyncio.sleep(0.02)
            limiter.set_limit(1)
            late = asyncio.create_task(worker("late"))
            await asyncio.sleep(0.02)
            held = (limiter.in_flight, list(entered))
            gate.set()
            await asyncio.gather(*first, late)
            return held, entered

        (in_flight, entered_while_held), entered = run(scenario())
        assert in_flight == 3 and entered_while_held == [0, 1, 2]  # nothing evicted, "late" waits for room
        assert entered[-1] == "late"

    def test_a_cancelled_waiter_leaves_no_slot_behind(self):
        async def scenario():
            limiter = _AdaptiveLimiter(1)
            await limiter.__aenter__()
            waiter = asyncio.create_task(limiter.__aenter__())
            await asyncio.sleep(0.01)
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            await limiter.__aexit__(None, None, None)
            await asyncio.wait_for(limiter.__aenter__(), 1)  # the slot is free again
            return limiter.in_flight, len(limiter._waiters)

        assert run(scenario()) == (1, 0)

    def test_a_waiter_cancelled_after_being_granted_gives_the_slot_back(self):
        async def scenario():
            limiter = _AdaptiveLimiter(1)
            await limiter.__aenter__()
            waiter = asyncio.create_task(limiter.__aenter__())
            await asyncio.sleep(0.01)
            await limiter.__aexit__(None, None, None)  # grants the slot to the waiter...
            waiter.cancel()  # ...which is cancelled before it resumes
            await asyncio.gather(waiter, return_exceptions=True)
            return limiter.in_flight

        assert run(scenario()) == 0


def _fake_fetch(*texts):
    """A metrics fetcher returning ``texts`` in turn, then hanging like an idle poll."""
    calls = []

    async def fetch():
        calls.append(1)
        if len(calls) > len(texts):
            await asyncio.sleep(30)
        item = texts[len(calls) - 1]
        return item() if callable(item) else item  # a callable reads the state at fetch time

    return fetch, calls


def _series(running=100, waiting=0, kv=0.5, preemptions=0):
    return (
        f"vllm:num_requests_running {running}\nvllm:num_requests_waiting {waiting}\n"
        f"vllm:kv_cache_usage_perc {kv}\nvllm:num_preemptions_total {preemptions}\n"
    )


async def _steer_through(limiter, ceiling, *texts):
    fetch, calls = _fake_fetch(*texts)
    limits = []

    async def recording_fetch():
        limits.append(limiter.limit)
        limiter.in_flight = limiter.limit  # the client keeps its limit full
        return await fetch()

    task = asyncio.create_task(_steer(limiter, ceiling=ceiling, fetch=recording_fetch, interval=0.005))
    while len(calls) < len(texts):
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return limits


class TestSteering:
    def test_ramps_up_while_the_server_runs_what_is_sent_and_stops_at_the_ceiling(self):
        limiter = _AdaptiveLimiter(64)
        server_runs_all = lambda: _series(running=limiter.limit, kv=0.1)  # noqa: E731
        limits = run(_steer_through(limiter, 200, *[server_runs_all] * 5))
        assert limits[:4] == [64, 96, 144, 200] and limiter.limit == 200

    def test_does_not_ramp_while_the_server_has_not_scheduled_what_was_sent(self):
        # Requests still on their way to the scheduler are invisible to /metrics: no running, no waiting, no KV.
        limiter = _AdaptiveLimiter(64)
        limits = run(_steer_through(limiter, 1024, *[_series(running=0, kv=0.0)] * 6))
        assert limiter.limit == 64 and set(limits) == {64}

    def test_the_startup_of_the_eurollm_ifstruct_run_no_longer_overshoots(self):
        # The readings the first live run steered by (2026-09-29, EuroLLM-22B, ifstruct, 2 s apart): the engine was
        # still warming up, so /metrics showed nothing of the 64 requests already sent. That run went 64 -> 1024.
        startup = [_series(running=0, waiting=0, kv=0.0)] * 7
        # ...then the server caught up with what had been sent.
        caught_up = [_series(running=64, waiting=0, kv=0.4)]
        limiter = _AdaptiveLimiter(64)
        limits = run(_steer_through(limiter, 1024, *startup, *caught_up))
        assert set(limits[:8]) == {64} and limiter.limit == 96

    def test_sheds_when_the_kv_cache_is_exhausted(self):
        limiter = _AdaptiveLimiter(300)
        run(_steer_through(limiter, 1000, _series(running=120, kv=0.97)))
        assert limiter.limit == 150  # 90 % of the 120 running would be 108, but a shed at most halves

    def test_preemptions_before_the_first_reading_are_not_new_ones(self):
        limiter = _AdaptiveLimiter(300)
        limits = run(_steer_through(limiter, 1000, _series(running=200, preemptions=5)))
        assert limits[0] == 300 and limiter.limit == 300  # held (200 of 300 seen), not shed for old preemptions

    def test_sheds_on_a_new_preemption(self):
        limiter = _AdaptiveLimiter(300)
        run(_steer_through(limiter, 1000, _series(running=200, preemptions=5), _series(running=200, preemptions=7)))
        assert limiter.limit == 180  # 90 % of the 200 running

    def test_a_server_without_metrics_is_driven_at_the_ceiling(self, caplog):
        limiter = _AdaptiveLimiter(64)
        with caplog.at_level("WARNING"):
            run(_steer_through(limiter, 500, None, None))
        assert limiter.limit == 500
        assert sum("reports no load" in record.message for record in caplog.records) == 1

    def test_a_missed_reading_keeps_the_limit_once_the_server_has_reported(self, caplog):
        # 2026-09-29: a /metrics fetch timed out under load and the limit jumped from 1639 to the 8192 ceiling.
        limiter = _AdaptiveLimiter(300)
        with caplog.at_level("WARNING"):
            run(_steer_through(limiter, 8192, _series(running=100, waiting=5), None, None))
        assert limiter.limit == 300
        assert not any("reports no load" in record.message for record in caplog.records)

    def test_gate_is_a_plain_semaphore_unless_adaptive(self):
        async def scenario(adaptive):
            client = make_client()
            client.adaptive_concurrency = adaptive
            client._fetch_load = AsyncMock(return_value=None)
            async with client._gate() as gate:
                return type(gate)

        assert run(scenario(False)) is asyncio.Semaphore
        assert run(scenario(True)) is _AdaptiveLimiter

    def test_the_adaptive_gate_starts_low_under_the_ceiling_and_stops_steering_on_exit(self):
        async def scenario():
            client = make_client()
            client.adaptive_concurrency = True
            client.concurrent_requests = 1024
            client._fetch_load = AsyncMock(return_value=None)
            async with client._gate() as gate:
                start = gate.limit
            leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            return start, leftover

        assert run(scenario()) == (64, [])

    def test_a_small_ceiling_is_never_exceeded_at_the_start(self):
        async def scenario():
            client = make_client()
            client.adaptive_concurrency = True
            client.concurrent_requests = 11
            client._fetch_load = AsyncMock(return_value=None)
            async with client._gate() as gate:
                return gate.limit

        assert run(scenario()) == 11

    def test_greedy_until_runs_through_the_adaptive_gate(self):
        client = make_client()
        client.adaptive_concurrency = True
        client._fetch_load = AsyncMock(return_value=None)
        doc = Doc(query="Q?", choices=[], gold_index=0, task_name="t", generation_size=8)
        doc.id = "0"
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_stream(["fin", "al"]))
        sdk.__aenter__ = AsyncMock(return_value=sdk)
        sdk.__aexit__ = AsyncMock(return_value=False)
        with patch.object(VLLMOpenAIClient, "_make_client", lambda self: sdk):
            (result,) = client.greedy_until([doc])
        assert result.text == ["final"]

    def test_the_option_is_off_by_default_and_not_part_of_the_cache_key(self):
        assert VLLMOpenAIModelConfig(model_name="org/m", base_url="http://x/v1").adaptive_concurrency is False
        with tempfile.TemporaryDirectory() as temp_dir:
            a = VLLMOpenAIModelConfig(model_name="org/m", base_url="http://x/v1", cache_dir=temp_dir)
            b = VLLMOpenAIModelConfig(
                model_name="org/m", base_url="http://x/v1", adaptive_concurrency=True, cache_dir=temp_dir
            )
            assert SampleCache(a).get_model_hash(a) == SampleCache(b).get_model_hash(b)


# ---------------------------------------------------------------------------
# 3c. Loglikelihood from client-side token ids
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """One id per character; BOS = 1 when special tokens are requested; a visible chat template."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "".join(m["content"] for m in messages) + ("<gen>" if add_generation_prompt else "")


def _echo_response_for(prompt_ids):
    """Echo logprobs shaped like vLLM's: None for the first prompt token, one generated token at the end."""
    tokens = [chr(i) for i in prompt_ids] + ["g"]
    token_logprobs = [None] + [-0.5] * (len(prompt_ids) - 1) + [-9.0]
    top_logprobs = [{tok: lp if lp is not None else 0.0} for tok, lp in zip(tokens, token_logprobs)]
    return make_sdk_completion_response(tokens, token_logprobs, top_logprobs=top_logprobs)


def _token_client(**kwargs):
    client = make_client(client_tokenization=True, **kwargs)
    client._tokenizer = FakeTokenizer()
    return client


class TestTokenIdLoglikelihood:
    def test_chat_context_is_rendered_locally_and_sent_as_ids(self):
        client = _token_client()
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(side_effect=lambda **kw: _echo_response_for(kw["prompt"]))
        doc = make_doc("ab", [" c", " dd"])

        result = run(client._process_doc_token_loglikelihood_async(doc, sdk, asyncio.Semaphore(4)))

        context = "ab<gen>"  # the local chat template, with the generation prompt
        assert result.input == context
        sent = [call.kwargs["prompt"] for call in sdk.completions.create.call_args_list]
        assert sent == [[ord(c) for c in context + " c"], [ord(c) for c in context + " dd"]]  # no BOS under a template
        assert result.output_tokens == [[ord(c) for c in " c"], [ord(c) for c in " dd"]]
        assert result.logprobs == pytest.approx(
            [-1.0, -1.5]
        )  # last len(continuation) positions, generated token excluded
        assert result.argmax_logits_eq_gold == [True, True]
        payload = sdk.completions.create.call_args.kwargs
        assert payload["echo"] is True and payload["logprobs"] == 1 and payload["max_tokens"] == 1

    def test_plain_text_context_gets_bos_and_the_continuation_does_not(self):
        client = _token_client(use_chat_template=False)
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(side_effect=lambda **kw: _echo_response_for(kw["prompt"]))
        doc = make_doc("ab", [" c"])

        result = run(client._process_doc_token_loglikelihood_async(doc, sdk, asyncio.Semaphore(4)))

        (sent,) = [call.kwargs["prompt"] for call in sdk.completions.create.call_args_list]
        assert sent[0] == 1 and sent[1:] == [ord(c) for c in "ab c"]
        assert result.input_tokens == [[1, ord("a"), ord("b")]]
        assert result.logprobs == pytest.approx([-1.0])

    def test_boundary_whitespace_belongs_to_the_continuation(self):
        """tok_encode_pair moves trailing context spaces into the continuation, as in-process."""
        client = _token_client(use_chat_template=False)
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(side_effect=lambda **kw: _echo_response_for(kw["prompt"]))
        doc = make_doc("ab ", ["c"])
        result = run(client._process_doc_token_loglikelihood_async(doc, sdk, asyncio.Semaphore(4)))
        assert result.output_tokens == [[ord(" "), ord("c")]]
        assert result.logprobs == pytest.approx([-1.0])

    def test_degraded_request_scores_minus_inf(self):
        client = _token_client()
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(return_value=None)
        with patch.object(client, "_request", AsyncMock(return_value=None)):
            result = run(
                client._process_doc_token_loglikelihood_async(make_doc("q", [" a"]), sdk, asyncio.Semaphore(1))
            )
        assert result.logprobs == [float("-inf")] and result.argmax_logits_eq_gold == [False]

    def test_images_keep_the_server_templated_chat_route(self):
        client = _token_client()
        pil = pytest.importorskip("PIL.Image")
        docs = [make_doc("Q?", [" A"], images=[pil.new("RGB", (2, 2))]), make_doc("plain", [" B"], doc_id="1")]
        calls = []

        async def fake_chat(doc, client, semaphore):
            calls.append(("chat", doc.query))
            return ModelResponse(input="x", logprobs=[-0.1], argmax_logits_eq_gold=[True])

        async def fake_tokens(doc, client, semaphore):
            calls.append(("ids", doc.query))
            return ModelResponse(input="y", logprobs=[-0.2], argmax_logits_eq_gold=[True])

        with (
            patch.object(client, "_process_doc_chat_loglikelihood_async", side_effect=fake_chat),
            patch.object(client, "_process_doc_token_loglikelihood_async", side_effect=fake_tokens),
        ):
            results = run(client._loglikelihood_async(docs, AsyncMock()))
        assert calls == [("chat", "Q?"), ("ids", "plain")]
        assert [r.logprobs for r in results] == [[-0.1], [-0.2]]

    def test_add_special_tokens_default_follows_the_template(self):
        assert _token_client().add_special_tokens is False
        assert _token_client(use_chat_template=False).add_special_tokens is True
        explicit = _token_client()
        explicit._add_special_tokens = True
        assert explicit.add_special_tokens is True

    def test_tokenizer_runs_repo_code_only_when_configured(self):
        loads = []

        def fake_from_pretrained(name, **kwargs):
            loads.append((name, kwargs))
            return MagicMock()

        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
            for trust in (None, True):
                extra = {} if trust is None else {"trust_remote_code": trust}
                config = VLLMOpenAIModelConfig(model_name="org/model-it", base_url="http://x/v1", **extra)
                VLLMOpenAIClient(config).tokenizer
        assert loads == [
            ("org/model-it", {"revision": None, "trust_remote_code": False}),
            ("org/model-it", {"revision": None, "trust_remote_code": True}),
        ]
        assert "trust_remote_code" in VLLMOpenAIModelConfig.CACHE_KEY_EXCLUDE

    def test_tokenizer_loads_the_pinned_revision_of_the_model_repo(self):
        loads = []

        def fake_from_pretrained(name, **kwargs):
            loads.append((name, kwargs["revision"]))
            return MagicMock()

        sha = "0123456789abcdef0123456789abcdef01234567"
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained):
            for extra in ({}, {"tokenizer": "org/other-tokenizer"}):
                config = VLLMOpenAIModelConfig(
                    model_name="org/model-it", base_url="http://x/v1", revision=sha, **extra
                )
                VLLMOpenAIClient(config).tokenizer
        assert loads == [("org/model-it", sha), ("org/other-tokenizer", None)]
        assert "revision" not in VLLMOpenAIModelConfig.CACHE_KEY_EXCLUDE

    def test_ids_over_the_context_budget_keep_their_tail(self):
        client = _token_client()
        client._context_budget = 6  # room for 5 ids beside the one generated token
        sdk = MagicMock()
        sdk.completions.create = AsyncMock(side_effect=lambda **kw: _echo_response_for(kw["prompt"]))

        result = run(
            client._process_doc_token_loglikelihood_async(make_doc("abcdef", [" c"]), sdk, asyncio.Semaphore(4))
        )

        assert sdk.completions.create.call_args.kwargs["prompt"] == [ord(c) for c in ("abcdef<gen>" + " c")[-5:]]
        assert result.logprobs == pytest.approx([-1.0])  # the continuation is still the last 2 positions
        assert result.input_tokens == [[ord(c) for c in "abcdef<gen>"]]  # reported untruncated

        client._context_budget = 3  # a continuation longer than the room is never cut into
        run(client._process_doc_token_loglikelihood_async(make_doc("ab", [" dd"]), sdk, asyncio.Semaphore(4)))
        assert sdk.completions.create.call_args.kwargs["prompt"] == [ord(c) for c in "ab<gen> dd"]

    def test_client_tokenization_off_keeps_legacy_routes(self):
        client = make_client()
        docs = [make_doc("Q?", [" A"])]

        async def fake_chat(doc, client, semaphore):
            return ModelResponse(input="x", logprobs=[-0.3], argmax_logits_eq_gold=[True])

        with patch.object(client, "_process_doc_chat_loglikelihood_async", side_effect=fake_chat):
            (result,) = run(client._loglikelihood_async(docs, AsyncMock()))
        assert result.logprobs == [-0.3]


# ---------------------------------------------------------------------------
# 4. Message construction (incl. multimodal)
# ---------------------------------------------------------------------------


class TestBuildLlContextMessages:
    def test_text_doc_uses_api_prompt(self):
        client = make_client()
        doc = make_doc("Question?", [" A", " B"])
        messages = client._build_ll_context_messages(doc)
        assert messages == [{"role": "user", "content": "Question?"}]

    def _make_image(self):
        pil = pytest.importorskip("PIL.Image")
        return pil.new("RGB", (2, 2), color=(255, 0, 0))

    def test_multi_image_doc_inline_markers(self):
        client = make_client()
        images = [self._make_image(), self._make_image()]
        doc = make_doc("Look at <image 1> and <image 2> now", [" A"], images=images)
        messages = client._build_ll_context_messages(doc)

        user_message = messages[-1]
        assert user_message["role"] == "user"
        types = [part["type"] for part in user_message["content"]]
        assert types == ["text", "image_url", "text", "image_url", "text"]
        for part in user_message["content"]:
            if part["type"] == "image_url":
                assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_doc_instruction_becomes_system_turn(self):
        client = make_client()
        doc = make_doc("<image 1> Solve.", [" A"], images=[self._make_image()], instruction="Be terse.")
        messages = client._build_ll_context_messages(doc)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == [{"type": "text", "text": "Be terse."}]
        assert messages[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# 5. Chat-route LL scoring
# ---------------------------------------------------------------------------


class TestProcessDocChatLoglikelihood:
    def test_two_choices_scored_with_boundary_whitespace(self):
        """The newline the template emits after the assistant turn opener is
        scored with the continuation, mirroring in-process tok_encode_pair."""
        client = make_client()
        doc = make_doc("Q?", [" A", " B"])

        responses = {
            " A": make_sdk_chat_response(
                prompt_logprobs=make_prompt_logprobs_payload(
                    [("<start_of_turn>", -0.1, 1), ("model", -0.2, 1), ("\n", -0.25, 1), ("▁A", -0.5, 1)]
                )
            ),
            " B": make_sdk_chat_response(
                prompt_logprobs=make_prompt_logprobs_payload(
                    [("<start_of_turn>", -0.1, 1), ("model", -0.2, 1), ("\n", -0.25, 1), ("▁B", -1.5, 2)]
                )
            ),
        }

        async def fake_call(sdk, messages):
            return responses[messages[-1]["content"]]

        sem = asyncio.Semaphore(4)
        with patch.object(client, "_call_api_chat_prompt_logprobs_async", side_effect=fake_call):
            result = run(client._process_doc_chat_loglikelihood_async(doc, MagicMock(), sem))

        assert result.logprobs == pytest.approx([-0.75, -1.75])  # "\n" + choice token
        assert result.argmax_logits_eq_gold == [True, False]

    def test_failed_choice_gets_neg_inf(self):
        client = make_client()
        doc = make_doc("Q?", [" A", " B"])

        async def fake_call(sdk, messages):
            if messages[-1]["content"] == " A":
                return make_sdk_chat_response(prompt_logprobs=make_prompt_logprobs_payload([("▁A", -0.5, 1)]))
            return None

        sem = asyncio.Semaphore(4)
        with patch.object(client, "_call_api_chat_prompt_logprobs_async", side_effect=fake_call):
            result = run(client._process_doc_chat_loglikelihood_async(doc, MagicMock(), sem))

        assert result.logprobs[0] == pytest.approx(-0.5)
        assert result.logprobs[1] == float("-inf")
        assert result.argmax_logits_eq_gold == [True, False]

    def test_alignment_failure_warns_and_falls_back(self, caplog):
        client = make_client()
        doc = make_doc("Q?", ["АБ"])  # decoded pieces below cannot rebuild this

        response = make_sdk_chat_response(
            prompt_logprobs=make_prompt_logprobs_payload([("xx", -0.3, 1), ("yy", -0.7, 1)])
        )

        async def fake_call(sdk, messages):
            return response

        import logging

        sem = asyncio.Semaphore(4)
        with patch.object(client, "_call_api_chat_prompt_logprobs_async", side_effect=fake_call):
            with caplog.at_level(logging.WARNING, logger="lighteval.models.endpoints.vllm_openai_model"):
                result = run(client._process_doc_chat_loglikelihood_async(doc, MagicMock(), sem))

        assert "Could not align continuation" in caplog.text
        # Fallback scores the trailing tokens covering the continuation's length:
        # "yy" (2 chars) covers the 2-char continuation, so only it is scored.
        assert result.logprobs == pytest.approx([-0.7])

    def test_prompt_logprobs_read_from_choice_level(self):
        # Some server versions attach prompt_logprobs to the choice, not the response.
        client = make_client()
        doc = make_doc("Q?", [" A"])
        response = make_sdk_chat_response()
        response.choices[0].prompt_logprobs = make_prompt_logprobs_payload([("▁A", -0.4, 1)])

        async def fake_call(sdk, messages):
            return response

        sem = asyncio.Semaphore(4)
        with patch.object(client, "_call_api_chat_prompt_logprobs_async", side_effect=fake_call):
            result = run(client._process_doc_chat_loglikelihood_async(doc, MagicMock(), sem))

        assert result.logprobs == pytest.approx([-0.4])


# ---------------------------------------------------------------------------
# 6. Echo-route LL scoring
# ---------------------------------------------------------------------------


class TestProcessDocEchoLoglikelihood:
    def test_two_choices_correct_logprobs(self):
        client = make_client(use_chat_template=False)
        doc = make_doc("Q:", [" A", " B"])

        responses = {
            "Q: A": make_sdk_completion_response(
                ["Q", ":", " A", "_gen"], [None, -0.1, -0.5, -0.9], text_offset=[0, 1, 2, 4]
            ),
            "Q: B": make_sdk_completion_response(
                ["Q", ":", " B", "_gen"], [None, -0.1, -1.5, -0.9], text_offset=[0, 1, 2, 4]
            ),
        }

        async def fake_call(sdk, full_text):
            return responses[full_text]

        sem = asyncio.Semaphore(4)
        with patch.object(client, "_call_api_text_completion_async", side_effect=fake_call):
            result = run(client._process_doc_loglikelihood_async(doc, "Q:", MagicMock(), sem))

        assert result.logprobs == pytest.approx([-0.5, -1.5])

    def test_failed_call_returns_neg_inf(self):
        client = make_client(use_chat_template=False)
        doc = make_doc("Q:", [" A"])

        async def fake_call(sdk, full_text):
            return None

        sem = asyncio.Semaphore(4)
        with patch.object(client, "_call_api_text_completion_async", side_effect=fake_call):
            result = run(client._process_doc_loglikelihood_async(doc, "Q:", MagicMock(), sem))

        assert result.logprobs == [float("-inf")]
        assert result.argmax_logits_eq_gold == [False]

    def test_rolling_sums_all_token_logprobs(self):
        client = make_client(use_chat_template=False)
        doc = make_doc("Hello world", choices=[])
        response = make_sdk_completion_response(["Hello", " world", "_gen"], [None, -0.3, -0.9])

        async def fake_call(sdk, full_text):
            assert full_text == "Hello world"
            return response

        sem = asyncio.Semaphore(4)
        with patch.object(client, "_call_api_text_completion_async", side_effect=fake_call):
            result = run(client._process_doc_rolling_async(doc, MagicMock(), sem))

        # token_logprobs[1:-1]: leading None dropped, trailing generated token dropped
        assert result.logprobs == pytest.approx([-0.3])


# ---------------------------------------------------------------------------
# 7. Dispatch and guards
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_chat_route_used_for_chat_template_client(self):
        client = make_client()
        docs = [make_doc("Q?", [" A", " B"], doc_id="0")]

        async def fake_process(doc, client, semaphore):
            return ModelResponse(input="ctx", logprobs=[-0.1, -0.9], argmax_logits_eq_gold=[True, False])

        with (
            patch.object(client, "_process_doc_chat_loglikelihood_async", side_effect=fake_process),
            patch.object(VLLMOpenAIClient, "_make_client", lambda self: AsyncMock()),
        ):
            results = client.loglikelihood(docs)

        assert len(results) == 1
        assert results[0].logprobs == pytest.approx([-0.1, -0.9])

    def test_plain_route_used_without_chat_template(self):
        client = make_client(use_chat_template=False)
        docs = [make_doc("Q?", [" A"], doc_id="0")]

        async def fake_process(doc, context_str, client, semaphore):
            assert context_str == "Q?"
            return ModelResponse(input=context_str, logprobs=[-0.2], argmax_logits_eq_gold=[True])

        with (
            patch.object(client, "_process_doc_loglikelihood_async", side_effect=fake_process),
            patch.object(VLLMOpenAIClient, "_make_client", lambda self: AsyncMock()),
        ):
            results = client.loglikelihood(docs)

        assert results[0].logprobs == pytest.approx([-0.2])

    def test_plain_route_rejects_images(self):
        client = make_client(use_chat_template=False)
        pil = pytest.importorskip("PIL.Image")
        doc = make_doc("Q?", [" A"], images=[pil.new("RGB", (2, 2))])
        with pytest.raises(ValueError, match="use_chat_template=True"):
            client.loglikelihood([doc])

    def test_greedy_plain_route_rejects_images(self):
        client = make_client(use_chat_template=False)
        pil = pytest.importorskip("PIL.Image")
        doc = Doc(query="Q?", choices=[], gold_index=0, task_name="t", generation_size=8)
        doc.images = [pil.new("RGB", (2, 2))]
        with (
            patch.object(VLLMOpenAIClient, "_make_client", lambda self: AsyncMock()),
            pytest.raises(ValueError, match="chat template"),
        ):
            client.greedy_until([doc])


# ---------------------------------------------------------------------------
# 8. max_length probe
# ---------------------------------------------------------------------------


class TestMaxLength:
    def test_configured_value_wins(self):
        client = make_client()
        client._max_length = 5632
        assert client.max_length == 5632

    def test_probe_reads_max_model_len(self, monkeypatch):
        client = make_client()
        client._max_length = None

        def fake_get(url, headers=None, timeout=None):
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request, json={"data": [{"id": "org/model-it", "max_model_len": 8192}]})

        monkeypatch.setattr(httpx, "get", fake_get)
        assert client.max_length == 8192

    def test_probe_failure_falls_back(self, monkeypatch):
        client = make_client()
        client._max_length = None

        def fake_get(url, headers=None, timeout=None):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", fake_get)
        assert client.max_length == 4096


def _stream_error(message="Unexpected token 200002 while expecting start token 200006") -> openai.APIError:
    return openai.APIError(
        message, request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"), body=None
    )


class TestErrorsInsideAStream:
    def test_an_error_inside_a_stream_is_retried_then_degraded(self):
        client = make_client(api_max_retry=3)
        call = AsyncMock(side_effect=_stream_error())
        assert run(client._request(call, label="test")) is None
        assert call.await_count >= 3  # the same bound as a 5xx

    def test_an_error_inside_a_stream_is_retried_until_the_sample_succeeds(self):
        client = make_client(api_max_retry=3)
        good = make_sdk_chat_response()
        call = AsyncMock(side_effect=[_stream_error(), good])
        assert run(client._request(call, label="test")) is good

    def test_a_generation_the_server_cannot_parse_degrades_only_that_request(self):
        async def breaks_mid_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0, finish_reason=None, delta=SimpleNamespace(content="par", reasoning_content=None)
                    )
                ]
            )
            raise _stream_error()

        client = make_client(api_max_retry=2)
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(side_effect=lambda **kwargs: breaks_mid_stream())
        assert run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 8, 1)) is None
        assert sdk.chat.completions.create.await_count >= 2


def test_the_connection_pool_holds_every_request_in_flight():
    # 2026-09-29: 2000 in flight through the SDK's default 1000-connection pool ran at 9k tok/s; a pool of 2000, 119k.
    client = make_client()
    client.concurrent_requests = 2048
    with patch.object(openai, "DefaultAsyncHttpxClient", wraps=openai.DefaultAsyncHttpxClient) as http_client:
        client._make_client()
    limits = http_client.call_args.kwargs["limits"]
    assert (limits.max_connections, limits.max_keepalive_connections) == (2048, 2048)


def test_an_idle_connection_expires_before_the_server_closes_it():
    # 2026-09-30: a pool of 1024 idle connections expiring at httpx's 5 s, the same as vLLM's keep-alive, reused
    # connections the server had just closed; every loglikelihood run logged ReadErrors and one group aborted.
    with patch.object(openai, "DefaultAsyncHttpxClient", wraps=openai.DefaultAsyncHttpxClient) as http_client:
        make_client()._make_client()
    assert http_client.call_args.kwargs["limits"].keepalive_expiry < 5.0


def test_the_open_files_limit_fits_every_connection(monkeypatch):
    # 2026-09-29: a worker container's soft limit of 1024 failed every connect past ~1000 in flight (EMFILE).
    limit = {"nofile": (1024, 524288)}
    monkeypatch.setattr(resource, "getrlimit", lambda _: limit["nofile"])
    monkeypatch.setattr(resource, "setrlimit", lambda _, value: limit.update(nofile=value))
    client = make_client()
    client.concurrent_requests = 8192
    client._make_client()
    assert limit["nofile"] == (8192 + 1024, 524288)


@pytest.mark.parametrize(
    "limit, expected, warns",
    [((65536, 524288), (65536, 524288), False), ((1024, 2048), (2048, 2048), True)],
    ids=["already-fits", "hard-limit-too-low"],
)
def test_the_open_files_limit_is_only_raised_up_to_the_hard_limit(monkeypatch, caplog, limit, expected, warns):
    current = {"nofile": limit}
    monkeypatch.setattr(resource, "getrlimit", lambda _: current["nofile"])
    monkeypatch.setattr(resource, "setrlimit", lambda _, value: current.update(nofile=value))
    with caplog.at_level("WARNING"):
        _fit_open_files(4096)
    assert current["nofile"] == expected
    assert any("cannot fit 4096 connections" in record.message for record in caplog.records) is warns
