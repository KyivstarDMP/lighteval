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
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from test_openai_scoring import make_prompt_logprobs_payload

from lighteval.models.endpoints.vllm_openai_model import VLLMOpenAIClient, VLLMOpenAIModelConfig
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
):
    """Construct a VLLMOpenAIClient bypassing __init__ (no cache, no SDK client)."""
    client = object.__new__(VLLMOpenAIClient)
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

    def test_default_temperature_zero_is_sent(self):
        standard, extra = make_client()._sampling_params()
        assert standard == {"temperature": 0}
        assert extra == {}

    def test_chat_generative_payload(self):
        client = make_client(generation_parameters=GenerationParameters(temperature=1.0, top_k=64))
        client.prompt_manager.chat_template_kwargs = {"enable_thinking": False}
        sdk = MagicMock()
        sdk.chat.completions.create = AsyncMock(return_value=make_sdk_chat_response())

        run(client._call_api_chat_generative(sdk, [{"role": "user", "content": "Q"}], 128, 1))

        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "org/model-it"
        assert kwargs["max_tokens"] == 128
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
        sdk.completions.create = AsyncMock(return_value=make_sdk_completion_response(["t"], [None]))

        run(client._call_api_text_generative(sdk, "plain prompt", 64, 1, None))

        kwargs = sdk.completions.create.call_args.kwargs
        assert kwargs["prompt"] == "plain prompt"
        assert kwargs["max_tokens"] == 64
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

    def test_timeout_backs_off_then_succeeds(self):
        client = make_client(api_max_retry=3)
        good = make_sdk_chat_response()
        call = AsyncMock(side_effect=[_timeout_error(), good])
        assert run(client._request(call, label="test")) is good


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
