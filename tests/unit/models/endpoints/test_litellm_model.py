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

import asyncio
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from lighteval.models.endpoints.litellm_model import LiteLLMClient
from lighteval.models.model_input import GenerationParameters
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.utils.imports import is_package_available


pytestmark = pytest.mark.skipif(not is_package_available("litellm"), reason="litellm extra is not installed")


def _build_client(model_name: str, generation_parameters: GenerationParameters) -> LiteLLMClient:
    client = LiteLLMClient.__new__(LiteLLMClient)
    client.model = model_name
    client.provider = "openai"
    client.base_url = None
    client.api_key = None
    client.generation_parameters = generation_parameters
    client._max_length = 10_000
    client.API_MAX_RETRY = 1
    client.API_RETRY_SLEEP = 0
    client.API_RETRY_MULTIPLIER = 1
    client.timeout = None
    return client


def _run_call_api(
    client: LiteLLMClient, *, supports_reasoning: bool, stop_sequence: list[str] | tuple | None = None
) -> Mock:
    """Drive ``__call_api`` once against a mocked ``litellm.acompletion``, returning that mock.

    Callers assert on ``.call_args.kwargs`` to check what was sent to the provider.
    """
    response = Mock()
    response.choices = [Mock(message=Mock(content="ok"))]

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=supports_reasoning):
        with patch(
            "lighteval.models.endpoints.litellm_model.litellm.acompletion", return_value=response
        ) as completion:
            asyncio.run(
                client._LiteLLMClient__call_api(
                    prompt=[{"role": "user", "content": "hello"}],
                    return_logits=False,
                    max_new_tokens=64,
                    num_samples=1,
                    stop_sequence=stop_sequence,
                )
            )

    return completion


@pytest.mark.parametrize(
    "reasoning_effort, supports_reasoning_value, expected_prepared_max_new_tokens",
    [
        (None, True, 100),
        ("none", True, 100),
        ("low", False, 100),
        ("low", True, 1000),
    ],
)
def test_prepare_max_new_tokens_boosts_only_with_reasoning_effort(
    reasoning_effort: str | None, supports_reasoning_value: bool, expected_prepared_max_new_tokens: int
):
    client = _build_client("openai/o3-mini", GenerationParameters(reasoning_effort=reasoning_effort))

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=supports_reasoning_value):
        assert client._prepare_max_new_tokens(100) == expected_prepared_max_new_tokens


def test_call_api_o_series_keeps_reasoning_effort_but_drops_sampling_params():
    client = _build_client("openai/o3-mini", GenerationParameters(temperature=0.2, top_p=0.9, reasoning_effort="low"))

    completion_kwargs = _run_call_api(client, supports_reasoning=False).call_args.kwargs
    assert completion_kwargs["reasoning_effort"] == "low"
    assert "temperature" not in completion_kwargs
    assert "top_p" not in completion_kwargs


def test_call_api_non_o_series_passes_full_litellm_generation_kwargs():
    client = _build_client(
        "google/gemini-2.5-flash", GenerationParameters(temperature=0.2, top_p=0.9, reasoning_effort="low")
    )

    completion_kwargs = _run_call_api(client, supports_reasoning=False).call_args.kwargs
    assert completion_kwargs["temperature"] == 0.2
    assert completion_kwargs["top_p"] == 0.9
    assert completion_kwargs["reasoning_effort"] == "low"


def test_call_api_openai_non_reasoning_uses_only_max_tokens():
    client = _build_client("openai/gpt-4.1-nano", GenerationParameters(max_new_tokens=96))

    completion_kwargs = _run_call_api(client, supports_reasoning=False).call_args.kwargs
    assert completion_kwargs["max_tokens"] == 64
    assert "max_completion_tokens" not in completion_kwargs


def test_call_api_openai_reasoning_keeps_max_completion_tokens():
    client = _build_client("openai/gpt-5-mini", GenerationParameters(max_new_tokens=96, reasoning_effort="low"))

    completion_kwargs = _run_call_api(client, supports_reasoning=True).call_args.kwargs
    assert completion_kwargs["max_tokens"] == 640
    assert completion_kwargs["max_completion_tokens"] == 96


@pytest.mark.parametrize("stop_sequence", [None, (), []], ids=["none", "empty-tuple", "empty-list"])
def test_call_api_sends_no_stop_when_task_has_no_stop_sequence(stop_sequence):
    # Tasks without stop sequences carry an empty tuple; OpenAI rejects `stop=[]`, so it must reach
    # litellm as None (which litellm drops) rather than as an empty container.
    client = _build_client("openai/gpt-4.1-nano", GenerationParameters())

    completion_kwargs = _run_call_api(client, supports_reasoning=False, stop_sequence=stop_sequence).call_args.kwargs
    assert completion_kwargs.get("stop") is None


def test_call_api_forwards_task_stop_sequence():
    client = _build_client("openai/gpt-4.1-nano", GenerationParameters())

    completion_kwargs = _run_call_api(client, supports_reasoning=False, stop_sequence=["\n"]).call_args.kwargs
    assert completion_kwargs["stop"] == ["\n"]


@pytest.mark.parametrize(
    "stop_sequence, expected_stop",
    [(["\n", "Answer:"], ["Answer:"]), (["\n"], None)],
    ids=["keeps-non-whitespace", "all-whitespace-becomes-none"],
)
def test_call_api_anthropic_drops_whitespace_only_stop_sequences(stop_sequence, expected_stop):
    client = _build_client("anthropic/claude-sonnet-4-5", GenerationParameters())
    client.provider = "anthropic"

    completion_kwargs = _run_call_api(client, supports_reasoning=False, stop_sequence=stop_sequence).call_args.kwargs
    assert completion_kwargs.get("stop") == expected_stop


def _build_client_for_greedy_until(model_name: str) -> LiteLLMClient:
    """A client wired for the full ``greedy_until`` -> ``litellm.acompletion`` path.

    On top of ``_build_client`` it sets the two attributes that path touches: a ``prompt_manager``
    (to build the message structure) and ``concurrent_requests`` (for the request semaphore).
    """
    client = _build_client(model_name, GenerationParameters())
    client.prompt_manager = PromptManager(use_chat_template=True, tokenizer=None, system_prompt=None)
    client.concurrent_requests = 1
    client._cache = None
    return client


@pytest.mark.parametrize(
    ("override_max_new_tokens", "expected_max_new_tokens"),
    [
        pytest.param(64, 64, id="override-wins"),
        pytest.param(None, 8, id="falls-back-to-doc"),
    ],
)
def test_greedy_until_max_new_tokens_precedence(override_max_new_tokens, expected_max_new_tokens):
    """generation_parameters.max_new_tokens (e.g. a pipeline reasoning override) must take
    precedence over a task's hardcoded doc.generation_size, mirroring VLLMModel's behavior."""
    client = _build_client_for_greedy_until("openai/gpt-4.1-nano")
    client.generation_parameters = GenerationParameters(max_new_tokens=override_max_new_tokens)

    doc = Doc(query="hi", choices=[], gold_index=0, generation_size=8)

    response = Mock()
    response.choices = [Mock(message=Mock(content="ok", reasoning_content=None))]

    with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
        with patch(
            "lighteval.models.endpoints.litellm_model.litellm.acompletion", return_value=response
        ) as completion:
            client.greedy_until([doc])

    assert completion.call_args.kwargs["max_tokens"] == expected_max_new_tokens


class TestImagePlacement:
    """Multimodal ``image_placement`` modes, end to end through ``greedy_until`` -> ``litellm.acompletion``."""

    def test_greedy_until_sends_image_in_completion_payload(self):
        client = _build_client_for_greedy_until("openai/gpt-4.1-nano")

        image = Image.new("RGB", (4, 4), (255, 0, 0))
        doc = Doc(query="What is in this image?", choices=[], gold_index=0, images=[image])

        response = Mock()
        response.choices = [Mock(message=Mock(content="ok", reasoning_content=None))]

        with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
            with patch(
                "lighteval.models.endpoints.litellm_model.litellm.acompletion", return_value=response
            ) as completion:
                client.greedy_until([doc])

        # The image must survive into the messages payload as a base64 PNG data URI.
        content = completion.call_args.kwargs["messages"][-1]["content"]
        image_parts = [part for part in content if part.get("type") == "image_url"]
        assert len(image_parts) == 1

    def test_greedy_until_inline_binds_marker_positionally(self):
        client = _build_client_for_greedy_until("openai/gpt-4.1-nano")
        client.prompt_manager = PromptManager(
            use_chat_template=True,
            tokenizer=None,
            system_prompt=None,
            image_placement="inline",
        )

        image_1 = Image.new("RGB", (4, 4), (255, 0, 0))
        image_2 = Image.new("RGB", (4, 4), (0, 255, 0))
        doc = Doc(
            query="Compare <image 1> with <image 2> please.",
            choices=[],
            gold_index=0,
            images=[image_1, image_2],
        )

        response = Mock()
        response.choices = [Mock(message=Mock(content="ok", reasoning_content=None))]

        with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
            with patch(
                "lighteval.models.endpoints.litellm_model.litellm.acompletion", return_value=response
            ) as completion:
                client.greedy_until([doc])

        # Exact text-splitting is covered by test_prompt_manager_class.TestOrderMultimodalContentPlacement;
        # this only needs to prove image_placement="inline" actually reaches _prepare_multimodal_context.
        content = completion.call_args.kwargs["messages"][-1]["content"]
        assert [part["type"] for part in content] == ["text", "image_url", "text", "image_url", "text"]

    def test_greedy_until_prepend_places_all_images_before_unmodified_text(self):
        client = _build_client_for_greedy_until("openai/gpt-4.1-nano")
        client.prompt_manager = PromptManager(
            use_chat_template=True,
            tokenizer=None,
            system_prompt=None,
            image_placement="prepend",
        )

        image_1 = Image.new("RGB", (4, 4), (255, 0, 0))
        image_2 = Image.new("RGB", (4, 4), (0, 255, 0))
        doc = Doc(
            query="<image 1> Compare with <image 2>.",
            choices=[],
            gold_index=0,
            images=[image_1, image_2],
        )

        response = Mock()
        response.choices = [Mock(message=Mock(content="ok", reasoning_content=None))]

        with patch("lighteval.models.endpoints.litellm_model.supports_reasoning", return_value=False):
            with patch(
                "lighteval.models.endpoints.litellm_model.litellm.acompletion", return_value=response
            ) as completion:
                client.greedy_until([doc])

        # Same rationale as the inline case above: exact text handling is already covered by
        # TestOrderMultimodalContentPlacement; this proves image_placement="prepend" reaches here.
        content = completion.call_args.kwargs["messages"][-1]["content"]
        assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
