# MIT License

# Copyright (c) 2025 The HuggingFace Team

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

import pytest

from lighteval.models.model_input import GenerationParameters


class TestGenerationParameters:
    @pytest.mark.parametrize(
        "model_args, expected",
        [
            (
                "generation_parameters={temperature: 0.7,top_p: 0.95},pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,max_model_length=32768,gpu_memory_utilisation=0.8",
                {"temperature": 0.7, "top_p": 0.95},
            ),
            (
                "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,generation_parameters={temperature: 0.7,top_p: 0.95},max_model_length=32768,gpu_memory_utilisation=0.8",
                {"temperature": 0.7, "top_p": 0.95},
            ),
            (
                "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=float16,data_parallel_size=4,max_model_length=32768,gpu_memory_utilisation=0.8,generation_parameters={temperature: 0.7,top_p: 0.95}",
                {"temperature": 0.7, "top_p": 0.95},
            ),
        ],
    )
    def test_extract_num_samples(self, model_args: str, expected):
        gen = GenerationParameters.from_model_args(model_args)
        for k, v in expected.items():
            assert getattr(gen, k) == v

    @pytest.mark.parametrize("reasoning_effort", ["low", "medium", "high"])
    def test_extract_reasoning_effort(self, reasoning_effort: str):
        model_args = (
            "pretrained=google/gemini-2.5-flash,"
            f'generation_parameters={{temperature: 0.2,reasoning_effort: "{reasoning_effort}"}},'
            "dtype=float16"
        )
        gen = GenerationParameters.from_model_args(model_args)

        assert gen.temperature == 0.2
        assert gen.reasoning_effort == reasoning_effort

    @pytest.mark.parametrize("reasoning_effort", ["low", "medium", "high"])
    def test_to_litellm_dict_includes_reasoning_effort(self, reasoning_effort: str):
        gen = GenerationParameters(temperature=0.2, top_p=0.9, reasoning_effort=reasoning_effort)

        assert gen.to_litellm_dict() == {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": reasoning_effort}

    def test_vllm_dict_excludes_reasoning_effort(self):
        gen = GenerationParameters(max_new_tokens=128, temperature=0.1, reasoning_effort="low")

        assert gen.to_vllm_dict() == {"max_tokens": 128, "temperature": 0.1}

    def test_vllm_dict_includes_skip_special_tokens(self):
        gen = GenerationParameters(max_new_tokens=128, temperature=0.1, skip_special_tokens=False)

        assert gen.to_vllm_dict() == {"max_tokens": 128, "temperature": 0.1, "skip_special_tokens": False}

    def test_vllm_openai_dict_excludes_reasoning_effort(self):
        gen = GenerationParameters(max_new_tokens=128, temperature=0.1, reasoning_effort="low")

        assert gen.to_vllm_openai_dict() == {"max_new_tokens": 128, "temperature": 0.1}


class TestToLitellmDict:
    """Tests for GenerationParameters.to_litellm_dict() — regression coverage for PR #1193."""

    def test_presence_penalty_included(self):
        """presence_penalty must not be silently dropped (bug fix PR #1193)."""
        gen = GenerationParameters(presence_penalty=0.4)
        result = gen.to_litellm_dict()
        assert "presence_penalty" in result
        assert result["presence_penalty"] == pytest.approx(0.4)

    def test_temperature_zero_included(self):
        """temperature=0 (the default) is included since 0 is not None."""
        gen = GenerationParameters()
        result = gen.to_litellm_dict()
        assert "temperature" in result
        assert result["temperature"] == 0

    def test_all_params_forwarded(self):
        """All chat-completion params are forwarded with correct key names."""
        gen = GenerationParameters(
            max_new_tokens=200,
            stop_tokens=["\n", "END"],
            temperature=0.8,
            top_p=0.95,
            seed=7,
            repetition_penalty=1.05,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )
        result = gen.to_litellm_dict()
        assert result["max_completion_tokens"] == 200
        assert result["stop"] == ["\n", "END"]
        assert result["temperature"] == pytest.approx(0.8)
        assert result["top_p"] == pytest.approx(0.95)
        assert result["seed"] == 7
        assert result["repetition_penalty"] == pytest.approx(1.05)
        assert result["frequency_penalty"] == pytest.approx(0.1)
        assert result["presence_penalty"] == pytest.approx(0.2)

    def test_none_values_excluded(self):
        """Parameters left as None are not forwarded."""
        gen = GenerationParameters(temperature=0.5)
        result = gen.to_litellm_dict()
        assert "max_completion_tokens" not in result
        assert "stop" not in result
        assert "seed" not in result
        assert "top_p" not in result
