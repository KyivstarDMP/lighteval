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

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from PIL import Image
from transformers import AutoTokenizer

from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig, build_vllm_token_prompts
from lighteval.tasks.requests import Doc


class TestVLLMPromptConstruction(unittest.TestCase):
    @staticmethod
    def _fake_vllm_modules():
        fake_inputs = ModuleType("vllm.inputs")
        fake_inputs.TokensPrompt = lambda *, prompt_token_ids: {  # noqa: E731
            "kind": "tokens_prompt",
            "prompt_token_ids": prompt_token_ids,
        }
        fake_vllm = ModuleType("vllm")
        fake_vllm.inputs = fake_inputs
        return {"vllm": fake_vllm, "vllm.inputs": fake_inputs}

    def test_build_vllm_token_prompts_uses_tokens_prompt_when_available(self):
        with patch.dict(sys.modules, self._fake_vllm_modules()):
            prompts = build_vllm_token_prompts([[1, 2], [3]])

        self.assertEqual(
            prompts,
            [
                {"kind": "tokens_prompt", "prompt_token_ids": [1, 2]},
                {"kind": "tokens_prompt", "prompt_token_ids": [3]},
            ],
        )

    def test_build_vllm_token_prompts_passes_multimodal_dicts_through(self):
        multimodal = {"prompt": "Question <start_of_image>", "multi_modal_data": {"image": ["img"]}}

        with patch.dict(sys.modules, self._fake_vllm_modules()):
            prompts = build_vllm_token_prompts([[1, 2], multimodal])

        self.assertEqual(prompts[0], {"kind": "tokens_prompt", "prompt_token_ids": [1, 2]})
        self.assertIs(prompts[1], multimodal)


class TestVLLMTokenizerCreation(unittest.TestCase):
    def test_tokenizer_created_with_correct_revision(self):
        config = VLLMModelConfig(
            model_name="lighteval/different-chat-templates-per-revision", revision="new_chat_template"
        )
        vllm_tokenizer = VLLMModel.__new__(VLLMModel)._create_auto_tokenizer(config)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.revision,
        )
        self.assertEqual(vllm_tokenizer.chat_template, tokenizer.chat_template)


class TestVLLMModelUseChatTemplate(unittest.TestCase):
    @patch("lighteval.models.vllm.vllm_model.VLLMModel._create_auto_model")
    def test_vllm_model_use_chat_template_with_different_model_names(self, mock_create_model):
        """Test that VLLMModel correctly calls uses_chat_template with different model names."""
        test_cases = [
            ("Qwen/Qwen3-0.6B", True),
            ("gpt2", False),
        ]

        for model_name, expected_result in test_cases:
            with self.subTest(model_name=model_name):
                # We skip the model creation phase
                mock_create_model.return_value = Mock()

                config = VLLMModelConfig(model_name=model_name)
                model = VLLMModel(config)

                self.assertEqual(model.use_chat_template, expected_result)
                self.assertEqual(model.use_chat_template, model._tokenizer.chat_template is not None)


class TestVLLMMultimodalPayload(unittest.TestCase):
    """The image carried by a Doc must survive into the vLLM prompt payload as
    ``multi_modal_data`` for every vLLM code path (generative and loglikelihood)."""

    @staticmethod
    def _bare_model() -> VLLMModel:
        model = VLLMModel.__new__(VLLMModel)
        model.DATASET_SPLITS = 1
        model.use_chat_template = True
        model.prompt_manager = Mock()
        return model

    def test_greedy_until_attaches_images_to_payload(self):
        model = self._bare_model()
        model.config = Mock()
        model.config.generation_parameters.max_new_tokens = 16
        model._max_length = None  # all-image split -> no tokenization/truncation
        model.prompt_manager.prepare_prompt_multimodal.return_value = "Question <start_of_image>"

        image = Image.new("RGB", (4, 4), (255, 0, 0))
        doc = Doc(query="q", choices=[], gold_index=0, images=[image], generation_size=16)

        captured = {}

        def fake_generate(inputs, **kwargs):
            captured["inputs"] = inputs
            output = Mock()
            output.outputs = [Mock(token_ids=[1], text="Відповідь: А")]
            output.prompt_token_ids = [1, 2, 3]
            return [output]

        model._generate = fake_generate
        model._greedy_until([doc])

        payload = captured["inputs"][0]
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["prompt"], "Question <start_of_image>")
        self.assertEqual(payload["multi_modal_data"]["image"], [image])

    def _loglikelihood_payloads(self, token_prompts: bool):
        model = self._bare_model()
        model._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model._add_special_tokens = False
        model.pairwise_tokenization = False
        model.config = Mock()
        model.config.loglikelihood_prefix_cache = False  # exercise the legacy ``_generate`` path
        model.config.loglikelihood_multimodal_token_prompts = token_prompts
        model.prompt_manager.prepare_prompt_multimodal.return_value = "Question <start_of_image>\n"

        image = Image.new("RGB", (4, 4), (0, 255, 0))
        doc = Doc(query="q", choices=["А", "Б"], gold_index=0, images=[image])

        captured = {}

        class _AnyLogprobs(dict):
            """Returns a rank-1 logprob for whatever continuation token is queried."""

            def __getitem__(self, _token):
                return Mock(rank=1, logprob=-0.1)

        def fake_generate(inputs, **kwargs):
            captured["inputs"] = inputs
            outputs = []
            for _ in inputs:
                output = Mock()
                output.prompt_token_ids = list(range(32))
                output.prompt_logprobs = [_AnyLogprobs() for _ in range(32)]
                outputs.append(output)
            return outputs

        model._generate = fake_generate
        model._loglikelihood_tokens([doc])
        return model, doc, image, captured["inputs"]

    def test_loglikelihood_attaches_images_to_payload(self):
        model, doc, image, payloads = self._loglikelihood_payloads(token_prompts=False)

        # Default: one text payload per choice, each a multimodal dict carrying the image (the
        # pre-existing behaviour, kept bit-for-bit: original context string + decoded continuation).
        self.assertEqual(len(payloads), len(doc.choices))
        for payload, choice in zip(payloads, doc.choices):
            self.assertIsInstance(payload, dict)
            self.assertNotIn("prompt_token_ids", payload)
            self.assertEqual(payload["prompt"], "Question <start_of_image>\n" + "\n" + choice)
            self.assertEqual(payload["multi_modal_data"]["image"], [image])

    def test_loglikelihood_token_prompts_carry_images_and_exact_continuation(self):
        model, doc, image, payloads = self._loglikelihood_payloads(token_prompts=True)

        # ``loglikelihood_multimodal_token_prompts``: pre-tokenized context + continuation plus the
        # image (token ids, not text: no BOS from vLLM's tokenizer, trailing whitespace scored once).
        self.assertEqual(len(payloads), len(doc.choices))
        context_ids = model._tokenizer.encode("Question <start_of_image>", add_special_tokens=False)
        for payload, choice in zip(payloads, doc.choices):
            self.assertIsInstance(payload, dict)
            self.assertNotIn("prompt", payload)
            self.assertEqual(payload["prompt_token_ids"][: len(context_ids)], context_ids)
            self.assertEqual(model._tokenizer.decode(payload["prompt_token_ids"][len(context_ids) :]), "\n" + choice)
            self.assertEqual(payload["multi_modal_data"]["image"], [image])
