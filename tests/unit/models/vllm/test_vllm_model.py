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
        model.config.generation_parameters.thinking_token_budget = None
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

    def test_loglikelihood_attaches_images_to_payload(self):
        model = self._bare_model()
        model._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model._add_special_tokens = False
        model.pairwise_tokenization = False
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

        # One payload per choice, each a multimodal dict carrying the image.
        self.assertEqual(len(captured["inputs"]), len(doc.choices))
        for payload in captured["inputs"]:
            self.assertIsInstance(payload, dict)
            self.assertTrue(payload["prompt"].startswith("Question <start_of_image>"))
            self.assertEqual(payload["multi_modal_data"]["image"], [image])


class TestVLLMRequestSeeds(unittest.TestCase):
    def test_each_prompt_gets_its_own_reproducible_seed(self):
        from lighteval.models.vllm.vllm_model import request_seeds

        prompts = [[1, 2, 3], [4, 5], [1, 2, 3], {"prompt": "Question <start_of_image>"}]
        seeds = request_seeds(1, prompts)
        self.assertEqual(seeds, request_seeds(1, prompts))  # reproducible
        self.assertEqual(seeds[0], seeds[2])  # the same prompt draws the same noise
        self.assertEqual(len({seeds[0], seeds[1], seeds[3]}), 3)  # different prompts don't
        self.assertNotEqual(seeds, request_seeds(2, prompts))  # another run seed, other noise
        self.assertTrue(all(0 <= seed < 2**32 for seed in seeds))

    def test_image_requests_with_the_same_text_draw_their_own_seeds(self):
        from lighteval.models.vllm.vllm_model import request_seeds

        red, blue = Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")
        text = "Question <start_of_image>"  # MMMU-Pro vision: the question is in the image
        requests = [{"prompt": text, "multi_modal_data": {"image": [image]}} for image in (red, blue, red.copy())]
        seeds = request_seeds(1, requests)
        self.assertNotEqual(seeds[0], seeds[1])  # the same text, other pixels
        self.assertEqual(seeds[0], seeds[2])  # the same pixels
        no_images = [{"prompt": text}, {"prompt": text, "multi_modal_data": {"image": []}}]
        self.assertEqual(len(set(request_seeds(1, no_images))), 1)

    def test_palette_images_with_the_same_indices_but_other_colours_draw_their_own_seeds(self):
        from lighteval.models.vllm.vllm_model import request_seeds

        red, blue = Image.new("P", (4, 4)), Image.new("P", (4, 4))
        red.putpalette([255, 0, 0] * 256)
        blue.putpalette([0, 0, 255] * 256)
        self.assertEqual(red.tobytes(), blue.tobytes())  # the same palette indices
        text = "Question <start_of_image>"
        requests = [{"prompt": text, "multi_modal_data": {"image": [image]}} for image in (red, blue)]
        seeds = request_seeds(1, requests)
        self.assertNotEqual(seeds[0], seeds[1])

    def test_async_requests_are_seeded_as_the_batch_path_seeds_them(self):
        import asyncio

        from lighteval.models.vllm import vllm_model

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update({"seed": None, "temperature": 1.0, **kwargs})

        sent = {}

        async def generate(request_id, prompt, sampling_params):
            sent[request_id] = (prompt, sampling_params.seed)
            yield "out"

        model = object.__new__(vllm_model.AsyncVLLMModel)
        model.config = Mock()
        model.config.generation_parameters.to_vllm_dict.return_value = {"seed": 7, "temperature": 1.0}
        model.config.generation_parameters.max_new_tokens = 8
        model.prompt_manager = Mock()
        model.prompt_manager.prepare_prompt.return_value = "Hello"
        model.prompt_manager.prepare_prompt_multimodal.return_value = "Look <start_of_image>"
        model._tokenizer = Mock(return_value={"input_ids": [5, 6]})
        model._add_special_tokens = False
        model.use_chat_template = True
        model.model = Mock()
        model.model.generate = generate
        image = Image.new("RGB", (4, 4), "red")
        with patch.object(vllm_model, "SamplingParams", FakeSamplingParams, create=True):
            asyncio.run(model._async_one_item(0, Doc(query="Hello", choices=[], gold_index=0), generative=True))
            asyncio.run(
                model._async_one_item(1, Doc(query="Look", choices=[], gold_index=0, images=[image]), generative=True)
            )

        self.assertEqual(sent["generative_0"][1], vllm_model.request_seeds(7, [[5, 6]])[0])
        image_prompt, image_seed = sent["generative_1"]
        self.assertEqual(image_seed, vllm_model.request_seeds(7, [image_prompt])[0])
        self.assertEqual(image_prompt["multi_modal_data"]["image"], [image])

    def test_generate_sends_one_seeded_sampling_params_per_request(self):
        from lighteval.models.vllm import vllm_model

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update({"seed": None, "temperature": 1.0, **kwargs})

        model = object.__new__(vllm_model.VLLMModel)
        model.config = Mock()
        model.config.generation_parameters.to_vllm_dict.return_value = {"seed": 7, "temperature": 1.0}
        model.data_parallel_size = 1
        model.model = Mock()
        model.model.generate.return_value = ["out-a", "out-b"]
        with (
            patch.dict(sys.modules, TestVLLMPromptConstruction._fake_vllm_modules()),
            patch.object(vllm_model, "SamplingParams", FakeSamplingParams, create=True),
        ):
            model._generate([[1, 2], [3]], max_new_tokens=8)

        sent = model.model.generate.call_args.kwargs["sampling_params"]
        self.assertEqual([params.seed for params in sent], vllm_model.request_seeds(7, [[1, 2], [3]]))
        self.assertTrue(all(params.max_tokens == 8 for params in sent))


class TestVLLMThinkingBudget(unittest.TestCase):
    """A thinking budget: vLLM's per-request budget, the engine's reasoning config, and a split's token limit."""

    GEMMA = {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}

    def test_a_budget_needs_a_reasoning_config_of_two_strings(self):
        from pydantic import ValidationError

        from lighteval.models.model_input import GenerationParameters

        budgeted = GenerationParameters(thinking_token_budget=512)
        with self.assertRaises(ValidationError):
            VLLMModelConfig(model_name="m", generation_parameters=budgeted)
        with self.assertRaises(ValidationError):
            VLLMModelConfig(model_name="m", reasoning_config={"reasoning_start_str": "<think>"})
        config = VLLMModelConfig(model_name="m", generation_parameters=budgeted, reasoning_config=self.GEMMA)
        self.assertEqual(config.reasoning_config, self.GEMMA)
        self.assertIn("reasoning_config", config.model_dump())  # so in the response cache's key

    def _model(self, **generation):
        from lighteval.models.model_input import GenerationParameters

        model = VLLMModel.__new__(VLLMModel)
        model.config = VLLMModelConfig(
            model_name="m",
            generation_parameters=GenerationParameters(**generation),
            reasoning_config=self.GEMMA if "thinking_token_budget" in generation else None,
        )
        model._tokenizer = Mock()
        model._tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [7] * len(text.split("|"))
        return model

    def test_a_splits_limit_is_the_budget_the_markers_and_the_tasks_room_within_the_configured_one(self):
        # each marker encodes to two ids here ("<", "channel>" split at "|"): 4 marker tokens in all
        self.assertEqual(self._model(max_new_tokens=10240)._max_new_tokens(1280), 10240)  # no budget
        self.assertEqual(self._model()._max_new_tokens(1280), 1280)
        self.assertEqual(self._model(max_new_tokens=10240, thinking_token_budget=2048)._max_new_tokens(1280), 3332)
        self.assertEqual(self._model(max_new_tokens=3000, thinking_token_budget=2048)._max_new_tokens(1280), 3000)
        self.assertEqual(self._model(thinking_token_budget=2048)._max_new_tokens(15), 2067)
        self.assertEqual(self._model(max_new_tokens=10240, thinking_token_budget=2048)._max_new_tokens(None), 10240)
        # a task that sets no room (generation_size=-1) adds nothing to the budget: the configured limit stands
        self.assertEqual(self._model(max_new_tokens=10240, thinking_token_budget=2048)._max_new_tokens(-1), 10240)
        self.assertEqual(self._model(thinking_token_budget=2048)._max_new_tokens(-1), -1)  # as without a budget

    def test_greedy_until_asks_each_split_for_its_own_limit(self):
        model = self._model(max_new_tokens=10240, thinking_token_budget=512)
        model.DATASET_SPLITS = 1
        model.use_chat_template = True
        model._max_length = None
        model.prompt_manager = Mock()
        model.prompt_manager.prepare_prompt_multimodal.return_value = "Question <start_of_image>"
        image = Image.new("RGB", (4, 4), (255, 0, 0))
        asked = []

        def fake_generate(inputs, max_new_tokens=None, **kwargs):
            asked.append(max_new_tokens)
            output = Mock()
            output.outputs = [Mock(token_ids=[1], text="A")]
            output.prompt_token_ids = [1]
            return [output for _ in inputs]

        model._generate = fake_generate
        docs = [Doc(query="q", choices=[], gold_index=0, images=[image], generation_size=size) for size in (15, 1280)]
        model._greedy_until(docs)
        self.assertEqual(sorted(asked), [512 + 4 + 15, 512 + 4 + 1280])

    def test_the_engine_gets_the_reasoning_config_and_the_async_engine_refuses_it(self):
        from lighteval.models.model_input import GenerationParameters
        from lighteval.models.vllm import vllm_model

        class FakeReasoningConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_config = ModuleType("vllm.config")
        fake_config.ReasoningConfig = FakeReasoningConfig
        config = VLLMModelConfig(
            model_name="m",
            data_parallel_size=2,  # the data-parallel path builds its workers from model_args and returns no engine
            generation_parameters=GenerationParameters(thinking_token_budget=512),
            reasoning_config=self.GEMMA,
        )
        model = VLLMModel.__new__(VLLMModel)
        model._max_length = 4096
        with patch.dict(sys.modules, {**TestVLLMPromptConstruction._fake_vllm_modules(), "vllm.config": fake_config}):
            self.assertIsNone(model._create_auto_model(config))
        self.assertEqual(model.model_args["reasoning_config"].kwargs, self.GEMMA)
        with self.assertRaises(ValueError):
            vllm_model.AsyncVLLMModel.__new__(vllm_model.AsyncVLLMModel)._create_auto_model(config)

    def test_async_scheduling_is_off_where_the_budget_reads_unfilled_outputs(self):
        # vLLM 0.19's budget misses a closer still a placeholder under async scheduling and forces a second one; 0.23's
        # budget reads the output after the copy and keeps async scheduling
        from lighteval.models.model_input import GenerationParameters
        from lighteval.models.vllm import vllm_model

        fake_config = ModuleType("vllm.config")
        fake_config.ReasoningConfig = lambda **kwargs: kwargs
        modules = {**TestVLLMPromptConstruction._fake_vllm_modules(), "vllm.config": fake_config}
        budgeted = {
            "generation_parameters": GenerationParameters(thinking_token_budget=512),
            "reasoning_config": self.GEMMA,
        }
        for reads_unfilled, expected in ((True, False), (False, None)):
            model = VLLMModel.__new__(VLLMModel)
            model._max_length = 4096
            config = VLLMModelConfig(model_name="m", data_parallel_size=2, **budgeted)
            with (
                patch.dict(sys.modules, modules),
                patch.object(vllm_model, "_budget_reads_unfilled_outputs", return_value=reads_unfilled),
            ):
                model._create_auto_model(config)
            self.assertEqual(model.model_args.get("async_scheduling"), expected)
        model = VLLMModel.__new__(VLLMModel)
        model._max_length = 4096
        with patch.dict(sys.modules, modules):
            model._create_auto_model(VLLMModelConfig(model_name="m", data_parallel_size=2))
        self.assertNotIn("async_scheduling", model.model_args)  # no budget: vLLM's default
        model = VLLMModel.__new__(VLLMModel)
        model._max_length = 4096
        with (
            patch.dict(sys.modules, modules),
            patch.object(vllm_model, "_budget_reads_unfilled_outputs", return_value=True),
        ):
            model._create_auto_model(
                VLLMModelConfig(model_name="m", data_parallel_size=2, reasoning_config=self.GEMMA)
            )
        self.assertNotIn("async_scheduling", model.model_args)  # markers without a budget: vLLM's default too

    def test_inline_model_args_carry_a_reasoning_config(self):
        config = VLLMModelConfig.from_args(
            'model_name=m,reasoning_config={reasoning_start_str:"<think>",reasoning_end_str:"</think>"},'
            "generation_parameters={thinking_token_budget:512}"
        )
        self.assertEqual(config.reasoning_config, {"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"})
        self.assertEqual(config.generation_parameters.thinking_token_budget, 512)
