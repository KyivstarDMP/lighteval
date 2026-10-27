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

import asyncio
import copy
import gc
import hashlib
import itertools
import logging
import os
from typing import ClassVar, Coroutine, Optional

import torch
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.models.utils import _simplify_name, uses_chat_template
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import is_package_available, requires


logger = logging.getLogger(__name__)


def build_vllm_token_prompts(inputs: list) -> list:
    """Build vLLM prompts across prompt-schema reorganizations.

    Text-only items arrive as token-id lists and are wrapped in ``TokensPrompt``;
    multimodal items arrive as already-built prompt dicts ({"prompt": ...,
    "multi_modal_data": ...}) and pass through untouched so vLLM's HF processor
    handles image fusion.
    """
    from vllm.inputs import TokensPrompt

    return [item if isinstance(item, dict) else TokensPrompt(prompt_token_ids=item) for item in inputs]


# How each request's seed is made. A new value whenever that changes: it is part of the response cache's identity
# (``VLLMModelConfig.cache_version``), so responses sampled under another scheme are never reused.
REQUEST_SEEDS = "sha256 of the run seed and the request: its token ids, or its text and image pixels"


def request_seeds(seed: int, inputs: list) -> list[int]:
    """Return one sampling seed per request, from the run's seed and the request.

    vLLM gives each seeded request its own generator, so requests that share a seed draw the same noise at every
    step and their samples are correlated: a run is not a set of independent samples, and changing the seed moves
    every row the same way. With a seed per request, different requests draw different noise, and a request draws
    the same noise in every run with this seed, whatever the batch (its output can still differ, since vLLM's
    numerics depend on the batch). A request is its token ids, or a multimodal prompt's text and images, so image
    questions that share their text don't share their noise.
    """
    return [
        int.from_bytes(hashlib.sha256(f"{seed}:{_request_key(item)}".encode()).digest()[:4], "big") for item in inputs
    ]


def _request_key(item) -> str:
    if not isinstance(item, dict):
        return ",".join(map(str, item))  # token ids
    images = (item.get("multi_modal_data") or {}).get("image") or []
    images = images if isinstance(images, (list, tuple)) else [images]
    return item.get("prompt", "") + "".join(f"\0{_image_digest(image)}" for image in images)


def _image_digest(image) -> str:
    """Return a digest of an image's pixels (a PIL image), else of its bytes."""
    if hasattr(image, "tobytes"):
        data = f"{image.mode}:{image.size}:".encode() + image.tobytes()
    else:
        data = bytes(image) if isinstance(image, (bytes, bytearray)) else repr(image).encode()
    return hashlib.sha256(data).hexdigest()


def _with_seed(sampling_params, seed: int):
    params = copy.deepcopy(sampling_params)
    params.seed = seed
    return params


if is_package_available("vllm"):
    import ray
    from more_itertools import distribute
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM

    try:
        # vLLM moved `get_tokenizer` to `vllm.tokenizers` in v0.12.0.
        # Keep the fallback while our lower bound remains on v0.11.x.
        from vllm.tokenizers import get_tokenizer
    except ModuleNotFoundError:
        from vllm.transformers_utils.tokenizer import get_tokenizer

    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").handlers.clear()

    logging.getLogger("ray").propagate = True
    logging.getLogger("ray").handlers.clear()
else:
    from unittest.mock import Mock

    LLM = SamplingParams = get_tokenizer = ray = distribute = destroy_distributed_environment = (
        destroy_model_parallel
    ) = Mock()
    AsyncLLM = AsyncEngineArgs = RequestOutput = Mock()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class VLLMModelConfig(ModelConfig):
    """Configuration class for VLLM inference engine.

    This configuration is used to load and configure models using the VLLM inference engine,
    which provides high-performance inference for large language models with features like
    PagedAttention, continuous batching, and efficient memory management.

    vllm doc: https://docs.vllm.ai/en/v0.7.1/serving/engine_args.html

    Attributes:
        model_name (str):
            HuggingFace Hub model ID or path to the model to load.
        tokenizer (str | None):
            HuggingFace Hub model ID or path to the tokenizer to load.
        revision (str):
            Git revision of the model. Defaults to "main".
        dtype (str):
            Data type for model weights. Defaults to "bfloat16". Options: "float16", "bfloat16", "float32".
        tensor_parallel_size (PositiveInt):
            Number of GPUs to use for tensor parallelism. Defaults to 1.
        data_parallel_size (PositiveInt):
            Number of GPUs to use for data parallelism. Defaults to 1.
        pipeline_parallel_size (PositiveInt):
            Number of GPUs to use for pipeline parallelism. Defaults to 1.
        gpu_memory_utilization (NonNegativeFloat):
            Fraction of GPU memory to use. Lower this if running out of memory. Defaults to 0.9.
        enable_prefix_caching (bool):
            Whether to enable prefix caching to speed up generation. May use more memory. Should be disabled for LFM2. Defaults to True.
        max_model_length (PositiveInt | None):
            Maximum sequence length for the model. If None, automatically inferred.
            Reduce this if encountering OOM issues (4096 is usually sufficient).
        quantization (str | None):
            Quantization method.
        load_format (str | None):
            The format of the model weights to load. choices: auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral, runai_streamer.
        swap_space (PositiveInt):
            CPU swap space size in GiB per GPU. Defaults to 4.
        seed (NonNegativeInt):
            Random seed for reproducibility. Defaults to 1234.
        trust_remote_code (bool):
            Whether to trust remote code when loading models. Defaults to False.
        add_special_tokens (bool):
            Whether to add special tokens during tokenization. Defaults to True.
        multichoice_continuations_start_space (bool):
            Whether to add a space before multiple choice continuations. Defaults to True.
        pairwise_tokenization (bool):
            Whether to tokenize context and continuation separately for loglikelihood evals. Defaults to False.
        max_num_seqs (PositiveInt):
            Maximum number of sequences per iteration. Controls batch size at prefill stage. Defaults to 128.
        max_num_batched_tokens (PositiveInt):
            Maximum number of tokens per batch. Defaults to 2048.
        subfolder (str | None):
            Subfolder within the model repository. Defaults to None.
        is_async (bool):
            Whether to use the async version of VLLM. Defaults to False.
        override_chat_template (bool):
            If True, we force the model to use a chat template. If alse, we prevent the model from using
            a chat template. If None, we use the default (true if present in the tokenizer, false otherwise)
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.
        reasoning_config (dict[str, str] | None, optional, defaults to None):
            What opens and closes the model's thought, as vLLM's `ReasoningConfig` takes them
            (`reasoning_start_str`, `reasoning_end_str`). Needed for `generation_parameters.thinking_token_budget`:
            vLLM counts a thought's tokens from the opener and, at the budget, forces the closer. The closer is also
            how vLLM tells that a thought closed by itself, so give the bare closer.

    Example:
        ```python
        config = VLLMModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_length=4096,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    cache_version: ClassVar[str | None] = REQUEST_SEEDS  # a seeded run's responses depend on how requests are seeded
    tokenizer: str | None = None
    revision: str = "main"  # revision of the model
    dtype: str = "bfloat16"
    tensor_parallel_size: PositiveInt = 1  # how many GPUs to use for tensor parallelism
    data_parallel_size: PositiveInt = 1  # how many GPUs to use for data parallelism
    pipeline_parallel_size: PositiveInt = 1  # how many GPUs to use for pipeline parallelism
    gpu_memory_utilization: NonNegativeFloat = 0.9  # lower this if you are running out of memory
    enable_prefix_caching: bool = None  # whether to enable prefix caching to speed up generation. May use more memory. Should be disabled for LFM2
    max_model_length: PositiveInt | None = (
        None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    )
    quantization: str | None = None
    load_format: str | None = None
    swap_space: PositiveInt = 4  # CPU swap space size (GiB) per GPU.
    seed: NonNegativeInt = 1234
    trust_remote_code: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = (
        True  # whether to add a space at the start of each continuation in multichoice generation
    )
    pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.
    max_num_seqs: PositiveInt = 128  # maximum number of sequences per iteration; This variable and `max_num_batched_tokens` effectively control the batch size at prefill stage. See https://github.com/vllm-project/vllm/issues/2492 for detailed explaination.
    max_num_batched_tokens: PositiveInt = 2048  # maximum number of tokens per batch
    subfolder: str | None = None
    is_async: bool = False  # Whether to use the async version or sync version of the model
    override_chat_template: bool = None
    skip_mm_profiling: bool = False
    attention_backend: str | None = None
    mm_encoder_attn_backend: str | None = None
    reasoning_config: dict[str, str] | None = None  # vLLM's ReasoningConfig, for a thinking budget

    @model_validator(mode="after")
    def _budget_needs_a_reasoning_config(self) -> "VLLMModelConfig":
        if self.generation_parameters.thinking_token_budget is not None and not self.reasoning_config:
            raise ValueError(
                "generation_parameters.thinking_token_budget needs reasoning_config: the strings that open and close "
                "a thought (reasoning_start_str, reasoning_end_str)"
            )
        if self.reasoning_config and set(self.reasoning_config) != {"reasoning_start_str", "reasoning_end_str"}:
            raise ValueError("reasoning_config takes reasoning_start_str and reasoning_end_str")
        return self


@requires("vllm")
class VLLMModel(LightevalModel):
    def __init__(
        self,
        config: VLLMModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self.config = config
        self.use_chat_template = uses_chat_template(
            model_name=config.model_name, override_chat_template=config.override_chat_template
        )
        self.data_parallel_size = config.data_parallel_size
        self.tensor_parallel_size = config.tensor_parallel_size
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config)

        self._max_length = (
            config.max_model_length
        )  # will be None if the config is None, then defined in _create_auto_model

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = self._create_auto_model(config)

        # self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        self.model_name = _simplify_name(config.model_name)
        self.model_sha = ""
        self.precision = config.dtype

        self.pairwise_tokenization = config.pairwise_tokenization

        self.prompt_manager = PromptManager(
            self.use_chat_template,
            self.tokenizer,
            config.system_prompt,
            config.chat_template_kwargs,
            config.image_placement,
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    @property
    def tokenizer(self):
        return self._tokenizer

    def cleanup(self):
        destroy_model_parallel()
        if self.model is not None:
            del self.model
        gc.collect()
        ray.shutdown()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def _create_auto_model(self, config: VLLMModelConfig) -> Optional[LLM]:
        """Creates an instance of the pretrained HF model.

        Args:
            config (VLLMModelConfig): The VLLM model configuration.

        Returns:
            Optional[LLM]: The created auto model instance.
        """
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enable_prefix_caching": config.enable_prefix_caching,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
            "enforce_eager": True,
            "skip_mm_profiling": config.skip_mm_profiling,
        }

        if config.quantization is not None:
            self.model_args["quantization"] = config.quantization
        if config.load_format is not None:
            self.model_args["load_format"] = config.load_format
        if config.attention_backend is not None:
            self.model_args["attention_backend"] = config.attention_backend
        if config.mm_encoder_attn_backend is not None:
            self.model_args["mm_encoder_attn_backend"] = config.mm_encoder_attn_backend
        if config.reasoning_config is not None:  # the single engine and the data-parallel workers build from these
            from vllm.config import ReasoningConfig

            self.model_args["reasoning_config"] = ReasoningConfig(**config.reasoning_config)

        if config.data_parallel_size > 1:
            self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"

            if self._max_length is None:
                # Todo: we will want to manage this automatically - atm this arg must be set at least 2 times (in gen params + model args) for
                # vllm models, which is an issue.
                logger.warning(
                    "The model max_length was not set in the model arguments. Since the model is using data parallelism, it is created later "
                    " with `ray`, so we can't infer the max_length automatically atm. It might raise issues later on: if it does, relaunch your "
                    "run, but set `max_model_length` explicitely in the model args."
                )
            return None

        model = LLM(**self.model_args)

        # If the max_length can't get extracted from the config, it will be inferred from the model
        # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
        # config and tk config, like mistralai/Mistral-7B-v0.1
        if self._max_length is None:
            self._max_length = model.llm_engine.model_config.max_model_len

        return model

    def _max_new_tokens(self, room: int | None) -> int | None:
        """Return a split's token limit: the configured one, else the task's own room (``generation_size``).

        With a thinking budget, the thought takes at most the budget and the answer keeps the task's room, so the limit
        is the budget, the thought's opener and closer and the room, within the configured limit.
        """
        configured = self.config.generation_parameters.max_new_tokens
        budget = self.config.generation_parameters.thinking_token_budget
        if budget is None or room is None:
            return configured or room
        limit = budget + self._thought_marker_tokens() + room
        return min(configured, limit) if configured else limit

    def _thought_marker_tokens(self) -> int:
        """Return the tokens of the thought's opener and closer, which a thinking budget doesn't count."""
        strings = self.config.reasoning_config or {}
        return sum(
            len(self.tokenizer.encode(strings[key], add_special_tokens=False))
            for key in ("reasoning_start_str", "reasoning_end_str")
            if strings.get(key)
        )

    def _create_auto_tokenizer(self, config: VLLMModelConfig):
        tokenizer = get_tokenizer(
            config.tokenizer or config.model_name,  # use HF tokenizer for non-HF models, like GGUF model.
            tokenizer_mode="auto",
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
        )

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        return self._greedy_until(docs)

    @staticmethod
    def _multimodal_prompt(prompt: str, doc: Doc) -> dict:
        """Build a vLLM multimodal prompt: chat-template text plus the document's images.

        The text already carries the model's image placeholder tokens (inserted by
        ``prepare_prompt_multimodal``); vLLM's HF processor expands them and fuses the
        images supplied in ``multi_modal_data``.
        """
        return {"prompt": prompt, "multi_modal_data": {"image": list(doc.images)}}

    def _greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
            if self.use_chat_template:
                stop_tokens = []
            else:
                # NOTE: we are assuming all items in a batch behave similarly (same
                # stop_tokens and max_tokens genrated) which is not necessarily
                # the case! Because of that we only use batch size of 1
                stop_tokens = split[0].stop_sequences or []

            max_new_tokens = self._max_new_tokens(split[0].generation_size)
            num_samples = split[0].num_samples

            # Build vLLM prompts per doc: docs carrying images become multimodal prompt
            # dicts (text + images) handled by vLLM's HF processor, so they are not
            # pre-tokenized or truncated here; text-only docs follow the token-id path.
            context = []
            inputs = []
            for doc in split:
                if doc.images:
                    prompt = self.prompt_manager.prepare_prompt_multimodal(doc)
                    inputs.append(self._multimodal_prompt(prompt, doc))
                else:
                    prompt = self.prompt_manager.prepare_prompt(doc)
                    inputs.append(None)  # placeholder, filled with token ids below
                context.append(prompt)

            text_indices = [i for i, doc in enumerate(split) if not doc.images]
            if text_indices:
                tokenized = self.tokenizer(
                    [context[i] for i in text_indices], add_special_tokens=self.add_special_tokens
                )["input_ids"]

                # The main question for this step is the following:
                # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
                # of losing some meaning, or have some generations that are exceedingly short?
                # The choice we go for here is to avoid truncating the prompt if we can, since it
                # should have been managed by the prompt creator/few shot manager if requested by the user.
                context_size = len(tokenized[0])

                # left truncate the inputs to the maximum length
                if self.max_length is None:
                    logger.warning(
                        "The model max_length was not set in the model arguments, so we cannot check if we need to truncate the context."
                    )
                elif max_new_tokens is not None:
                    if context_size + max_new_tokens > self.max_length:
                        logger.warning(
                            f"{context_size + max_new_tokens=} which is greater than {self.max_length=}. Truncating context to {self.max_length - max_new_tokens} tokens."
                        )
                        context_size = self.max_length - max_new_tokens
                        if context_size < 0:
                            logger.critical(
                                f"{context_size=} is less than 0, either reduce the max_new_tokens or increase model max length."
                            )
                            raise ValueError("Context size is less than 0.")
                        tokenized = [input[-context_size:] for input in tokenized]
                else:
                    if context_size > self.max_length:
                        logger.warning(
                            f"{context_size=} which is greater than {self.max_length=}. Truncating context to {self.max_length} tokens."
                        )
                        context_size = self.max_length
                        tokenized = [input[-context_size:] for input in tokenized]

                for position, i in enumerate(text_indices):
                    inputs[i] = tokenized[position]

            vllm_outputs = self._generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                returns_logits=False,
                num_samples=num_samples,
            )

            for i, vllm_output in enumerate(vllm_outputs):
                output_token_ids = [outputs.token_ids for outputs in vllm_output.outputs]
                result = [output.text for output in vllm_output.outputs]
                input_token_ids = vllm_output.prompt_token_ids

                cur_response = ModelResponse(
                    input=context[i],
                    text=result,
                    output_tokens=list(output_token_ids),
                    input_tokens=input_token_ids,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    def _generate(
        self,
        inputs: list[list[int]],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
    ) -> list:
        """Contains the actual logic of the generation."""

        if generate:
            sampling_params = SamplingParams(**self.config.generation_parameters.to_vllm_dict())
            sampling_params.n = num_samples
            sampling_params.max_tokens = max_new_tokens
            sampling_params.stop = stop_tokens
            sampling_params.logprobs = 1 if returns_logits else 0
            if num_samples > 1 and sampling_params.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )
        else:
            sampling_params = SamplingParams(
                temperature=0.0,
                prompt_logprobs=1,
                max_tokens=1,
                detokenize=False,
            )
        if generate and sampling_params.seed is not None:
            sampling_params = [
                _with_seed(sampling_params, seed) for seed in request_seeds(sampling_params.seed, inputs)
            ]
        per_request = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(inputs)

        if self.data_parallel_size > 1:

            @ray.remote(num_gpus=self.tensor_parallel_size)
            def run_inference_one_model(model_args: dict, requests):
                llm = LLM(**model_args)
                prompts = build_vllm_token_prompts([prompt for prompt, _ in requests])
                return llm.generate(prompts=prompts, sampling_params=[params for _, params in requests])

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, list(zip(inputs, per_request)))]
            object_refs = [run_inference_one_model.remote(self.model_args, req) for req in requests]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            outputs = [
                x
                for x in itertools.chain.from_iterable(itertools.zip_longest(*[list(x) for x in results]))
                if x is not None
            ]
        else:
            prompts = build_vllm_token_prompts(inputs)
            outputs = self.model.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
            )

        return outputs

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        return self._loglikelihood_tokens(docs)

    def _loglikelihood_tokens(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=1)
        res = []

        for split in tqdm(dataset.splits_iterator()):
            contexts = [
                self.prompt_manager.prepare_prompt_multimodal(doc)
                if doc.images
                else self.prompt_manager.prepare_prompt(doc)
                for doc in split
            ]

            inputs = []
            tokenized_continuations_batch = []
            tokenized_contexts_batch = []

            for context, doc in zip(contexts, split):
                tokenized_contexts, tokenized_continuations = self.tok_encode_pair(
                    context, doc.choices, pairwise=self.pairwise_tokenization
                )
                for tokenized_context, tokenized_continuation in zip(tokenized_contexts, tokenized_continuations):
                    if doc.images:
                        # Append the continuation as text and let vLLM's HF processor tokenize the
                        # whole (image + text) prompt. The continuation stays at the very end, so the
                        # logprob slicing below (which counts back ``len(continuation)`` tokens) still
                        # lines up regardless of image-token expansion earlier in the prompt.
                        continuation_text = self.tokenizer.decode(tokenized_continuation)
                        inputs.append(self._multimodal_prompt(context + continuation_text, doc))
                    else:
                        inputs.append(tokenized_context + tokenized_continuation)
                    tokenized_continuations_batch.append(tokenized_continuation)
                    tokenized_contexts_batch.append(tokenized_context)

            outputs = self._generate(inputs, generate=False)

            flat_index = 0
            for i, doc in enumerate(split):
                outputs_doc = outputs[flat_index : flat_index + len(doc.choices)]
                tokenized_continuations_doc = tokenized_continuations_batch[flat_index : flat_index + len(doc.choices)]
                tokenized_contexts_doc = tokenized_contexts_batch[flat_index : flat_index + len(doc.choices)]
                logprobs_doc = []
                argmax_doc = []
                output_tokens_doc = []
                input_tokens_doc = []

                for output, context, continuation in zip(
                    outputs_doc, tokenized_contexts_doc, tokenized_continuations_doc
                ):
                    actual_input_len = len(output.prompt_token_ids)
                    continuation_len = len(continuation)
                    continuation_start_idx = actual_input_len - continuation_len
                    continuation_prompt_logprobs = output.prompt_logprobs[continuation_start_idx:]
                    # Use the *actual* prompt token ids at the continuation positions for the
                    # logprob lookup. In the multimodal path the continuation is appended as text
                    # and the whole prompt is re-tokenized, so boundary tokens may differ from our
                    # separately-tokenized ``continuation``. vLLM only includes the top-1 token and
                    # the actual prompt token in ``prompt_logprobs`` (prompt_logprobs=1), so looking
                    # up our continuation token directly can raise KeyError. The actual prompt token
                    # is always present. In the text-only path these tokens equal ``continuation``.
                    actual_continuation_tokens = output.prompt_token_ids[continuation_start_idx:]

                    continuation_logprobs = []
                    for token, logprobs_at_position in zip(actual_continuation_tokens, continuation_prompt_logprobs):
                        continuation_logprobs.append(logprobs_at_position[token])

                    bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                    continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]

                    continuation_logprobs = sum(continuation_logprobs)
                    logprobs_doc.append(continuation_logprobs)
                    argmax_doc.append(bool_score)
                    output_tokens_doc.append(continuation)
                    input_tokens_doc.append(context)

                answer = ModelResponse(
                    input=contexts[i],
                    input_tokens=input_tokens_doc,
                    output_tokens=output_tokens_doc,
                    logprobs=logprobs_doc,
                    argmax_logits_eq_gold=argmax_doc,
                )
                res.append(answer)
                flat_index += len(doc.choices)

        return dataset.get_original_order(res)

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        raise NotImplementedError()


@requires("vllm")
class AsyncVLLMModel(VLLMModel):
    """VLLM models which deploy async natively (no ray). Supports DP and PP/TP but not batch size > 1"""

    DATASET_SPLITS = 1
    is_async = True

    def cleanup(self):
        if self.model is not None:
            del self.model
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    def _create_auto_model(self, config: VLLMModelConfig):
        """Creates an instance of the async vllm model loaded from HF. Requires using the v1 of VLLM.

        Returns:
            AsyncLLM: The created async VLLM instance
        """
        if config.reasoning_config is not None or config.generation_parameters.thinking_token_budget is not None:
            raise ValueError("the async vLLM engine doesn't take a thinking budget; use the sync engine")
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "data_parallel_size": config.data_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
            "enforce_eager": True,
            "skip_mm_profiling": config.skip_mm_profiling,
        }
        if config.attention_backend is not None:
            self.model_args["attention_backend"] = config.attention_backend
        if config.mm_encoder_attn_backend is not None:
            self.model_args["mm_encoder_attn_backend"] = config.mm_encoder_attn_backend

        if config.data_parallel_size > 1:
            self._batch_size = "auto"

        model = AsyncLLM.from_engine_args(AsyncEngineArgs(**self.model_args))

        # If the max_length can't get extracted from the config, it will be inferred from the model
        if self._max_length is None:
            self._max_length = model.model_config.max_model_len

        return model

    async def _async_one_item(
        self,
        index: int,
        doc: Doc,
        generative: bool,
    ) -> Coroutine[None, list, str]:
        """Contains the actual logic of the generation."""
        sampling_params = SamplingParams(**self.config.generation_parameters.to_vllm_dict())

        base_prompt = (
            self.prompt_manager.prepare_prompt_multimodal(doc)
            if doc.images
            else self.prompt_manager.prepare_prompt(doc)
        )

        if not generative:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False
            prompt = base_prompt + doc.choice
            index_str = f"logprob_{index}"
        else:
            sampling_params.n = doc.num_samples
            if sampling_params.n > 1:
                # Todo clementine: investigate more
                logger.warning(
                    "Careful, there can be unexpected behavior when using sampling evals with the async vllm model"
                )
            sampling_params.max_tokens = self.config.generation_parameters.max_new_tokens or doc.generation_size
            sampling_params.stop = [] if self.use_chat_template else doc.stop_sequences
            sampling_params.logprobs = int(doc.use_logits)
            prompt = base_prompt
            index_str = f"generative_{index}"

        if doc.images:
            # Attach the images so vLLM's HF processor fuses them with the placeholder
            # tokens already present in ``base_prompt``.
            prompt = self._multimodal_prompt(prompt, doc)

        if generative and sampling_params.seed is not None:
            # the batch path's request: the prompt's token ids, or the multimodal prompt with its images
            request = (
                prompt
                if isinstance(prompt, dict)
                else self.tokenizer(prompt, add_special_tokens=self.add_special_tokens)["input_ids"]
            )
            sampling_params.seed = request_seeds(sampling_params.seed, [request])[0]

        generator = self.model.generate(request_id=index_str, prompt=prompt, sampling_params=sampling_params)
        try:
            while output := await anext(generator):
                continue
        except StopAsyncIteration:
            pass

        return output

    async def _async_batch(self, docs: list[Doc], generative: bool) -> list:
        processed_requests = [
            self._async_one_item(index=index, doc=doc, generative=generative) for index, doc in enumerate(docs)
        ]
        results = await asyncio.gather(*processed_requests)
        return results

    @cached(SamplingMethod.GENERATIVE)
    async def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        results = []

        responses = await self._async_batch(docs=docs, generative=True)

        for response in responses:
            output_token_ids = [outputs.token_ids for outputs in response.outputs]
            full_logprobs = [output.logprobs for output in response.outputs] or []
            logprobs = [logprob[token_id].logprob for token_id, logprob in zip(output_token_ids[0], full_logprobs[0])]
            result = [output.text for output in response.outputs]
            input_token_ids = response.prompt_token_ids

            cur_response = ModelResponse(
                text=result,
                logprobs=logprobs,
                output_tokens=list(output_token_ids),
                input_tokens=input_token_ids,
            )
            results.append(cur_response)

        return results

    @cached(SamplingMethod.LOGPROBS)
    async def loglikelihood(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met and
        stores the logprobs.

        Args:
            docs (list[Doc]): List of documents containing the context and choices.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        results = []

        responses = await self._async_batch(docs=docs, generative=False)

        for response, input in zip(responses, docs):
            continuation_logprobs = []
            for token, logprobs in zip(input.tokenized_continuation[::-1], response.prompt_logprobs[::-1]):
                continuation_logprobs.append(logprobs[token])
            bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
            continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
            answer = ModelResponse(
                input_tokens=input.tokenized_context + input.tokenized_continuation,
                output_tokens=input.tokenized_continuation,
                logprobs=sum(continuation_logprobs),
                argmax_logits_eq_gold=bool_score,
            )
            results.append(answer)

        return results
