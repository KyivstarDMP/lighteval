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

"""A loglikelihood choice the server rejected is scored -inf: it must survive the sample cache and count as wrong."""

import tempfile

import pytest
from datasets import Dataset, load_dataset

from lighteval.metrics.metrics_sample import LoglikelihoodAcc
from lighteval.models.endpoints.vllm_openai_model import VLLMOpenAIModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.cache_management import SampleCache


INF = float("-inf")


def test_an_unscorable_choice_keeps_its_minus_inf_through_the_cache():
    # 2026-09-30: the reload wrote -inf as null, and np.argmax over [None, ...] crashed a 2 h MMMU-Pro group
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = SampleCache(VLLMOpenAIModelConfig(model_name="org/m", base_url="http://x/v1", cache_dir=temp_dir))
        path = f"{temp_dir}/samples.parquet"
        response = ModelResponse(logprobs=[INF, -1.5], argmax_logits_eq_gold=[False, True])
        Dataset.from_list([{"sample_id": "0", "sample": cache._dump_sample(response)}]).to_parquet(path)
        row = load_dataset("parquet", data_files=path, split="train").to_pandas().set_index("sample_id").loc["0"]
        assert cache._load_sample(row).logprobs == [INF, -1.5]


@pytest.mark.parametrize(
    "logprobs, gold, expected",
    [([INF, INF, INF], 0, 0), ([INF, -2.0, -1.0], 2, 1), ([-1.0, INF, INF], 1, 0)],
    ids=["none-scorable", "a-scorable-gold-wins", "an-unscorable-gold-loses"],
)
def test_loglikelihood_accuracy_counts_a_doc_without_a_scorable_choice_as_wrong(logprobs, gold, expected):
    doc = Doc(query="q", choices=[" A", " B", " C"], gold_index=gold, task_name="t")
    assert LoglikelihoodAcc().compute(doc=doc, model_response=ModelResponse(logprobs=logprobs)) == expected
