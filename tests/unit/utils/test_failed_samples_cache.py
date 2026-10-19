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

"""A sample the backend failed to answer is returned as is but never cached, so the next run asks again."""

from lighteval.models.endpoints.vllm_openai_model import VLLMOpenAIModelConfig
from lighteval.models.model_output import ModelResponse, RequestFailure
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached


class _Model:
    def __init__(self, cache_dir, answers: dict[str, ModelResponse]):
        self._cache = SampleCache(
            VLLMOpenAIModelConfig(model_name="org/m", base_url="http://x/v1", cache_dir=str(cache_dir))
        )
        self.answers = answers
        self.asked: list[list[str]] = []

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        self.asked.append([doc.id for doc in docs])
        return [self.answers[doc.id] for doc in docs]


def _docs(count: int) -> list[Doc]:
    return [Doc(query=f"q{i}", choices=[], gold_index=0, task_name="t|0", id=str(i)) for i in range(count)]


def test_a_failed_sample_is_returned_but_asked_again_by_the_next_run(tmp_path):
    answers = {
        "0": ModelResponse(text=["a"]),
        "1": ModelResponse(text=[""], error=RequestFailure.RETRIES_EXHAUSTED),
        "2": ModelResponse(text=["c"]),
    }
    first = _Model(tmp_path, answers).greedy_until(_docs(3))
    assert [(r.text, r.error) for r in first] == [(["a"], None), ([""], "retries_exhausted"), (["c"], None)]

    rerun = _Model(tmp_path, {**answers, "1": ModelResponse(text=["b"])})
    assert [r.text for r in rerun.greedy_until(_docs(3))] == [["a"], ["b"], ["c"]]
    assert rerun.asked == [["1"]]


def test_a_task_whose_every_sample_failed_leaves_nothing_in_the_cache(tmp_path):
    model = _Model(tmp_path, {"0": ModelResponse(text=[""], error=RequestFailure.STALLED)})
    assert [r.error for r in model.greedy_until(_docs(1))] == ["stalled"]
    assert not list(tmp_path.rglob("*.parquet"))
