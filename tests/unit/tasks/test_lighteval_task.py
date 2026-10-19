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


import json

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc


def dummy_prompt_function(item, task_name):
    return Doc(query=item["text"], choices=["A", "B"], gold_index=0, task_name=task_name)


def test_revision_check():
    # Test with a different revision
    cfg_with_revision = LightevalTaskConfig(
        name="test_task_revision",
        prompt_function=dummy_prompt_function,
        hf_repo="lighteval-tests-datasets/dataset-test-1",
        hf_subset="default",
        evaluation_splits=["train"],
        metrics=[],
        hf_revision="25175defadfde48b131b7cd7573ad6f59f868306",
    )
    task_with_revision = LightevalTask(cfg_with_revision)
    docs = task_with_revision.eval_docs()
    queries = [doc.query for doc in docs]
    assert queries == ["hi", "how are you?"]


def test_dataset_filter():
    # Setup

    cfg = LightevalTaskConfig(
        name="test_task",
        prompt_function=dummy_prompt_function,
        hf_repo="lighteval-tests-datasets/dataset-test-1",
        hf_subset="default",
        hf_filter=lambda x: x["text"] == "hi",
        metrics=[],
        evaluation_splits=["train"],
    )
    task = LightevalTask(cfg)

    filtered_docs = task.eval_docs()
    assert len(filtered_docs) == 1
    assert filtered_docs[0].query == "hi"


def test_hf_data_files(tmp_path):
    # create a small jsonl dataset
    data_file = tmp_path / "data.jsonl"
    src_docs = [f"document {i}" for i in range(3)]
    data_file.write_text("\n".join([f'{{"text": "{doc}"}}' for doc in src_docs]))

    cfg = LightevalTaskConfig(
        name="test_data_files",
        prompt_function=dummy_prompt_function,
        hf_repo="json",
        hf_subset="default",
        metrics=[],
        evaluation_splits=["train"],
        hf_data_files=str(data_file),
    )
    task = LightevalTask(cfg)

    eval_docs = task.eval_docs()
    assert [doc.query for doc in eval_docs] == src_docs


def conversation_prompt_function(item, task_name):
    """Final user turn as the query; earlier turns carried on the doc as few-shot samples."""
    history = [
        Doc(query=turn["user"], choices=[turn["assistant"]], gold_index=0, task_name=task_name)
        for turn in item["history"]
    ]
    return Doc(query=item["text"], choices=[""], gold_index=0, task_name=task_name, fewshot_samples=history)


def _conversation_task(tmp_path, num_fewshots):
    data_file = tmp_path / "conversations.jsonl"
    rows = [
        {"text": "And the capital?", "history": [{"user": "Name a country.", "assistant": "France."}]},
        {"text": "Hello", "history": []},
    ]
    data_file.write_text("\n".join(json.dumps(row) for row in rows))
    cfg = LightevalTaskConfig(
        name="test_conversation",
        prompt_function=conversation_prompt_function,
        hf_repo="json",
        hf_subset="default",
        metrics=[],
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_select="sequential",
        hf_data_files=str(data_file),
    )
    cfg.num_fewshots = num_fewshots
    return LightevalTask(cfg)


def test_doc_fewshot_samples_kept_at_zero_shots(tmp_path):
    docs = {doc.query: doc for doc in _conversation_task(tmp_path, num_fewshots=0).get_docs()}

    assert [(s.query, s.get_golds()[0]) for s in docs["And the capital?"].fewshot_samples] == [
        ("Name a country.", "France.")
    ]
    assert docs["Hello"].fewshot_samples == []
    messages = PromptManager(use_chat_template=True, tokenizer=None).prepare_prompt_api(docs["And the capital?"])
    assert messages == [
        {"role": "user", "content": "Name a country."},
        {"role": "assistant", "content": "France."},
        {"role": "user", "content": "And the capital?"},
    ]


def test_sampled_fewshots_added_before_doc_turns(tmp_path):
    # With shots requested, sampled examples come first, then the doc's own turns.
    docs = {doc.query: doc for doc in _conversation_task(tmp_path, num_fewshots=1).get_docs()}

    convo = docs["And the capital?"]
    assert len(convo.fewshot_samples) == 2
    assert convo.fewshot_samples[-1].query == "Name a country."
    messages = PromptManager(use_chat_template=True, tokenizer=None).prepare_prompt_api(convo)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[2:] == [
        {"role": "user", "content": "Name a country."},
        {"role": "assistant", "content": "France."},
        {"role": "user", "content": "And the capital?"},
    ]
    # A doc without its own turns gets only the sampled example, as before.
    assert len(docs["Hello"].fewshot_samples) == 1
