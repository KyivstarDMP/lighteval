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

import unittest

from lighteval.utils.utils import remove_reasoning_tags


class TestRemoveReasoningTags(unittest.TestCase):
    def test_remove_reasoning_tags(self):
        text = "<think> Reasoning section </think> Answer section"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, " Answer section")

    def test_remove_multiple_tags(self):
        text = "<think> Reasoning </think> Interlude <think> More reasoning </think> Answer"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, " Interlude  Answer")

    def test_no_tags(self):
        text = "No reasoning tags here."
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "No reasoning tags here.")

    def test_empty_text(self):
        text = ""
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "")

    def test_no_opening_tag(self):
        text = "No opening tag <think> Reasoning section. </think> Answer section"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "No opening tag  Answer section")

    def test_no_closing_tag(self):
        text = "<think> Reasoning section. Answer section"
        tag_pairs = [("<think>", "</think>")]
        result = remove_reasoning_tags(text, tag_pairs)
        self.assertEqual(result, "")  # an unterminated thought (budget exhausted) is all reasoning

    def test_unclosed_span_after_a_closed_one(self):
        self.assertEqual(remove_reasoning_tags("<think>a</think>Answer<think>b", [("<think>", "</think>")]), "Answer")

    def test_empty_start_tag_keeps_the_answer_after_its_closer(self):
        for thinking in (True, False, None):
            with self.subTest(thinking=thinking):
                self.assertEqual(
                    remove_reasoning_tags("Reasoning</think>Answer", [("", "</think>")], thinking=thinking), "Answer"
                )

    def test_a_later_pair_with_the_same_start_tag_closes_the_span(self):
        # gpt-oss: the full final-channel closer first, the shorter one as a fallback
        pairs = [("<|channel|>analysis", "<|channel|>final<|message|>"), ("<|channel|>analysis", "<|channel|>final")]
        for thinking in (True, None):
            with self.subTest(thinking=thinking):
                self.assertEqual(
                    remove_reasoning_tags(
                        "<|channel|>analysisSix times seven.<|channel|>final#### 42", pairs, thinking
                    ),
                    "#### 42",
                )
                self.assertEqual(
                    remove_reasoning_tags("<|channel|>analysisX<|channel|>final<|message|>42", pairs, thinking), "42"
                )
                self.assertEqual(remove_reasoning_tags("<|channel|>analysisSix times seven, so", pairs, thinking), "")

    def test_an_answer_after_a_closed_thought_survives_in_every_mode(self):
        gpt_oss_pairs = [
            ("<|channel|>analysis", "<|channel|>final<|message|>"),
            ("<|channel|>analysis", "<|channel|>final"),
        ]
        for thinking in (True, False, None):
            with self.subTest(thinking=thinking):
                self.assertEqual(
                    remove_reasoning_tags(
                        "<|channel|>analysisSix times seven.<|channel|>final#### 42", gpt_oss_pairs, thinking
                    ),
                    "#### 42",
                )
                self.assertEqual(
                    remove_reasoning_tags("Six times seven.</think>#### 42", [("", "</think>")], thinking), "#### 42"
                )
                self.assertEqual(
                    remove_reasoning_tags(
                        "<think>Six times seven.</think>#### 42", [("<think>", "</think>")], thinking
                    ),
                    "#### 42",
                )


class TestRemoveReasoningTagsByMode(unittest.TestCase):
    """A span that opens and never closes: all thought with thinking on, the answer with thinking off."""

    def test_an_explicit_start_tag_without_a_closer(self):
        pairs = [("<think>", "</think>")]
        text = "Answer<think>Six times seven, so"
        self.assertEqual(remove_reasoning_tags(text, pairs, thinking=True), "Answer")
        self.assertEqual(remove_reasoning_tags(text, pairs, thinking=False), text)
        self.assertEqual(remove_reasoning_tags(text, pairs), "Answer")  # mode unknown: cut, as before

    def test_an_empty_start_tag_without_a_closer(self):
        # the chat template opened the span in the prompt, and the output never closed it (budget exhausted)
        pairs = [("", "</think>")]
        self.assertEqual(remove_reasoning_tags("Six times seven, so", pairs, thinking=True), "")
        self.assertEqual(remove_reasoning_tags("No closing tag", pairs, thinking=False), "No closing tag")
        self.assertEqual(remove_reasoning_tags("No closing tag", pairs), "No closing tag")  # mode unknown: kept

    def test_unclosed_after_closed_without_thinking_keeps_the_rest(self):
        pairs = [("<think>", "</think>")]
        self.assertEqual(
            remove_reasoning_tags("<think>a</think>Answer<think>b", pairs, thinking=False), "Answer<think>b"
        )

    def test_an_empty_start_tag_keeps_the_answer_when_another_pair_took_its_closer(self):
        pairs = [("<think>", "</think>"), ("", "</think>")]
        self.assertEqual(remove_reasoning_tags("<think>x</think>42", pairs, thinking=True), "42")
        self.assertEqual(remove_reasoning_tags("<think>x", pairs, thinking=True), "")  # opened and never closed

    def test_several_empty_start_tags_need_only_one_closer(self):
        pairs = [("", "</think>"), ("", "</reasoning>")]
        self.assertEqual(remove_reasoning_tags("Reasoning</think>42", pairs, thinking=True), "42")
        self.assertEqual(remove_reasoning_tags("still thinking", pairs, thinking=True), "")  # never left the span

    def test_an_answer_the_server_already_split_is_not_emptied(self):
        # A server reasoning parser returns only the answer, and the client passes no tags for it.
        self.assertEqual(remove_reasoning_tags("#### 42", [], thinking=True), "#### 42")
        # The default pair finds nothing to cut in it either.
        self.assertEqual(remove_reasoning_tags("#### 42", [("<think>", "</think>")], thinking=True), "#### 42")
        # Empty-start-tag pairs would empty it: the closer never reaches the answer. Pass none with a server parser.
        self.assertEqual(remove_reasoning_tags("#### 42", [("", "</think>")], thinking=True), "")
