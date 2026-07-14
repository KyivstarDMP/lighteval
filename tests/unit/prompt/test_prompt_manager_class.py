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

from unittest.mock import Mock

import pytest

from lighteval.tasks.prompt_manager import ImagePlacement, PromptManager, order_multimodal_content
from lighteval.tasks.requests import Doc


class TestPromptManager:
    """Test suite for the PromptManager class."""

    def test_init_default_values(self):
        """Test PromptManager initialization with default values."""
        pm = PromptManager()
        assert pm.use_chat_template is False
        assert pm.tokenizer is None
        assert pm.system_prompt is None
        assert pm.chat_template_kwargs == {}

    def test_init_with_chat_template(self):
        """Test PromptManager initialization with chat template enabled."""
        tokenizer = Mock()
        system_prompt = "You are a helpful assistant."
        pm = PromptManager(
            use_chat_template=True,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert pm.use_chat_template is True
        assert pm.tokenizer == tokenizer
        assert pm.system_prompt == system_prompt
        assert pm.chat_template_kwargs == {"enable_thinking": False}

    def test_prepare_prompt_plain_text_basic(self):
        """Test prepare_prompt with plain text format and basic document."""
        pm = PromptManager()
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        result = pm.prepare_prompt(doc)
        assert result == "What is 2+2?"

    def test_prepare_prompt_plain_text_with_system_prompt(self):
        """Test prepare_prompt with plain text format and system prompt."""
        pm = PromptManager(system_prompt="You are a math tutor.")
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        result = pm.prepare_prompt(doc)
        assert result == "You are a math tutor.\n\nWhat is 2+2?"

    def test_prepare_prompt_plain_text_with_instruction(self):
        """Test prepare_prompt with plain text format and instruction."""
        pm = PromptManager()
        doc = Doc(
            query="What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction="Please answer the following question:",
        )

        result = pm.prepare_prompt(doc)
        assert result == "Please answer the following question:\n\nWhat is 2+2?"

    def test_prepare_prompt_plain_text_with_system_and_instruction(self):
        """Test prepare_prompt with plain text format, system prompt and instruction."""
        pm = PromptManager(system_prompt="You are a math tutor.")
        doc = Doc(
            query="What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction="Please answer the following question:",
        )

        result = pm.prepare_prompt(doc)
        assert result == "You are a math tutor.\n\nPlease answer the following question:\n\nWhat is 2+2?"

    def test_prepare_prompt_plain_text_with_fewshot(self):
        """Test prepare_prompt with plain text format and few-shot examples."""
        pm = PromptManager()

        # Create few-shot sample
        fewshot_doc = Doc(query="What is 1+1?", choices=["1", "2", "3"], gold_index=1)

        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)
        doc.fewshot_samples = [fewshot_doc]

        result = pm.prepare_prompt(doc)
        assert result == "What is 1+1? 2\n\nWhat is 2+2?"

    def test_prepare_prompt_plain_text_with_fewshot_and_instruction(self):
        """Test prepare_prompt with plain text format, few-shot examples and instruction."""
        pm = PromptManager()

        # Create few-shot sample with instruction
        fewshot_doc = Doc(
            query="Please answer the following question: What is 1+1?",
            choices=["1", "2", "3"],
            gold_index=1,
            instruction="Please answer the following question:",
        )

        doc = Doc(
            query="Please answer the following question: What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction="Please answer the following question:",
        )
        doc.fewshot_samples = [fewshot_doc]

        result = pm.prepare_prompt(doc)
        assert result == "Please answer the following question:\n\nWhat is 1+1? 2\n\nWhat is 2+2?"

    def test_prepare_prompt_chat_template_basic(self):
        """Test prepare_prompt with chat template format and basic document."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<|user|>\nWhat is 2+2?<|assistant|>"

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer)
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        result = pm.prepare_prompt(doc)
        assert result == "<|user|>\nWhat is 2+2?<|assistant|>"
        tokenizer.apply_chat_template.assert_called_once()

    def test_prepare_prompt_chat_template_passes_chat_template_kwargs(self):
        """Test prepare_prompt forwards chat_template_kwargs to apply_chat_template."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<|user|>\nWhat is 2+2?<|assistant|>"

        pm = PromptManager(
            use_chat_template=True,
            tokenizer=tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        result = pm.prepare_prompt(doc)
        assert result == "<|user|>\nWhat is 2+2?<|assistant|>"

        call_args = tokenizer.apply_chat_template.call_args
        assert call_args.kwargs["enable_thinking"] is False

    def test_prepare_prompt_chat_template_with_system_prompt(self):
        """Test prepare_prompt with chat template format and system prompt."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = (
            "<|system|>\nYou are a math tutor.<|user|>\nWhat is 2+2?<|assistant|>"
        )

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer, system_prompt="You are a math tutor.")
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        result = pm.prepare_prompt(doc)
        assert result == "<|system|>\nYou are a math tutor.<|user|>\nWhat is 2+2?<|assistant|>"

        # Verify the call arguments
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a math tutor."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    def test_prepare_prompt_chat_template_with_fewshot(self):
        """Test prepare_prompt with chat template format and few-shot examples."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = (
            "<|user|>\nWhat is 1+1?<|assistant|>\n2<|user|>\nWhat is 2+2?<|assistant|>"
        )

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer)

        # Create few-shot sample
        fewshot_doc = Doc(query="What is 1+1?", choices=["1", "2", "3"], gold_index=1)

        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)
        doc.fewshot_samples = [fewshot_doc]

        result = pm.prepare_prompt(doc)
        assert result == "<|user|>\nWhat is 1+1?<|assistant|>\n2<|user|>\nWhat is 2+2?<|assistant|>"

        # Verify the call arguments
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 1+1?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "2"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "What is 2+2?"

    def test_prepare_prompt_chat_template_with_fewshot_and_instruction(self):
        """Test prepare_prompt with chat template format, few-shot examples and instruction."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<|user|>\nPlease answer the following question: What is 1+1?<|assistant|>\n2<|user|>\nWhat is 2+2?<|assistant|>"

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer)

        # Create few-shot sample with instruction
        fewshot_doc = Doc(
            query="Please answer the following question: What is 1+1?",
            choices=["1", "2", "3"],
            gold_index=1,
            instruction="Please answer the following question: ",
        )

        doc = Doc(
            query="Please answer the following question: What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction="Please answer the following question: ",
        )
        doc.fewshot_samples = [fewshot_doc]

        result = pm.prepare_prompt(doc)
        assert (
            result
            == "<|user|>\nPlease answer the following question: What is 1+1?<|assistant|>\n2<|user|>\nWhat is 2+2?<|assistant|>"
        )

        # Verify the call arguments
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Please answer the following question: What is 1+1?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "2"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "What is 2+2?"

    def test_prepare_prompt_chat_template_with_fewshot_and_instruction_and_system_prompt(self):
        """Test prepare_prompt with chat template format, few-shot examples, instruction and system prompt."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<|system|>You are a math tutor.\nPlease answer the following question:<|user|>\nWhat is 1+1?<|assistant|>\n2<|user|>\nWhat is 2+2?<|assistant|>"

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer, system_prompt="You are a math tutor.")

        # Create few-shot sample with instruction
        fewshot_doc = Doc(
            query="Please answer the following question: What is 1+1?",
            choices=["1", "2", "3"],
            gold_index=1,
            instruction="Please answer the following question:",
        )

        doc = Doc(
            query="Please answer the following question: What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction="Please answer the following question:",
        )
        doc.fewshot_samples = [fewshot_doc]

        result = pm.prepare_prompt(doc)
        assert (
            result
            == "<|system|>You are a math tutor.\nPlease answer the following question:<|user|>\nWhat is 1+1?<|assistant|>\n2<|user|>\nWhat is 2+2?<|assistant|>"
        )

        # Verify the call arguments
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 4

    def test_prepare_prompt_chat_template_no_tokenizer(self):
        """Test prepare_prompt with chat template but no tokenizer raises error."""
        pm = PromptManager(use_chat_template=True, tokenizer=None)
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        with pytest.raises(AssertionError, match="Tokenizer must be set for chat template formatting."):
            pm.prepare_prompt(doc)

    def test_prepare_prompt_api_basic(self):
        """Test prepare_prompt_api with basic document."""
        pm = PromptManager()
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        result = pm.prepare_prompt_api(doc)
        assert result == [{"role": "user", "content": "What is 2+2?"}]

    def test_prepare_prompt_api_with_system_prompt(self):
        """Test prepare_prompt_api with system prompt."""
        pm = PromptManager(system_prompt="You are a math tutor.")
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)

        result = pm.prepare_prompt_api(doc)
        assert result == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
        ]

    def test_prepare_prompt_api_with_instruction(self):
        """Test prepare_prompt_api with instruction."""
        pm = PromptManager()
        doc = Doc(
            query="What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction="Please answer the following question: ",
        )

        result = pm.prepare_prompt_api(doc)

        assert result == [
            {"role": "user", "content": "Please answer the following question: What is 2+2?"},
        ]

    def test_prepare_prompt_api_with_system_and_instruction(self):
        """Test prepare_prompt_api with system prompt and instruction."""
        pm = PromptManager(system_prompt="You are a math tutor.")
        doc = Doc(
            query="What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction="Please answer the following question: ",
        )

        result = pm.prepare_prompt_api(doc)
        assert result == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Please answer the following question: What is 2+2?"},
        ]

    def test_prepare_prompt_multimodal_basic(self):
        """Test prepare_prompt_multimodal with basic multimodal document."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<|user|>\n<image>What is in this image?<|assistant|>"

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer)

        # Mock image
        mock_image = Mock()

        doc = Doc(
            query="What is in this image?", choices=["A cat", "A dog", "A bird"], gold_index=0, images=[mock_image]
        )

        result = pm.prepare_prompt_multimodal(doc)
        assert result == "<|user|>\n<image>What is in this image?<|assistant|>"

        # Verify the call arguments
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "What is in this image?"
        assert messages[0]["content"][1]["type"] == "image"
        assert messages[0]["content"][1]["image"] == mock_image

    def test_prepare_prompt_multimodal_passes_chat_template_kwargs(self):
        """Test prepare_prompt_multimodal forwards chat_template_kwargs to apply_chat_template."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<|user|>\n<image>What is in this image?<|assistant|>"

        pm = PromptManager(
            use_chat_template=True,
            tokenizer=tokenizer,
            chat_template_kwargs={"enable_thinking": False},
        )

        mock_image = Mock()
        doc = Doc(
            query="What is in this image?", choices=["A cat", "A dog", "A bird"], gold_index=0, images=[mock_image]
        )

        result = pm.prepare_prompt_multimodal(doc)
        assert result == "<|user|>\n<image>What is in this image?<|assistant|>"

        call_args = tokenizer.apply_chat_template.call_args
        assert call_args.kwargs["enable_thinking"] is False

    def test_prepare_prompt_multimodal_with_system_prompt(self):
        """Test prepare_prompt_multimodal with system prompt."""
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = (
            "<|system|>\nYou are a helpful assistant.<|user|>\n<image>What is in this image?<|assistant|>"
        )

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer, system_prompt="You are a helpful assistant.")

        mock_image = Mock()
        doc = Doc(
            query="What is in this image?", choices=["A cat", "A dog", "A bird"], gold_index=0, images=[mock_image]
        )

        result = pm.prepare_prompt_multimodal(doc)
        assert result == "<|system|>\nYou are a helpful assistant.<|user|>\n<image>What is in this image?<|assistant|>"

        # Verify the call arguments
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) == 2

    def test_prepare_prompt_multimodal_no_chat_template(self):
        """Test prepare_prompt_multimodal without chat template raises error."""
        pm = PromptManager(use_chat_template=False)
        mock_image = Mock()
        doc = Doc(
            query="What is in this image?", choices=["A cat", "A dog", "A bird"], gold_index=0, images=[mock_image]
        )

        with pytest.raises(ValueError, match="Multimodal prompts are only supported with chat template format."):
            pm.prepare_prompt_multimodal(doc)

    def test_prepare_prompt_multimodal_no_tokenizer(self):
        """Test prepare_prompt_multimodal without tokenizer raises error."""
        pm = PromptManager(use_chat_template=True, tokenizer=None)
        mock_image = Mock()
        doc = Doc(
            query="What is in this image?", choices=["A cat", "A dog", "A bird"], gold_index=0, images=[mock_image]
        )

        with pytest.raises(ValueError, match="Multimodal prompts are only supported with chat template format."):
            pm.prepare_prompt_multimodal(doc)

    def test_prepare_prompt_multimodal_no_images(self):
        """Test prepare_prompt_multimodal without images raises error."""
        tokenizer = Mock()
        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer)
        doc = Doc(query="What is in this image?", choices=["A cat", "A dog", "A bird"], gold_index=0, images=None)

        with pytest.raises(ValueError, match="Multimodal prompts require images to be provided in the document."):
            pm.prepare_prompt_multimodal(doc)

    def test_extract_query_no_instruction(self):
        """Test _extract_query with no instruction."""
        pm = PromptManager()
        result = pm._extract_query("What is 2+2?", None)
        assert result == "What is 2+2?"

    def test_extract_query_with_instruction(self):
        """Test _extract_query with instruction but no previous shots."""
        pm = PromptManager()
        result = pm._extract_query("Please answer: What is 2+2?", "Please answer: ")
        assert result == "What is 2+2?"

    def test_prepare_prompt_with_multiple_fewshot_examples(self):
        """Test prepare_prompt with multiple few-shot examples."""
        pm = PromptManager()

        # Create few-shot samples
        fewshot_doc1 = Doc(query="What is 1+1?", choices=["1", "2", "3"], gold_index=1)
        fewshot_doc2 = Doc(query="What is 3+3?", choices=["5", "6", "7"], gold_index=1)

        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)
        doc.fewshot_samples = [fewshot_doc1, fewshot_doc2]

        result = pm.prepare_prompt(doc)
        assert result == "What is 1+1? 2\n\nWhat is 3+3? 6\n\nWhat is 2+2?"

    def test_prepare_prompt_with_empty_fewshot_samples(self):
        """Test prepare_prompt with empty few-shot samples."""
        pm = PromptManager()
        doc = Doc(query="What is 2+2?", choices=["3", "4", "5"], gold_index=1)
        doc.fewshot_samples = []

        result = pm.prepare_prompt(doc)
        assert result == "What is 2+2?"

    def test_prepare_prompt_with_complex_instruction_removal(self):
        """Test prepare_prompt with complex instruction removal in few-shot examples."""
        pm = PromptManager()

        # Create few-shot sample with complex instruction
        instruction = "Please provide a detailed mathematical answer to the following question:"
        fewshot_doc = Doc(query="What is 1+1?", choices=["1", "2", "3"], gold_index=1, instruction=instruction)

        doc = Doc(
            query="What is 2+2?",
            choices=["3", "4", "5"],
            gold_index=1,
            instruction=instruction,
        )
        doc.fewshot_samples = [fewshot_doc]

        result = pm.prepare_prompt(doc)
        expected = f"{instruction}\n\nWhat is 1+1? 2\n\nWhat is 2+2?"
        assert result == expected


class TestOrderMultimodalContentPlacement:
    """Test suite for ``order_multimodal_content``'s ``placement`` modes."""

    def test_inline_binds_marker_positionally(self):
        image_parts = [{"type": "image", "image": "IMG"}]
        content = order_multimodal_content("before <image> after", image_parts)
        assert [c["type"] for c in content] == ["text", "image", "text"]
        assert content[0]["text"] == "before "
        assert content[2]["text"] == " after"

    def test_inline_with_no_marker_prepends(self):
        image_parts = [{"type": "image", "image": "IMG"}]
        content = order_multimodal_content("no marker here", image_parts)
        assert [c["type"] for c in content] == ["image", "text"]

    def test_prepend_attaches_all_images_before_unmodified_text(self):
        image_parts = [{"type": "image", "image": "IMG1"}, {"type": "image", "image": "IMG2"}]
        query = "<image 1> Compare with <image 2>."
        content = order_multimodal_content(query, image_parts, placement=ImagePlacement.prepend)

        assert content[0] == image_parts[0]
        assert content[1] == image_parts[1]
        assert content[2] == {"type": "text", "text": query}

    def test_prepend_accepts_plain_string(self):
        image_parts = [{"type": "image", "image": "IMG"}]
        content = order_multimodal_content("<image 1> what is this?", image_parts, placement="prepend")
        assert content[0]["type"] == "image"

    def test_prepend_keeps_markers_verbatim_as_readable_text(self):
        image_parts = [{"type": "image", "image": "IMG"}]
        query = "<image 1> what is this?"
        content = order_multimodal_content(query, image_parts, placement=ImagePlacement.prepend)
        assert "<image 1>" in content[-1]["text"]

    def test_rejects_invalid_placement(self):
        with pytest.raises(ValueError, match="image_placement must be one of"):
            order_multimodal_content("<image>", [{"type": "image", "image": "IMG"}], placement="sideways")


class TestPromptManagerImagePlacement:
    """Test suite for ``PromptManager``'s ``image_placement`` constructor argument."""

    def test_default_image_placement_is_inline(self):
        pm = PromptManager()
        assert pm.image_placement is ImagePlacement.inline

    def test_accepts_enum_member(self):
        pm = PromptManager(image_placement=ImagePlacement.prepend)
        assert pm.image_placement is ImagePlacement.prepend

    def test_accepts_plain_string(self):
        pm = PromptManager(image_placement="prepend")
        assert pm.image_placement is ImagePlacement.prepend

    def test_does_not_touch_chat_template_kwargs(self):
        pm = PromptManager(chat_template_kwargs={"enable_thinking": False}, image_placement="prepend")
        assert pm.chat_template_kwargs == {"enable_thinking": False}

    def test_prepare_prompt_multimodal_does_not_leak_image_placement_into_chat_template_kwargs(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **_kwargs: messages

        pm = PromptManager(use_chat_template=True, tokenizer=tokenizer, image_placement=ImagePlacement.prepend)
        doc = Doc(query="<image 1> text", choices=["A"], gold_index=0, images=[Mock()])

        content = pm.prepare_prompt_multimodal(doc)[0]["content"]
        assert content[0]["type"] == "image"  # placement was actually threaded through

        call_kwargs = tokenizer.apply_chat_template.call_args.kwargs
        assert "image_placement" not in call_kwargs
