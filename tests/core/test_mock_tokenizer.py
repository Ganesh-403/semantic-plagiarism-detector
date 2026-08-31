
import pytest
from tests.conftest import mock_fast_tokenizer
import torch

def test_mock_fast_tokenizer_returns_fixed_tensors(mock_fast_tokenizer):
    """Verify that the mock tokenizer produces the expected tensor formats."""
    output = mock_fast_tokenizer("Hello world")
    assert "input_ids" in output
    assert "attention_mask" in output
    
    assert isinstance(output["input_ids"], torch.Tensor)
    assert output["input_ids"].shape == (1, 16)
    assert output["attention_mask"].shape == (1, 16)

def test_mock_fast_tokenizer_deterministic(mock_fast_tokenizer):
    """Verify that the mock tokenizer produces consistent results for the same length."""
    output1 = mock_fast_tokenizer("Hello world")
    output2 = mock_fast_tokenizer("Hello earth")
    assert torch.equal(output1["input_ids"], output2["input_ids"])

def test_mock_fast_tokenizer_batching(mock_fast_tokenizer):
    """Verify batch processing behavior."""
    output = mock_fast_tokenizer(["one", "two", "three"])
    assert output["input_ids"].shape == (3, 16)
    assert output["attention_mask"].shape == (3, 16)

def test_mock_fast_tokenizer_padding_and_eos_tokens(mock_fast_tokenizer):
    """Verify properties expected by HuggingFace models."""
    assert mock_fast_tokenizer.pad_token_id == 0
    assert mock_fast_tokenizer.eos_token_id == 2
    assert mock_fast_tokenizer.model_max_length == 512
