"""
test_ai_detector.py
-------------------
Tests for AI-generated text detection functionality.
"""


from unittest.mock import MagicMock, patch
import pytest

from src.core.ai_detector import (
    detect_ai_probability,
    detect_ai_probability_batch,
    detect_document_ai_probability,
    detect_documents_ai_probability,
)

from src.core.ai_detector import (detect_ai_probability,
                                  detect_ai_probability_batch,
                                  detect_document_ai_probability,
                                  detect_documents_ai_probability)



@pytest.fixture(autouse=True)
def mock_transformers_pipeline():
    """Autouse fixture to mock Hugging Face pipeline across all tests in this module."""
    with patch("transformers.pipeline") as mock_pipe:
        mock_classifier = MagicMock()
        # Mock pipeline output format: [{'label': 'Fake', 'score': 0.85}]
        mock_classifier.return_value = [[{"label": "Fake", "score": 0.85}]]
        mock_pipe.return_value = mock_classifier
        yield mock_pipe


def test_detect_ai_probability_empty_text():
    """Test that empty text returns 0.0 probability."""
    result = detect_ai_probability("")
    assert result == 0.0


def test_detect_ai_probability_none():
    """Test that None input returns 0.0 probability."""
    result = detect_ai_probability(None)
    assert result == 0.0


def test_detect_ai_probability_batch_empty():
    """Test that empty list returns empty list."""
    result = detect_ai_probability_batch([])
    assert result == []


def test_detect_document_ai_probability_empty():
    """Test that empty chunks return zero probabilities."""
    result = detect_document_ai_probability([])
    assert result["overall"] == 0.0
    assert result["max"] == 0.0
    assert result["chunk_scores"] == []


def test_detect_documents_ai_probability_empty():
    """Test that empty dict returns empty dict."""
    result = detect_documents_ai_probability({})
    assert result == {}


def test_detect_documents_ai_probability_single_doc():
    """Test AI detection with a single document."""
    chunked_docs = {
        "test_doc.txt": ["This is a test chunk of text.", "Another test chunk here."]
    }
    result = detect_documents_ai_probability(chunked_docs)

    assert "test_doc.txt" in result
    assert "overall" in result["test_doc.txt"]
    assert "max" in result["test_doc.txt"]
    assert "chunk_scores" in result["test_doc.txt"]
    assert len(result["test_doc.txt"]["chunk_scores"]) == 2
    assert 0.0 <= result["test_doc.txt"]["overall"] <= 1.0
    assert 0.0 <= result["test_doc.txt"]["max"] <= 1.0


def test_detect_ai_probability_batch_mixed():
    """Test batch detection with mixed empty and non-empty texts."""
    texts = ["Some text", "", None, "More text"]
    result = detect_ai_probability_batch(texts)

    assert len(result) == 4
    assert result[1] == 0.0  # Empty string
    assert result[2] == 0.0  # None
    assert 0.0 <= result[0] <= 1.0
    assert 0.0 <= result[3] <= 1.0
    