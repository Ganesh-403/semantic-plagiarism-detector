"""Tests for src/core/embeddings.py"""

from src.core.embeddings import generate_embeddings


def test_embed_empty_text_returns_zero_vector():
    vec = generate_embeddings([""])
    assert len(vec) == 0 or all(
        v == 0.0 for v in vec[0]
    ), "empty input should return empty or zero vector"
