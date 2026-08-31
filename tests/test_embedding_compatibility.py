"""Tests for embedding model compatibility."""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.core.embedding_compatibility import (
    EmbeddingMetadata,
    ensure_compatible_metadata,
    validate_embedding_vector,
)


def _metadata(
    model: str = "model-a",
    version: str = "1",
    dimension: int = 384,
) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        model_identifier=model,
        model_version=version,
        dimension=dimension,
        normalization_strategy="l2",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def test_compatible_models_are_accepted():
    metadata = _metadata()

    ensure_compatible_metadata(metadata, metadata)


def test_model_change_is_rejected():
    expected = _metadata(model="model-a")
    actual = _metadata(model="model-b")

    with pytest.raises(ValueError, match="Incompatible embedding metadata"):
        ensure_compatible_metadata(expected, actual)


def test_model_version_change_is_rejected():
    expected = _metadata(version="1")
    actual = _metadata(version="2")

    with pytest.raises(ValueError, match="Incompatible embedding metadata"):
        ensure_compatible_metadata(expected, actual)


def test_dimension_change_is_rejected():
    expected = _metadata(dimension=384)
    actual = _metadata(dimension=768)

    with pytest.raises(ValueError, match="Incompatible embedding metadata"):
        ensure_compatible_metadata(expected, actual)


def test_vector_dimension_mismatch_is_rejected():
    metadata = _metadata(dimension=384)
    vector = np.zeros(768, dtype=np.float32)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        validate_embedding_vector(vector, metadata)


def test_normalized_vector_with_matching_dimension_is_accepted():
    metadata = _metadata(dimension=384)
    vector = np.zeros(384, dtype=np.float32)

    validate_embedding_vector(vector, metadata)