import numpy as np

from tests.conftest import MockDataFactory, dummy_embeddings


def test_mock_data_factory_embed_chunks_empty():
    """Ensure embed_chunks returns an empty array when given no chunks."""
    result = MockDataFactory.embed_chunks([])
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_mock_data_factory_embed_chunks_basic():
    """Ensure embed_chunks produces correct dimensionality and values for normal input."""
    chunks = ["first chunk", "second chunk", "third chunk"]
    result = MockDataFactory.embed_chunks(chunks, batch_size=2)

    # Check dimensions
    assert result.shape == (3, 384)

    # Check data type
    assert result.dtype == np.float32

    # Check value integrity
    expected_val = 1.0 / (384**0.5)
    assert np.allclose(result, expected_val)


def test_mock_data_factory_embed_chunks_large():
    """Ensure embed_chunks handles large batches correctly."""
    chunks = [f"chunk {i}" for i in range(100)]
    result = MockDataFactory.embed_chunks(chunks, batch_size=10)

    assert result.shape == (100, 384)
    expected_val = 1.0 / (384**0.5)
    assert np.allclose(result, expected_val)


def test_dummy_embeddings_structure():
    """Validate the consolidated dummy embeddings structure."""
    embeddings = dummy_embeddings()
    assert isinstance(embeddings, dict)
    assert "doc_A" in embeddings
    assert "doc_B" in embeddings
    assert "doc_C" in embeddings

    # Ensure they have standard shapes
    assert embeddings["doc_A"].shape == (2, 3)
    assert embeddings["doc_B"].shape == (2, 3)
    assert embeddings["doc_C"].shape == (1, 3)


def test_mock_factory_fixture(mock_factory):
    """Ensure the mock_factory pytest fixture yields a valid factory instance."""
    assert isinstance(mock_factory, MockDataFactory)
    assert hasattr(mock_factory, "embed_chunks")


def test_mock_embed_chunks_fixture(mock_embed_chunks):
    """Ensure the mock_embed_chunks fixture points directly to the correct static method."""
    assert callable(mock_embed_chunks)
    result = mock_embed_chunks(["hello"])
    assert result.shape == (1, 384)
