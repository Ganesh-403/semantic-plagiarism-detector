# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
