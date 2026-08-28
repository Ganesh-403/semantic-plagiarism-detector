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

"""
test_mock_faiss_index_fixture_issue_3248.py
---------------------------------------------
Unit test suite for Issue #3248:
Validates that the mock_faiss_index pytest fixture in conftest.py
supports adding synthetic vectors and querying nearest neighbors.
"""

import numpy as np


def test_mock_faiss_index_add_and_search_vectors(mock_faiss_index):
    """Verify mock_faiss_index fixture adds synthetic vectors and queries nearest neighbors."""
    dimension = 384
    # Create 10 synthetic vectors
    np.random.seed(42)
    vectors = np.random.randn(10, dimension).astype(np.float32)

    # Add synthetic vectors to index
    mock_faiss_index.add_vectors(vectors)

    # Query nearest neighbors for the first vector
    query_vector = vectors[0:1]
    distances, indices = mock_faiss_index.search_vectors(query_vector, k=3)

    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    # The nearest neighbor to vector[0] should be index 0 with distance ~0
    assert indices[0][0] == 0
    assert np.isclose(distances[0][0], 0.0, atol=1e-4)


def test_mock_faiss_index_get_nearest_neighbors(mock_faiss_index):
    """Verify get_nearest_neighbors helper function works as intended."""
    dimension = 384
    vectors = np.ones((5, dimension), dtype=np.float32)
    mock_faiss_index.add_vectors(vectors)

    distances, indices = mock_faiss_index.get_nearest_neighbors(vectors[0], k=2)
    assert distances.shape == (1, 2)
    assert indices.shape == (1, 2)
