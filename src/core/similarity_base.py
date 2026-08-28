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

"""similarity_base.py
------------------
Defines the abstract base similarity computation engine using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np


class BaseSimilarityEngine(ABC):
    """
    Abstract Base class for Similarity Computation Engines.
    Provides standard interface for pairwise similarity and matrix computation.
    """

    @abstractmethod
    def compute_pairwise_similarity(
        self,
        doc1: str | np.ndarray,
        doc2: str | np.ndarray,
    ) -> float:
        """
        Compute similarity score between two documents.

        Args:
            doc1: First document (either text string or embedding vector).
            doc2: Second document (either text string or embedding vector).

        Returns:
            A similarity score strictly bounded in [0.0, 1.0].
        """
        pass

    @abstractmethod
    def compute_matrix(
        self,
        documents: dict[str, str | np.ndarray] | list[str | np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity matrix across a collection of documents.

        Args:
            documents: A dictionary mapping doc names to texts/embeddings,
                       or a list/array of texts/embeddings.

        Returns:
            An N x N symmetric numpy array containing similarity scores in range [0.0, 1.0].
        """
        pass
