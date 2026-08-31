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
