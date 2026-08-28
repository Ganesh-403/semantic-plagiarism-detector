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

"""similarity_engines.py
----------------------
Implements concrete similarity engines and the SimilarityEngineFactory.
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.core.similarity import document_similarity_matrix
from src.core.similarity_base import BaseSimilarityEngine


class SemanticSimilarityEngine(BaseSimilarityEngine):
    """
    Concrete Strategy for Semantic Similarity computation.
    Uses SentenceTransformer embeddings and optionally FAISS.
    """

    def __init__(
        self,
        embedding_model: Any = None,
        faiss_index: Any = None,
        registry: Any = None,
        batch_size: Optional[int] = None,
        min_threshold: float = 0.0,
        min_percentile: Optional[float] = None,
        pooling: str = "mean",
    ):
        """
        Initialize the Semantic Similarity Engine.

        Args:
            embedding_model: Instance of an embedding model that responds to .embed() or similar.
            faiss_index: Optional pre-built FAISS index.
            registry: Optional chunk registry for FAISS.
            batch_size: Batch size for matrix computation.
            min_threshold: Minimum similarity score to keep.
            min_percentile: Optional percentile threshold for filtering.
            pooling: Pooling strategy for multi-chunk document embeddings ('mean' or 'max').
        """
        if not isinstance(pooling, str) or pooling.lower() not in ("mean", "max"):
            raise ValueError(
                f"Invalid pooling method '{pooling}'. Supported methods: 'mean', 'max'."
            )
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.registry = registry
        self.batch_size = batch_size
        self.min_threshold = min_threshold
        self.min_percentile = min_percentile
        self.pooling = pooling

    def _get_embedding(self, doc: str | np.ndarray) -> np.ndarray:
        """Helper to resolve a document to its embedding representation."""
        if isinstance(doc, np.ndarray):
            return doc

        if isinstance(doc, str):
            if self.embedding_model is not None:
                # Call embedding method if model exists
                if hasattr(self.embedding_model, "embed"):
                    return self.embedding_model.embed(doc)
                elif hasattr(self.embedding_model, "encode"):
                    return self.embedding_model.encode(doc)

            # Lazy load fallback model to avoid circular import issues
            from src.core.embedding_model import get_document_embedding

            return get_document_embedding(doc)

        raise TypeError("Document must be a string or a numpy array of embeddings.")

    def compute_pairwise_similarity(
        self,
        doc1: str | np.ndarray,
        doc2: str | np.ndarray,
    ) -> float:
        emb1 = self._get_embedding(doc1)
        emb2 = self._get_embedding(doc2)

        pooling_fn = np.max if self.pooling.lower() == "max" else np.mean

        # Pool if chunk embeddings are provided (2D matrix of shape chunks x dim)
        if emb1.ndim == 2 and emb1.shape[0] > 0:
            vec1 = pooling_fn(emb1, axis=0)
        elif emb1.ndim == 1 and emb1.shape[0] > 0:
            vec1 = emb1
        else:
            vec1 = np.zeros(384)

        if emb2.ndim == 2 and emb2.shape[0] > 0:
            vec2 = pooling_fn(emb2, axis=0)
        elif emb2.ndim == 1 and emb2.shape[0] > 0:
            vec2 = emb2
        else:
            vec2 = np.zeros(384)

        # Cosine similarity
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        sim = cosine_similarity(vec1, vec2)[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def compute_matrix(
        self,
        documents: dict[str, str | np.ndarray] | list[str | np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(documents, dict):
            # Resolve all to embeddings
            embeddings = {k: self._get_embedding(v) for k, v in documents.items()}
            df_or_arr = document_similarity_matrix(
                embeddings,
                batch_size=self.batch_size,
                min_threshold=self.min_threshold,
                min_percentile=self.min_percentile,
                pooling=self.pooling,
            )
            if isinstance(df_or_arr, pd.DataFrame):
                return df_or_arr.to_numpy()
            return df_or_arr

        elif isinstance(documents, (list, np.ndarray)):
            # If empty
            if len(documents) == 0:
                return np.array([[]])

            embeddings_list = [self._get_embedding(d) for d in documents]
            df_or_arr = document_similarity_matrix(
                embeddings_list,
                batch_size=self.batch_size,
                min_threshold=self.min_threshold,
                min_percentile=self.min_percentile,
                pooling=self.pooling,
            )
            if isinstance(df_or_arr, pd.DataFrame):
                return df_or_arr.to_numpy()
            return df_or_arr

        else:
            raise TypeError("documents must be a dictionary, list, or numpy array.")

    def search_similar_chunks(
        self,
        query: str | np.ndarray,
        top_k: int = 10,
        exclude_doc: Optional[str] = None,
        threshold: float = 0.0,
    ) -> list[tuple[Any, float]]:
        """Query FAISS index for similar chunks if index is configured."""
        if self.faiss_index is None or self.registry is None:
            raise ValueError("FAISS index and registry are not configured.")

        # Embed query text if necessary
        if isinstance(query, str):
            emb = self._get_embedding(query)
            if emb.ndim == 2:
                emb = np.mean(emb, axis=0)
        else:
            emb = query

        from src.core.faiss_index import search_similar_chunks

        return search_similar_chunks(
            emb,
            self.faiss_index,
            self.registry,
            top_k=top_k,
            exclude_doc=exclude_doc,
            threshold=threshold,
        )


class LexicalSimilarityEngine(BaseSimilarityEngine):
    """
    Concrete Strategy for Lexical Similarity computation.
    Supports TF-IDF, Jaccard, Dice, Overlap, Character/Word N-grams, and Levenshtein.
    """

    def __init__(
        self,
        algorithm: str = "tfidf",
        stopwords: Optional[Iterable[str]] = None,
        ngram_size: int = 3,
        custom_stopwords: Optional[set[str]] = None,
        corpus: Optional[list[str]] = None,
    ):
        """
        Initialize the Lexical Similarity Engine.

        Args:
            algorithm: Name of lexical algorithm ('tfidf', 'jaccard', 'dice', 'overlap', 'ngram', 'char_ngram', 'levenshtein').
            stopwords: Stopwords iterable to use.
            ngram_size: N-gram size (n) for char/word n-gram similarity.
            custom_stopwords: Set of custom stopwords for TF-IDF.
            corpus: Document corpus to fit TF-IDF vectorizer parameters.
        """
        self.algorithm = algorithm.lower()
        self.stopwords = stopwords
        self.ngram_size = ngram_size
        self.custom_stopwords = custom_stopwords
        self.corpus = corpus

    def compute_pairwise_similarity(
        self,
        doc1: str | np.ndarray,
        doc2: str | np.ndarray,
    ) -> float:
        if not isinstance(doc1, str) or not isinstance(doc2, str):
            raise TypeError("Lexical similarity requires string document inputs.")

        if self.algorithm == "tfidf":
            if self.corpus:
                from src.core.lexical_similarity import compute_tfidf_lexical_similarity

                return compute_tfidf_lexical_similarity(doc1, doc2, self.corpus)
            else:
                from src.core.lexical_similarity import calculate_lexical_similarity

                return calculate_lexical_similarity(doc1, doc2, self.custom_stopwords)

        elif self.algorithm == "jaccard":
            from src.core.lexical_similarity import jaccard_similarity

            return jaccard_similarity(doc1, doc2, self.stopwords)

        elif self.algorithm == "dice":
            from src.core.lexical_similarity import dice_coefficient

            return dice_coefficient(doc1, doc2, self.stopwords)

        elif self.algorithm == "overlap":
            from src.core.lexical_similarity import overlap_coefficient

            return overlap_coefficient(doc1, doc2, self.stopwords)

        elif self.algorithm == "ngram":
            from src.core.lexical_similarity import n_gram_overlap

            return n_gram_overlap(doc1, doc2, self.ngram_size, self.stopwords)

        elif self.algorithm == "char_ngram":
            from src.core.lexical_similarity import compute_char_ngram_similarity

            return compute_char_ngram_similarity(doc1, doc2, self.ngram_size)

        elif self.algorithm == "levenshtein":
            try:
                from thefuzz import fuzz

                return fuzz.ratio(doc1, doc2) / 100.0
            except ImportError:
                import difflib

                return difflib.SequenceMatcher(None, doc1, doc2).ratio()

        else:
            raise ValueError(f"Unknown lexical algorithm: {self.algorithm}")

    def compute_matrix(
        self,
        documents: dict[str, str | np.ndarray] | list[str | np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(documents, dict):
            # Resolve all values to str
            doc_dict = {k: str(v) for k, v in documents.items()}

            # TF-IDF has its own optimized matrix implementation
            if self.algorithm == "tfidf":
                from src.core.lexical_similarity import lexical_similarity_matrix

                df = lexical_similarity_matrix(
                    doc_dict, custom_stopwords=self.custom_stopwords
                )
                return df.to_numpy()

            # Otherwise, build pairwise
            doc_names = list(doc_dict.keys())
            n = len(doc_names)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        matrix[i][j] = 1.0
                    else:
                        sim = self.compute_pairwise_similarity(
                            doc_dict[doc_names[i]], doc_dict[doc_names[j]]
                        )
                        matrix[i][j] = sim
                        matrix[j][i] = sim
            return matrix

        elif isinstance(documents, (list, np.ndarray)):
            # Resolve all to str
            doc_list = [str(d) for d in documents]
            n = len(doc_list)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        matrix[i][j] = 1.0
                    else:
                        sim = self.compute_pairwise_similarity(doc_list[i], doc_list[j])
                        matrix[i][j] = sim
                        matrix[j][i] = sim
            return matrix

        else:
            raise TypeError("documents must be a dictionary, list, or numpy array.")


class HybridSimilarityEngine(BaseSimilarityEngine):
    """
    Concrete Strategy for Hybrid Similarity computation.
    Combines semantic and lexical similarity scores using a weighted formula.
    """

    def __init__(
        self,
        semantic_engine: SemanticSimilarityEngine,
        lexical_engine: LexicalSimilarityEngine,
        alpha: float = 0.7,
    ):
        """
        Initialize the Hybrid Similarity Engine.

        Args:
            semantic_engine: Instance of SemanticSimilarityEngine.
            lexical_engine: Instance of LexicalSimilarityEngine.
            alpha: Weight of the semantic engine (0.0 <= alpha <= 1.0).
        """
        if not (0.0 <= alpha <= 1.0):
            from src.errors import SIM_WEIGHT_OUT_OF_RANGE

            raise ValueError(SIM_WEIGHT_OUT_OF_RANGE.format(w=alpha))

        self.semantic_engine = semantic_engine
        self.lexical_engine = lexical_engine
        self.alpha = alpha

    def compute_pairwise_similarity(
        self,
        doc1: str | np.ndarray,
        doc2: str | np.ndarray,
    ) -> float:
        # Check if lexical computation is possible (e.g. requires string inputs)
        try:
            lex_sim = self.lexical_engine.compute_pairwise_similarity(doc1, doc2)
            has_lex = True
        except (TypeError, ValueError):
            lex_sim = 0.0
            has_lex = False

        sem_sim = self.semantic_engine.compute_pairwise_similarity(doc1, doc2)

        if not has_lex:
            return sem_sim

        hybrid_score = self.alpha * sem_sim + (1.0 - self.alpha) * lex_sim
        return float(np.clip(hybrid_score, 0.0, 1.0))

    def compute_matrix(
        self,
        documents: dict[str, str | np.ndarray] | list[str | np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        sem_matrix = self.semantic_engine.compute_matrix(documents)

        try:
            lex_matrix = self.lexical_engine.compute_matrix(documents)
            has_lex = True
        except (TypeError, ValueError):
            lex_matrix = None
            has_lex = False

        if not has_lex or lex_matrix is None:
            return sem_matrix

        if sem_matrix.shape != lex_matrix.shape:
            from src.errors import SIM_SHAPE_MISMATCH

            raise ValueError(SIM_SHAPE_MISMATCH)

        hybrid_matrix = self.alpha * sem_matrix + (1.0 - self.alpha) * lex_matrix
        return hybrid_matrix


class SimilarityEngineFactory:
    """
    Factory class to dynamically instantiate and configure BaseSimilarityEngine subclasses
    based on application settings or parameters.
    """

    @staticmethod
    def create_engine(
        engine_type: str = "semantic",
        **kwargs,
    ) -> BaseSimilarityEngine:
        """
        Create and configure a similarity engine of the specified type.

        Args:
            engine_type: Type of similarity engine ('semantic', 'lexical', 'hybrid').
            **kwargs: Extra parameters to pass to the engine constructor.

        Returns:
            An instance of BaseSimilarityEngine.
        """
        engine_type = engine_type.lower()

        if engine_type == "semantic":
            return SemanticSimilarityEngine(
                embedding_model=kwargs.get("embedding_model"),
                faiss_index=kwargs.get("faiss_index"),
                registry=kwargs.get("registry"),
                batch_size=kwargs.get("batch_size"),
                min_threshold=kwargs.get("min_threshold", 0.0),
                min_percentile=kwargs.get("min_percentile"),
            )

        elif engine_type == "lexical":
            return LexicalSimilarityEngine(
                algorithm=kwargs.get("algorithm", "tfidf"),
                stopwords=kwargs.get("stopwords"),
                ngram_size=kwargs.get("ngram_size", 3),
                custom_stopwords=kwargs.get("custom_stopwords"),
                corpus=kwargs.get("corpus"),
            )

        elif engine_type == "hybrid":
            # For hybrid, we build default semantic and lexical engines if not provided
            sem_engine = kwargs.get("semantic_engine")
            if sem_engine is None:
                sem_engine = SemanticSimilarityEngine(
                    embedding_model=kwargs.get("embedding_model"),
                    faiss_index=kwargs.get("faiss_index"),
                    registry=kwargs.get("registry"),
                    batch_size=kwargs.get("batch_size"),
                    min_threshold=kwargs.get("min_threshold", 0.0),
                    min_percentile=kwargs.get("min_percentile"),
                )

            lex_engine = kwargs.get("lexical_engine")
            if lex_engine is None:
                lex_engine = LexicalSimilarityEngine(
                    algorithm=kwargs.get("algorithm", "tfidf"),
                    stopwords=kwargs.get("stopwords"),
                    ngram_size=kwargs.get("ngram_size", 3),
                    custom_stopwords=kwargs.get("custom_stopwords"),
                    corpus=kwargs.get("corpus"),
                )

            alpha = kwargs.get("alpha", 0.7)
            return HybridSimilarityEngine(
                semantic_engine=sem_engine,
                lexical_engine=lex_engine,
                alpha=alpha,
            )

        else:
            raise ValueError(f"Unknown similarity engine type: {engine_type}")
