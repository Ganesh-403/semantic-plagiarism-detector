from __future__ import annotations

"""
embedding_model.py
------------------
Wrapper around SentenceTransformers for generating semantic embeddings.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - Multilingual support for English and many other languages
  - 384-dimensional embeddings
  - Strong performance on semantic similarity tasks
  - MIT licensed; safe for academic use
"""

import logging
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from src.core.config import EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# ── Singleton model loader ─────────────────────────────────────────────────────
_DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def _detect_device(model: SentenceTransformer | None = None) -> str:
    """Detect active PyTorch compute device (cpu, cuda, or mps)."""
    if model is not None and hasattr(model, "device"):
        dev = getattr(model, "device")
        if isinstance(dev, str):
            return dev
        if hasattr(dev, "type") and isinstance(getattr(dev, "type", None), str):
            return dev.type
    try:
        if (
            hasattr(torch, "cuda")
            and hasattr(torch.cuda, "is_available")
            and torch.cuda.is_available()
        ):
            return "cuda"
    except Exception:
        pass
    try:
        if (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and hasattr(torch.backends.mps, "is_available")
            and torch.backends.mps.is_available()
        ):
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_model_name() -> str:
    """Return the configured sentence-transformers model name."""
    return os.getenv("SEMANTIC_PLAGIARISM_MODEL", _DEFAULT_MODEL_NAME)


def _get_cache_dir() -> str | None:
    """Return the configured HuggingFace model cache directory, if any."""
    return os.getenv("HF_HUB_CACHE") or os.getenv("TRANSFORMERS_CACHE")

def get_embedding_model_info() -> tuple[str, int]:
    """
    Return the active embedding model name and embedding dimension.
    """
    model = _get_model()
    return _get_model_name(), model.get_sentence_embedding_dimension()


class EmbeddingModelManager:
    """Manages the SentenceTransformer embedding model lifecycle and fallbacks."""
    _instance = None

    @classmethod
    def get_instance(cls) -> EmbeddingModelManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model(self) -> SentenceTransformer:
        global _model
        if _model is None:
            primary = _get_model_name()
            fallback = "all-MiniLM-L6-v2"
            cache_dir = _get_cache_dir()
            logger.info(f"[embedding_model] Loading model: {primary} ...")
            logger.info(f"[embedding_model] Model cache target: {cache_dir or 'default (~/.cache/huggingface)'}")
            try:
                _model = SentenceTransformer(primary, cache_folder=cache_dir)
                device = _detect_device(_model)
                logger.info(
                    "Initializing SentenceTransformer model [%s] on device [%s]",
                    primary,
                    device,
                )
                logger.info("[embedding_model] Model loaded successfully.")
            except Exception:
                logger.warning(
                    "Primary embedding model %s unavailable. Falling back to %s",
                    primary,
                    fallback,
                )
                _model = SentenceTransformer(fallback, cache_folder=cache_dir)
                device = _detect_device(_model)
                logger.info(
                    "Initializing SentenceTransformer model [%s] on device [%s]",
                    fallback,
                    device,
                )
        return _model


def _get_model() -> SentenceTransformer:
    """Lazy-load the Sentence Transformer model (singleton pattern)."""
    return EmbeddingModelManager.get_instance().get_model()



# ── Public API ─────────────────────────────────────────────────────────────────


def embed_chunks(
    chunks: List[str], batch_size: int = EMBEDDING_BATCH_SIZE
) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks:     List of text strings to embed.
        batch_size: Number of texts encoded per forward pass (defaults to EMBEDDING_BATCH_SIZE).

    Returns:
        numpy array of shape (N, 384) where N = len(chunks).
    """
    if not chunks:
        return np.array([])

    model = _get_model()
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=False,  # Keep console clean in Streamlit
        normalize_embeddings=True,  # L2-normalise → cosine sim = dot product
    )
    return embeddings


def embed_documents(
    chunked_docs: dict, batch_size: int = EMBEDDING_BATCH_SIZE
) -> dict:
    """
    Embed all chunks across multiple documents.

    Args:
        chunked_docs: Dict mapping document name → list of chunk strings.
        batch_size:   Batch size forwarded to encode().

    Returns:
        Dict mapping document name → numpy array of embeddings (shape: N×384).
    """
    embeddings = {}
    all_chunks = []
    doc_chunk_counts = []
    doc_names = []

    # Initialize all documents with empty arrays
    for doc_name in chunked_docs.keys():
        embeddings[doc_name] = np.array([])

    for doc_name, chunks in chunked_docs.items():
        if not chunks:
            logger.warning(
                f"[embedding_model] Warning: '{doc_name}' has no chunks. Skipping."
            )
            continue
        all_chunks.extend(chunks)
        doc_chunk_counts.append(len(chunks))
        doc_names.append(doc_name)

    if not all_chunks:
        return embeddings

    # Call embed_chunks once for the entire batch of chunks across all documents
    all_embeddings = embed_chunks(all_chunks, batch_size=batch_size)

    # Map the embeddings back to the original documents
    start_idx = 0
    for doc_name, count in zip(doc_names, doc_chunk_counts):
        end_idx = start_idx + count
        embeddings[doc_name] = all_embeddings[start_idx:end_idx]
        start_idx = end_idx

    return embeddings


def get_document_embedding(doc_embedding: np.ndarray) -> np.ndarray:
    """
    Compute a single document-level embedding by averaging its chunk embeddings.

    Args:
        doc_embedding: Array of shape (N, 384) for N chunks.

    Returns:
        1-D array of shape (384,).
    """
    if doc_embedding.ndim == 1:
        return doc_embedding  # Already a single embedding
    return np.mean(doc_embedding, axis=0)
