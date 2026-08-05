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

Recent Additions (Issue #920):
- Modified embed_chunks() and embed_documents() to process embeddings in
  explicit Python-level mini-batches. This prevents memory spikes when
  processing large document sets (100+ files) by yielding memory back to
  the garbage collector between batch forward passes.
"""

import logging
import os
import gc
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


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
            logger.info(
                f"[embedding_model] Model cache target: {cache_dir or 'default (~/.cache/huggingface)'}"
            )
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


def embed_chunks(chunks: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks using explicit mini-batching.

    To optimize GPU/CPU memory utilization and prevent memory spikes when
    processing large document sets (100+ files), this function processes
    the input chunks in Python-level mini-batches of size `batch_size`.
    After each batch is encoded, the intermediate results are accumulated
    and memory is implicitly yielded back to the garbage collector.

    Args:
        chunks: List of text strings to embed.
        batch_size: Number of texts encoded per forward pass. Defaults to 32
                    to balance throughput and memory consumption.

    Returns:
        numpy array of shape (N, 384) where N = len(chunks). Returns an
        empty array of shape (0, 384) if the input list is empty.
    """
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model()
    all_embeddings: List[np.ndarray] = []
    total_chunks = len(chunks)

    logger.debug(
        "[embedding_model] Processing %d chunks in mini-batches of size %d",
        total_chunks,
        batch_size,
    )

    # Process in explicit mini-batches to optimize memory utilization
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]

        # Encode the current mini-batch
        # show_progress_bar=False keeps console clean in Streamlit
        # normalize_embeddings=True ensures L2-normalisation (cosine sim = dot product)
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        all_embeddings.append(batch_embeddings)

        # Periodically trigger garbage collection for very large datasets
        # to free up memory from intermediate tensor allocations
        if (i // batch_size) % 10 == 0 and i > 0:
            gc.collect()

    # Combine all mini-batch results into a single contiguous NumPy array
    # np.vstack efficiently stacks arrays of shape (batch_size, 384)
    combined_embeddings = np.vstack(all_embeddings)

    return combined_embeddings


def embed_documents(
    chunked_docs: Dict[str, List[str]], batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Embed all chunks across multiple documents using optimized mini-batching.

    This function flattens all document chunks into a single list, processes
    them through embed_chunks() which utilizes mini-batching to prevent
    memory spikes, and then maps the resulting embeddings back to their
    respective documents.

    Args:
        chunked_docs: Dict mapping document name → list of chunk strings.
        batch_size: Batch size forwarded to embed_chunks(). Defaults to 32.

    Returns:
        Dict mapping document name → numpy array of embeddings (shape: N×384).
        Documents with no chunks will have an empty array of shape (0, 384).
    """
    embeddings: Dict[str, np.ndarray] = {}
    all_chunks: List[str] = []
    doc_chunk_counts: List[int] = []
    doc_names: List[str] = []

    # Initialize all documents with empty arrays to ensure consistent return types
    for doc_name in chunked_docs.keys():
        embeddings[doc_name] = np.empty((0, 384), dtype=np.float32)

    # Flatten all chunks while tracking document boundaries
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
        logger.info("[embedding_model] No chunks to embed across all documents.")
        return embeddings

    logger.info(
        "[embedding_model] Embedding %d total chunks across %d documents with batch_size=%d",
        len(all_chunks),
        len(doc_names),
        batch_size,
    )

    # Call embed_chunks once for the entire flattened batch of chunks
    # embed_chunks handles the internal mini-batching for memory optimization
    all_embeddings = embed_chunks(all_chunks, batch_size=batch_size)

    # Map the combined embeddings back to the original documents
    start_idx = 0
    for doc_name, count in zip(doc_names, doc_chunk_counts):
        end_idx = start_idx + count
        embeddings[doc_name] = all_embeddings[start_idx:end_idx]
        start_idx = end_idx

    logger.info("[embedding_model] Document embedding complete.")
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
