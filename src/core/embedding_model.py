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

Recent Additions (Issue #1580):
- Added verify_model_cache_integrity() to detect zero-byte (corrupted)
  cached SentenceTransformer weight files and automatically re-download
  the model when the cached copy is unusable.
"""

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.quantization
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Singleton model loader ─────────────────────────────────────────────────────
_DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None
_quantized_model: SentenceTransformer | None = None


def _apply_dynamic_quantization(model: SentenceTransformer) -> SentenceTransformer:
    """Apply PyTorch dynamic INT8 quantization to the model's Linear layers.

    Dynamic quantization computes the quantization parameters (scale and zero-point)
    for activations dynamically, just like static quantization, but the weights
    are quantized statically. This significantly reduces memory footprint and
    increases inference speed on CPU hosts without requiring a calibration dataset.

    Args:
        model: The loaded SentenceTransformer model instance.

    Returns:
        The quantized SentenceTransformer model.
    """
    logger.info(
        "[embedding_model] Applying dynamic INT8 quantization to Linear layers..."
    )
    try:
        # Quantize only the Linear layers within the transformer modules
        # This preserves the embedding output dimensions while reducing memory
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=False,  # Return a new instance to preserve the original float32 model
        )
        logger.info("[embedding_model] Dynamic quantization applied successfully.")
        return quantized_model
    except Exception as exc:
        logger.warning(
            "[embedding_model] Failed to apply dynamic quantization: %s. "
            "Falling back to float32 model.",
            exc,
        )
        return model


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
    """Manages the SentenceTransformer embedding model lifecycle, fallbacks, and quantization."""

    _instance = None

    def __init__(self, quantize_model: bool = False):
        """Initialize the EmbeddingModelManager.

        Args:
            quantize_model: If True, applies dynamic INT8 quantization to the
                            model's Linear layers to reduce RAM usage by ~50% on CPU.
        """
        self.quantize_model = quantize_model
        self._model = None
        self._quantized_model = None

    @classmethod
    def get_instance(cls, quantize_model: bool = False) -> "EmbeddingModelManager":
        if cls._instance is None:
            cls._instance = cls(quantize_model=quantize_model)
        elif quantize_model and not cls._instance.quantize_model:
            # Update instance if quantization is requested but not yet applied
            cls._instance.quantize_model = True
            cls._instance._quantized_model = None  # Force reload/quantize
        return cls._instance

    def get_model(self) -> SentenceTransformer:
        global _model, _quantized_model

        if self.quantize_model:
            if _quantized_model is not None:
                return _quantized_model
        else:
            if _model is not None:
                return _model

        primary = _get_model_name()
        fallback = "all-MiniLM-L6-v2"
        cache_dir = _get_cache_dir()
        logger.info(f"[embedding_model] Loading model: {primary} ...")
        logger.info(
            f"[embedding_model] Model cache target: {cache_dir or 'default (~/.cache/huggingface)'}"
        )

        try:
            _repair_corrupted_model_cache(_resolve_cache_root(), primary)
            loaded_model = SentenceTransformer(primary, cache_folder=cache_dir)
            device = _detect_device(loaded_model)
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
            loaded_model = SentenceTransformer(fallback, cache_folder=cache_dir)
            device = _detect_device(loaded_model)
            logger.info(
                "Initializing SentenceTransformer model [%s] on device [%s]",
                fallback,
                device,
            )

        if self.quantize_model:
            _quantized_model = _apply_dynamic_quantization(loaded_model)
            return _quantized_model

        _model = loaded_model
        return _model


def _get_model() -> SentenceTransformer:
    """Lazy-load the Sentence Transformer model (singleton pattern)."""
    return EmbeddingModelManager.get_instance().get_model()


# ── Public API ─────────────────────────────────────────────────────────────────


# Model weight filenames inspected by verify_model_cache_integrity() (issue #1580)
_MODEL_WEIGHT_FILENAMES = ("pytorch_model.bin", "model.safetensors")


def _resolve_cache_root() -> Path:
    """Resolve the HuggingFace hub cache root used for model downloads."""
    configured = _get_cache_dir()
    if configured:
        return Path(configured)
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


def _model_cache_subdir(cache_root: Path, model_name: str) -> Path:
    """Return the HuggingFace hub cache sub-directory for ``model_name``."""
    hub_name = model_name.replace("/", "--")
    return cache_root / f"models--{hub_name}"


def verify_model_cache_integrity(cache_dir: Path) -> bool:
    """Verify cached SentenceTransformer weight files are not corrupted.

    Acceptance criteria (issue #1580):
    - Inspects the model weight files ``pytorch_model.bin`` and
      ``model.safetensors`` for zero-byte sizes.
    - Returns ``True`` when the cache holds no corrupted weight files.
    - Returns ``False`` when any cached weight file is zero bytes or
      unreadable, allowing the caller to re-download the model.

    Args:
        cache_dir: Path to the HuggingFace cache directory to inspect.
            This may be the hub root (``~/.cache/huggingface/hub``) or a
            model-specific ``models--<org>--<name>`` sub-directory.

    Returns:
        ``True`` if every cached weight file has a non-zero, readable
        size (or no weight files are cached yet); ``False`` when any
        weight file is zero bytes or cannot be stat-ed.

    Example:
        >>> from pathlib import Path
        >>> from src.core.embedding_model import verify_model_cache_integrity
        >>> verify_model_cache_integrity(Path("~/.cache/huggingface/hub").expanduser())
        True
    """
    cache_path = Path(cache_dir)

    if not cache_path.is_dir():
        logger.debug(
            "[embedding_model] Model cache dir %s not found; nothing to verify.",
            cache_path,
        )
        return True

    corrupted: List[tuple[Path, str]] = []
    for root, _, filenames in os.walk(cache_path):
        for filename in filenames:
            if filename not in _MODEL_WEIGHT_FILENAMES:
                continue
            weight_path = Path(root) / filename
            try:
                size = weight_path.stat().st_size
            except OSError as exc:
                corrupted.append((weight_path, f"unreadable ({exc})"))
                continue
            if size == 0:
                corrupted.append((weight_path, "zero-byte"))

    if corrupted:
        for weight_path, reason in corrupted:
            logger.warning(
                "[embedding_model] Corrupted model weight file detected: %s (%s).",
                weight_path,
                reason,
            )
        return False

    return True


def _repair_corrupted_model_cache(cache_root: Path, model_name: str) -> None:
    """Remove a corrupted cached model so it is re-downloaded on load.

    When the cached copy of ``model_name`` contains zero-byte weight
    files, the entire ``models--<name>`` cache directory is removed so
    the next ``SentenceTransformer`` load re-downloads a healthy copy
    (issue #1580).
    """
    model_cache_dir = _model_cache_subdir(cache_root, model_name)
    if not model_cache_dir.is_dir():
        return

    if not verify_model_cache_integrity(model_cache_dir):
        logger.warning(
            "[embedding_model] Corrupted cache detected for model %s at %s; "
            "removing it so the model is re-downloaded.",
            model_name,
            model_cache_dir,
        )
        shutil.rmtree(model_cache_dir, ignore_errors=True)


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
