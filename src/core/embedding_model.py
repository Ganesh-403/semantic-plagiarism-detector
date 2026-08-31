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

Recent Additions (Issue #3479):
- Added release_large_batch_memory() so callers (_process_scan_job and the
  batch CLI commands) can explicitly run gc.collect() and, when CUDA is
  available, torch.cuda.empty_cache() after scans larger than 20 documents.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.quantization
from sentence_transformers import SentenceTransformer

from src.core.text_chunking import ChunkString
from src.exceptions import ModelInitializationError

logger = logging.getLogger(__name__)

try:
    import optimum.onnxruntime
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

# ── Singleton model loader ─────────────────────────────────────────────────────
_DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None
_quantized_model: SentenceTransformer | None = None
_model_lock = threading.Lock()



def _apply_dynamic_quantization(model: SentenceTransformer) -> SentenceTransformer:
    """Apply PyTorch dynamic INT8 quantization to the model's Linear layers.

    Dynamic quantization is intended for CPU inference. Apple Silicon's MPS
    backend does not benefit from this INT8 path, so MPS models are returned
    unchanged to preserve native floating-point performance.

    Args:
        model: The loaded SentenceTransformer model instance.

    Returns:
        The quantized SentenceTransformer model, or the original model when
        running on MPS.
    """
    device = _detect_device(model)
    if device == "mps":
        logger.warning(
            "[embedding_model] Dynamic INT8 quantization is disabled on MPS; "
            "using the unquantized floating-point model to preserve performance."
        )
        return model

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
    """Detect the active PyTorch compute device.

    Supports NVIDIA CUDA, AMD ROCm (exposed by PyTorch through the CUDA
    device API), Intel XPU, and Apple MPS. ROCm intentionally returns
    ``"cuda"`` because PyTorch uses ``torch.device("cuda")`` for HIP
    devices as well.
    """
    if model is not None and hasattr(model, "device"):
        dev = getattr(model, "device")
        if isinstance(dev, str):
            return dev
        if hasattr(dev, "type") and isinstance(getattr(dev, "type", None), str):
            return dev.type

    # Intel oneAPI/XPU devices. Keep this ahead of CUDA so an available XPU
    # is not shadowed by another backend exposed by the same PyTorch build.
    try:
        xpu = getattr(torch, "xpu", None)
        if (
            xpu is not None
            and hasattr(xpu, "is_available")
            and xpu.is_available()
        ):
            return "xpu"
    except Exception:
        pass

    # PyTorch exposes AMD ROCm through the CUDA API. ``torch.version.hip`` is
    # the reliable indicator that the installed PyTorch build targets HIP.
    # ``torch.backends.cuda.is_built()`` is checked as a safe fallback for
    # CUDA-enabled builds where the HIP version metadata is unavailable.
    try:
        cuda = getattr(torch, "cuda", None)
        cuda_available = (
            cuda is not None
            and hasattr(cuda, "is_available")
            and cuda.is_available()
        )
        cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
        cuda_built = bool(
            cuda_backend is not None
            and hasattr(cuda_backend, "is_built")
            and cuda_backend.is_built()
        )
        hip_version = getattr(getattr(torch, "version", None), "hip", None)

        if cuda_available and hip_version:
            logger.info("[embedding_model] AMD ROCm/HIP device detected.")
            return "cuda"
        if cuda_available and cuda_built:
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


def is_quantization_enabled() -> bool:
    """Check if dynamic INT8 quantization is enabled via ENABLE_EMBEDDING_QUANTIZATION environment variable.

    Returns:
        True if ENABLE_EMBEDDING_QUANTIZATION is set to 'true', '1', or 'yes' (case-insensitive).
    """
    env_val = os.getenv("ENABLE_EMBEDDING_QUANTIZATION", "false").lower().strip()
    return env_val in ("true", "1", "yes")


def get_device(model: "SentenceTransformer | None" = None) -> str:
    """Public helper function to inspect and get the target hardware compute device.

    Checks available hardware backends in priority order:
    1. Intel XPU acceleration
    2. NVIDIA CUDA / AMD ROCm HIP GPU acceleration
    3. Apple Silicon Metal Performance Shaders (MPS) acceleration via torch.backends.mps.is_available()
    4. CPU fallback

    Args:
        model: Optional SentenceTransformer model instance to inspect active device attribute.

    Returns:
        Device string identifier ("cuda", "mps", "xpu", or "cpu").
    """
    return _detect_device(model)


class EmbeddingModelManager:
    """Manages the SentenceTransformer embedding model lifecycle, fallbacks, and quantization."""

    _instance = None
    _instance_lock = threading.Lock()

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
    def get_instance(cls, quantize_model: bool | None = None) -> "EmbeddingModelManager":
        if quantize_model is None:
            quantize_model = is_quantization_enabled()

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(quantize_model=quantize_model)
        elif quantize_model and not cls._instance.quantize_model:
            with cls._instance_lock:
                if quantize_model and not cls._instance.quantize_model:
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

        with _model_lock:
            if self.quantize_model:
                if _quantized_model is not None:
                    return _quantized_model
            else:
                if _model is not None:
                    return _model

            primary = _get_model_name()
            fallback = os.getenv("SEMANTIC_PLAGIARISM_FALLBACK_MODEL", "all-MiniLM-L6-v2")
            cache_dir = _get_cache_dir()
            logger.info(f"[embedding_model] Loading model: {primary} ...")
            logger.info(
                f"[embedding_model] Model cache target: {cache_dir or 'default (~/.cache/huggingface)'}"
            )

            try:
                _repair_corrupted_model_cache(_resolve_cache_root(), primary)
                kwargs = {}
                if _ONNX_AVAILABLE:
                    kwargs["backend"] = "onnx"
                    logger.info("[embedding_model] optimum[onnxruntime] detected. Enabling ONNX backend for 2x-3x CPU speedup.")

                loaded_model = SentenceTransformer(primary, cache_folder=cache_dir, **kwargs)
                device = _detect_device(loaded_model)
                logger.info(
                    "Initialized Embedding Model: %s | Dimensions: %d | Target Device: %s",
                    primary,
                    loaded_model.get_sentence_embedding_dimension(),
                    device,
                )
                logger.info("[embedding_model] Model loaded successfully.")
            except Exception as primary_exc:
                logger.warning(
                    "Primary embedding model %s unavailable. Falling back to %s",
                    primary,
                    fallback,
                )
                try:
                    kwargs_fallback = {}
                    if _ONNX_AVAILABLE:
                        kwargs_fallback["backend"] = "onnx"
                    loaded_model = SentenceTransformer(fallback, cache_folder=cache_dir, **kwargs_fallback)
                except Exception as fallback_exc:
                    raise ModelInitializationError(
                        "Unable to initialize the embedding model. Both the configured "
                        f"primary model '{primary}' and fallback model '{fallback}' "
                        "failed to load. This usually means the models are unavailable "
                        "from the configured Hugging Face cache or cannot be downloaded. "
                        "For offline or air-gapped deployments, download both models "
                        "on an internet-connected machine with 'hf download', copy "
                        "the model directories into the deployment environment, and "
                        "configure SEMANTIC_PLAGIARISM_MODEL to point to the local "
                        "primary model directory. For example: `hf download "
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 "
                        "--local-dir /opt/models/paraphrase-multilingual-MiniLM-L12-v2` "
                        "and `hf download sentence-transformers/all-MiniLM-L6-v2 "
                        "--local-dir /opt/models/all-MiniLM-L6-v2`. "
                        f"Primary error: {primary_exc!r}; fallback error: {fallback_exc!r}"
                    ) from fallback_exc
                device = _detect_device(loaded_model)
                logger.info(
                    "Initialized Fallback Embedding Model: %s | Dimensions: %d | Target Device: %s",
                    fallback,
                    loaded_model.get_sentence_embedding_dimension(),
                    device,
                )

            if self.quantize_model:
                _quantized_model = _apply_dynamic_quantization(loaded_model)
                return _quantized_model

            _model = loaded_model
            return _model


def _get_model() -> SentenceTransformer:
    """Lazy-load the Sentence Transformer model (singleton pattern)."""
    quantize = is_quantization_enabled()
    return EmbeddingModelManager.get_instance(quantize_model=quantize).get_model()


# ── Public API ─────────────────────────────────────────────────────────────────


# Model weight file extensions inspected by verify_model_cache_integrity()
# (issue #1580, extended for issue #2919). Extension-based matching avoids
# breaking when HuggingFace ships new weight formats (e.g. GGUF, ONNX)
# under different filenames.
_MODEL_WEIGHT_EXTENSIONS = (".bin", ".safetensors", ".onnx", ".gguf")


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

    Acceptance criteria (issue #1580, extended by #2919):
    - Inspects any cached file matching a known model weight extension
      (``.bin``, ``.safetensors``, ``.onnx``, ``.gguf``) for zero-byte
      sizes.    - Returns ``True`` when the cache holds no corrupted weight files.
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

    corrupted: list[tuple[Path, str]] = []
    for root, _, filenames in os.walk(cache_path):
        for filename in filenames:
            if not filename.endswith(_MODEL_WEIGHT_EXTENSIONS):
                continue
            weight_path = Path(root) / filename
            try:
                size = weight_path.stat().st_size
            except OSError as exc:
                corrupted.append((weight_path, f"unreadable ({exc})"))
                continue
            if size <= 1024 * 1024:
                corrupted.append((weight_path, "too small (<= 1MB)"))

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


# Issue #3479: batches larger than this many documents trigger explicit
# garbage collection (and CUDA cache release when applicable).
LARGE_BATCH_GC_THRESHOLD = 20


def release_large_batch_memory(batch_size: int) -> None:
    """Explicitly free heap memory after large batch scans (Issue #3479).

    Processing 100+ documents in a single batch leaves NumPy arrays and
    PyTorch tensors on the heap. ``_process_scan_job`` and the batch CLI
    commands call this once a scan finishes so those intermediates are
    released promptly instead of lingering until the next allocation cycle.

    Args:
        batch_size: Number of documents/chunks processed in the finished
            batch. Cleanup only runs when this exceeds
            ``LARGE_BATCH_GC_THRESHOLD``.
    """
    if batch_size <= LARGE_BATCH_GC_THRESHOLD:
        return

    logger.debug(
        "[embedding_model] Releasing memory after large batch of %d items",
        batch_size,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_DEFAULT_BATCH_SIZE = 32


def _get_embedding_batch_size() -> int:
    """Return the configured batch size for chunk embeddings from environment."""
    raw = os.getenv("EMBEDDING_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE))
    try:
        val = int(raw)
        return val if val > 0 else _DEFAULT_BATCH_SIZE
    except (ValueError, TypeError):
        return _DEFAULT_BATCH_SIZE


def embed_chunks(
    chunks: list[str],
    batch_size: int | None = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks using explicit mini-batching.

    To optimize GPU/CPU memory utilization and prevent memory spikes when
    processing large document sets (100+ files), this function processes
    the input chunks in Python-level mini-batches of size `batch_size`.
    After each batch is encoded, the intermediate results are accumulated
    and memory is implicitly yielded back to the garbage collector.

    Args:
        chunks: List of text strings to embed.
        batch_size: Number of texts encoded per forward pass. Defaults to None,
                    which resolves to EMBEDDING_BATCH_SIZE from environment (default: 32)
                    to balance throughput and memory consumption.
        cancel_callback: Optional callback returning True if processing should be cancelled.

    Returns:
        numpy array of shape (N, 384) where N = len(chunks). Returns an
        empty array of shape (0, 384) if the input list is empty.
    """
    if not chunks:
        model = _get_model()
        dimension = model.get_sentence_embedding_dimension()
        return np.empty((0, dimension), dtype=np.float32)
    if batch_size is None:
        batch_size = _get_embedding_batch_size()

    model = _get_model()
    all_embeddings: list[np.ndarray] = []
    total_chunks = len(chunks)

    logger.debug(
        "[embedding_model] Processing %d chunks in mini-batches of size %d",
        total_chunks,
        batch_size,
    )

    # Process in explicit mini-batches to optimize memory utilization
    for i in range(0, total_chunks, batch_size):
        if cancel_callback and cancel_callback():
            logger.info("[embedding_model] Embedding forward pass cancelled by callback.")
            raise RuntimeError("Scan job cancelled")

        batch = [
            chunk.text if isinstance(chunk, ChunkString) else chunk
            for chunk in chunks[i : i + batch_size]
        ]

        # Encode the current mini-batch
        # show_progress_bar=False keeps console clean in Streamlit
        # normalize_embeddings=True ensures L2-normalisation (cosine sim = dot product)
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
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
    chunked_docs: dict[str, list[str]], batch_size: int | None = None
) -> dict[str, np.ndarray]:
    """
    Embed all chunks across multiple documents using optimized mini-batching.

    This function flattens all document chunks into a single list, processes
    them through embed_chunks() which utilizes mini-batching to prevent
    memory spikes, and then maps the resulting embeddings back to their
    respective documents.

    Args:
        chunked_docs: Dict mapping document name → list of chunk strings.
        batch_size: Batch size forwarded to embed_chunks(). Defaults to None,
                    which resolves to EMBEDDING_BATCH_SIZE from environment (default: 32).

    Returns:
        Dict mapping document name → numpy array of embeddings (shape: N×384).
        Documents with no chunks will have an empty array of shape (0, 384).
    """
    embeddings: dict[str, np.ndarray] = {}
    all_chunks: list[str] = []
    doc_chunk_counts: list[int] = []
    doc_names: list[str] = []

    # Initialize all documents with empty arrays to ensure consistent return types
    model = _get_model()
    embedding_dimension = model.get_sentence_embedding_dimension()

    for doc_name in chunked_docs.keys():
        embeddings[doc_name] = np.empty(
            (0, embedding_dimension),
            dtype=np.float32,
        )
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

    effective_batch_size = (
        batch_size if batch_size is not None else _get_embedding_batch_size()
    )

    logger.info(
        "[embedding_model] Embedding %d total chunks across %d documents with batch_size=%d",
        len(all_chunks),
        len(doc_names),
        effective_batch_size,
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


def warmup_embedding_model() -> bool:
    """Executes a dummy inference pass on startup to pre-load weights
    and trigger JIT compilation, eliminating first-request latency overhead.
    """
    logger.info("Initializing embedding model warmup routine...")
    try:
        dummy_text = "Warmup"
        _ = embed_chunks([dummy_text])
        logger.info("Embedding model warmup completed successfully. JIT layers compiled.")
        return True
    except Exception as e:
        logger.error(f"Embedding model warmup failed: {str(e)}", exc_info=True)
        # Fail gracefully to avoid blocking the main runtime setup if the network/device drops
        return False
