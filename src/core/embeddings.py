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
src/core/embeddings.py
----------------------
Generates semantic embeddings for text documents using
SentenceTransformers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

# Load a lightweight, fast model for semantic embeddings.
_model = SentenceTransformer("all-MiniLM-L6-v2")


def _encode_batch(text_batch: Sequence[str]) -> np.ndarray:
    """Encode one ordered batch through the shared model."""
    encoded = _model.encode(
        list(text_batch),
        convert_to_numpy=True,
    )
    return np.asarray(encoded)


def _split_into_batches(
    texts: Sequence[str],
    worker_count: int,
) -> list[Sequence[str]]:
    """Split texts into contiguous, non-empty worker batches."""
    batch_size = ceil(len(texts) / worker_count)
    return [
        texts[start : start + batch_size] for start in range(0, len(texts), batch_size)
    ]


def generate_embeddings(
    texts: list[str],
    num_threads: int = 4,
) -> np.ndarray:
    """Generate dense embeddings while preserving input order.

    For ``num_threads > 1``, the input is divided into contiguous
    batches and encoded through a :class:`ThreadPoolExecutor`.
    Results are concatenated in submission order, so row ``n`` always
    corresponds to input text ``n``.

    The single-threaded path deliberately avoids creating an executor,
    preserving the original behavior and minimizing overhead for small
    workloads.

    Args:
        texts: Input text chunks in the required output order.
        num_threads: Maximum number of worker threads. Defaults to 4.

    Returns:
        A two-dimensional NumPy array with one embedding row per input
        string. Empty input returns an empty ``(0, 0)`` float array.

    Raises:
        TypeError: If ``num_threads`` is not an integer or any text is
            not a string.
        ValueError: If ``num_threads`` is less than one.
        RuntimeError: If worker batches return incompatible shapes.
        Exception: Worker/model exceptions are propagated unchanged.
    """
    if isinstance(num_threads, bool) or not isinstance(
        num_threads,
        int,
    ):
        raise TypeError("num_threads must be an integer.")
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1.")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("All embedding inputs must be strings.")

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    worker_count = min(num_threads, len(texts))
    if worker_count == 1:
        result = _encode_batch(texts)
        if result.ndim != 2 or result.shape[0] != len(texts):
            raise RuntimeError("Embedding model returned an unexpected output shape.")
        return result

    batches = _split_into_batches(texts, worker_count)

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="embedding-worker",
    ) as executor:
        # executor.map preserves the input batch order while still
        # running the batch calls concurrently.
        encoded_batches = list(executor.map(_encode_batch, batches))

    if not encoded_batches:
        return np.empty((0, 0), dtype=np.float32)

    embedding_dimensions = {
        batch.shape[1] for batch in encoded_batches if batch.ndim == 2
    }
    if len(embedding_dimensions) != 1 or any(
        batch.ndim != 2 for batch in encoded_batches
    ):
        raise RuntimeError("Embedding workers returned incompatible output shapes.")

    result = np.concatenate(encoded_batches, axis=0)
    if result.shape[0] != len(texts):
        raise RuntimeError("Embedding workers returned an unexpected row count.")

    return result
