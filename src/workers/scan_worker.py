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
src/workers/scan_worker.py
--------------------------
Isolated worker process logic for the distributed task queue
(Issue #3146).

Each worker runs in its own thread (or can be launched as a separate
process via ``python -m src.workers.scan_worker``). It pulls jobs from
the :class:`TaskQueue`, executes the embedding pipeline on the batch
payload, and reports the result back.

The payload schema (stored as JSON in ``task_jobs.payload``):

    {
        "files": {"doc1.pdf": "<base64 bytes>", "doc2.txt": "<base64 bytes>"},
        "threshold": 0.59,
        "top_k": 3,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

The result schema (stored as JSON in ``task_jobs.result``):

    {
        "documents_processed": 2,
        "total_chunks": 15,
        "flagged_pairs": 3,
        "similarity_matrix": [[1.0, 0.72], [0.72, 1.0]],
        "document_names": ["doc1.pdf", "doc2.txt"]
    }
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.db import task_db
from src.workers.task_queue import TaskQueue, get_default_queue

logger = logging.getLogger(__name__)


def execute_scan_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute the embedding + similarity pipeline on a job payload.

    This function is intentionally pure (no DB, no queue interaction)
    so it can be unit-tested in isolation.

    Args:
        payload: The decoded JSON payload from ``task_jobs.payload``.
            Must contain a ``files`` dict mapping filename → base64 bytes.

    Returns:
        A result dict with the similarity matrix and chunk count.

    Raises:
        ValueError: If the payload is malformed.
        Exception: Any pipeline error propagates to the caller (the
            worker loop catches it and marks the job as failed).
    """
    files_b64: dict[str, str] = payload.get("files", {})
    if not files_b64:
        raise ValueError("Payload must contain a non-empty 'files' dict.")

    threshold: float = payload.get("threshold", 0.59)
    top_k: int = payload.get("top_k", 3)
    chunk_size: int = payload.get("chunk_size", 1000)
    chunk_overlap: int = payload.get("chunk_overlap", 200)

    # Decode base64 → raw bytes.
    files_bytes: dict[str, bytes] = {}
    for name, b64_data in files_b64.items():
        try:
            files_bytes[name] = base64.b64decode(b64_data)
        except Exception as exc:
            raise ValueError(
                f"Failed to decode base64 for file '{name}': {exc}"
            ) from exc

    # Step 1: Extract text from each file.
    raw_texts: dict[str, str] = {}
    for name, data in files_bytes.items():
        try:
            from src.core.document_parser import extract_text

            extracted = extract_text(data, name)
            if extracted and extracted.strip():
                raw_texts[name] = extracted
        except Exception as exc:
            logger.warning("Failed to extract text from %s: %s", name, exc)
            continue

    if not raw_texts:
        raise ValueError("No text could be extracted from any of the provided files.")

    # Step 2: Chunk documents.
    from src.core.text_chunking import chunk_documents

    chunked: dict[str, list[str]] = chunk_documents(
        raw_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Flatten chunks for embedding.
    all_chunks: list[str] = []
    for doc_name, chunks in chunked.items():
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks were produced from the provided documents.")

    # Step 3: Embed chunks.
    from src.core.embedding_model import embed_chunks

    embeddings: np.ndarray = embed_chunks(all_chunks)

    # Step 4: Compute pairwise similarity (if ≥2 docs).
    flagged_pairs = 0
    sim_matrix: list[list[float]] = []

    if len(raw_texts) >= 2:
        try:
            from src.core.similarity import document_similarity_matrix

            doc_names = list(raw_texts.keys())
            doc_embeddings: list[np.ndarray] = []
            offset = 0
            for name in doc_names:
                n_chunks = len(chunked[name])
                if n_chunks > 0:
                    doc_emb = embeddings[offset : offset + n_chunks].mean(axis=0)
                    doc_embeddings.append(doc_emb)
                offset += n_chunks

            if len(doc_embeddings) >= 2:
                emb_matrix = np.stack(doc_embeddings)
                sim_matrix = document_similarity_matrix(emb_matrix).tolist()
                for i in range(len(sim_matrix)):
                    for j in range(i + 1, len(sim_matrix)):
                        if sim_matrix[i][j] >= threshold:
                            flagged_pairs += 1
        except Exception as exc:
            logger.warning("Similarity computation failed: %s", exc)

    return {
        "documents_processed": len(raw_texts),
        "total_chunks": len(all_chunks),
        "flagged_pairs": flagged_pairs,
        "similarity_matrix": sim_matrix,
        "document_names": list(raw_texts.keys()),
    }


class ScanWorker:
    """A single worker that pulls jobs from the queue and executes them.

    Multiple ``ScanWorker`` instances can run concurrently in separate
    threads to parallelise batch processing.
    """

    def __init__(
        self,
        worker_id: str | None = None,
        queue: TaskQueue | None = None,
    ) -> None:
        self.worker_id = worker_id or f"scan-worker-{threading.get_ident()}"
        self._queue = queue or get_default_queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the worker in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run, name=self.worker_id, daemon=True
        )
        self._thread.start()
        logger.info("Started scan worker: %s", self.worker_id)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        self._queue.signal_shutdown()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        """Main worker loop: dequeue → execute → report result."""
        while not self._stop_event.is_set():
            try:
                job = self._queue.dequeue(timeout=5.0)
                if job is None:
                    continue
                self._process_job(job)
            except Exception as exc:
                logger.error("Worker %s unhandled error: %s", self.worker_id, exc)
                time.sleep(1.0)

    def _process_job(self, job: dict[str, Any]) -> None:
        """Execute a single job and report the result back to the queue."""
        job_id = job["id"]
        payload = job.get("payload", {})
        logger.info("Worker %s processing job %s", self.worker_id, job_id)

        try:
            result = execute_scan_job(payload)
            self._queue.complete(job_id, result)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            self._queue.fail(job_id, error_msg)


# ── CLI entry point ────────────────────────────────────────────


def main() -> None:
    """Run a standalone scan worker process.

    Usage::

        python -m src.workers.scan_worker
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(levelname)s %(name)s — %(message)s",
    )
    logger.info("Starting standalone scan worker…")

    q = get_default_queue()
    requeued = q.requeue_stale_processing()
    if requeued:
        logger.info("Re-queued %d stale PROCESSING jobs.", requeued)

    worker = ScanWorker()
    worker.start()

    try:
        while worker._thread and worker._thread.is_alive():
            worker._thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
        worker.stop()
        logger.info("Worker stopped. Goodbye.")


if __name__ == "__main__":
    main()
