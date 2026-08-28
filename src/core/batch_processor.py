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
Batch Processing Optimization for Large Document Sets & Dashboard Engine.

Provides parallel processing, incremental indexing, progress tracking,
and batch job management capabilities.
"""

import json
import logging
import multiprocessing
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================


class BatchStatus(Enum):
    """Status of a batch processing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class BatchPriority(Enum):
    """Priority levels for batch jobs."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class BatchJob:
    """Represents a single batch processing job."""

    job_id: str
    name: str = ""
    file_paths: list[str] = field(default_factory=list)
    document_paths: list[str] = field(default_factory=list)
    status: Any = "pending"
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: Any = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[Any] = None
    completed_at: Optional[Any] = None
    progress: float = 0.0
    total_files: int = 0
    processed_files: int = 0
    total_documents: int = 0
    processed_documents: int = 0
    flagged_pairs: int = 0
    high_severity_count: int = 0
    error_message: Optional[str] = None
    errors: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    results: Any = field(default_factory=list)

    def __post_init__(self):
        if not self.file_paths and self.document_paths:
            self.file_paths = self.document_paths
        elif not self.document_paths and self.file_paths:
            self.document_paths = self.file_paths

        self.total_files = len(self.file_paths)
        self.total_documents = self.total_files

    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if isinstance(self.started_at, (int, float)) and isinstance(
            self.completed_at, (int, float)
        ):
            return self.completed_at - self.started_at
        return None

    def get_progress_percentage(self) -> float:
        """Get progress as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary."""
        data = asdict(self)
        if isinstance(self.status, Enum):
            data["status"] = self.status.value
        if isinstance(self.priority, Enum):
            data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchJob":
        """Create job from dictionary."""
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = BatchStatus(data["status"])
            except ValueError:
                pass
        if "priority" in data and isinstance(data["priority"], str):
            try:
                data["priority"] = BatchPriority(data["priority"])
            except ValueError:
                pass
        return cls(**data)

    def update_progress(self, processed: int, flagged: int = 0, high: int = 0):
        """Update job progress."""
        self.processed_files = processed
        self.processed_documents = processed
        self.flagged_pairs += flagged
        self.high_severity_count += high
        if self.total_files > 0:
            self.progress = (processed / self.total_files) * 100.0


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 10
    max_workers: int = 4
    use_parallel: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.59
    faiss_top_k: int = 5
    enable_webhook: bool = True
    webhook_threshold: float = 0.90
    output_directory: str = "batch_results"
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    timeout_seconds: int = 300
    ocr_language: str = "eng"
    ocr_dpi: int = 250
    threshold: float = 0.59
    use_hybrid_scoring: bool = False
    cross_lingual_mode: bool = False
    save_progress: bool = True
    progress_file: str = ".cache/batch_progress.json"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchConfig":
        """Create config from dictionary."""
        return cls(**data)


# ============================================================================
# BATCH PROCESSOR
# ============================================================================


class BatchProcessor:
    """Main batch processor with parallelization and progress tracking."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.jobs: dict[str, BatchJob] = {}
        self._jobs: dict[str, BatchJob] = self.jobs
        self._active_job: Optional[str] = None
        self._lock = threading.RLock()
        self._callbacks: list[Callable] = []
        self._progress_callbacks: list[Callable] = self._callbacks
        self._stop_processing = False

        self.metrics = {
            "total_documents": 0,
            "total_time": 0.0,
            "avg_time_per_doc": 0.0,
            "successful": 0,
            "failed": 0,
            "peak_memory_mb": 0,
        }

        Path(".cache").mkdir(parents=True, exist_ok=True)
        logger.info(
            f"BatchProcessor initialized with {self.config.max_workers} workers"
        )

    def register_callback(self, callback: Callable):
        """Register a callback for status updates."""
        self._callbacks.append(callback)

    def register_progress_callback(self, callback: Callable) -> None:
        """Register a callback for progress updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: str, job: BatchJob):
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, job)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_progress(self, job_id: str, progress: float, message: str = "") -> None:
        """Notify all registered callbacks of progress update."""
        for callback in self._callbacks:
            try:
                callback(job_id, progress, message)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

    def create_job(
        self,
        name: str,
        document_paths: list[str],
        priority: BatchPriority = BatchPriority.NORMAL,
        metadata: Optional[dict[str, Any]] = None,
    ) -> BatchJob:
        """Create a new batch processing job."""
        job_id = str(uuid.uuid4())[:12]
        job = BatchJob(
            job_id=job_id,
            name=name,
            file_paths=document_paths,
            document_paths=document_paths,
            priority=priority,
            metadata=metadata or {},
        )
        with self._lock:
            self.jobs[job_id] = job
        logger.info(f"Created batch job {job_id}: {name} ({len(document_paths)} docs)")
        self._notify_callbacks("created", job)
        return job

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def get_active_job(self) -> Optional[BatchJob]:
        """Get the currently active job."""
        with self._lock:
            if self._active_job:
                return self.jobs.get(self._active_job)
            return None

    def list_jobs(
        self,
        status: Optional[BatchStatus] = None,
        priority: Optional[BatchPriority] = None,
    ) -> list[BatchJob]:
        """List jobs with optional filtering."""
        with self._lock:
            jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status or j.status == status.value]
        if priority:
            jobs = [
                j
                for j in jobs
                if j.priority == priority or j.priority == priority.value
            ]
        return sorted(jobs, key=lambda j: str(j.created_at), reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            if job.status in (
                BatchStatus.COMPLETED,
                BatchStatus.CANCELLED,
                "completed",
                "cancelled",
            ):
                return False
            job.status = BatchStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
        self._notify_callbacks("cancelled", job)
        logger.info(f"Cancelled batch job {job_id}")
        return True

    def stop_processing(self) -> None:
        """Stop ongoing processing."""
        self._stop_processing = True

    def _process_single_document(
        self, file_path: str, file_bytes: bytes, config: BatchConfig
    ) -> dict[str, Any]:
        """Process a single document."""
        start_time = time.time()
        result = {
            "file_path": file_path,
            "status": "success",
            "error": None,
            "processing_time": 0.0,
            "chunks": [],
            "embedding": None,
            "word_count": 0,
        }

        try:
            from src.core.document_parser import (
                extract_text,
                prepare_text_for_embedding,
            )
            from src.core.embedding_model import embed_chunks
            from src.core.text_chunking import chunk_documents

            extracted_text = extract_text(
                file_bytes,
                filename=file_path,
                language=config.ocr_language,
                dpi=config.ocr_dpi,
            )

            if not extracted_text:
                result["status"] = "failed"
                result["error"] = "No text extracted"
                return result

            prepared = prepare_text_for_embedding(extracted_text)
            chunks = chunk_documents(
                [prepared],
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            if not chunks:
                result["status"] = "failed"
                result["error"] = "No chunks generated"
                return result

            embeddings = embed_chunks(chunks)

            result["chunks"] = chunks
            result["embedding"] = (
                embeddings.tolist()
                if isinstance(embeddings, np.ndarray)
                else embeddings
            )
            result["word_count"] = len(extracted_text.split())
            result["chunk_count"] = len(chunks)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Failed to process {file_path}: {e}")

        result["processing_time"] = time.time() - start_time
        return result

    def _process_batch(
        self, batch_files: list[tuple[str, bytes]], batch_index: int
    ) -> list[dict[str, Any]]:
        """Process a batch of documents in parallel."""
        results = []

        if self.config.use_parallel and len(batch_files) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for file_path, file_bytes in batch_files:
                    future = executor.submit(
                        self._process_single_document,
                        file_path,
                        file_bytes,
                        self.config,
                    )
                    futures.append((file_path, future))

                for file_path, future in futures:
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        results.append(result)
                    except Exception as e:
                        results.append(
                            {
                                "file_path": file_path,
                                "status": "failed",
                                "error": str(e),
                                "processing_time": 0.0,
                            }
                        )
        else:
            for file_path, file_bytes in batch_files:
                result = self._process_single_document(
                    file_path, file_bytes, self.config
                )
                results.append(result)

        return results

    def process_documents(
        self, file_bytes_dict: dict[str, bytes], job_id: Optional[str] = None
    ) -> BatchJob:
        """Process a batch of documents."""
        if not file_bytes_dict:
            raise ValueError("No documents to process")

        if job_id is None:
            job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        files = list(file_bytes_dict.keys())
        job = BatchJob(
            job_id=job_id,
            name=f"Batch {job_id}",
            file_paths=files,
            document_paths=files,
            total_files=len(files),
            total_documents=len(files),
        )

        with self._lock:
            self.jobs[job_id] = job
            self._active_job = job_id

        batch_size = self.config.batch_size
        file_items = list(file_bytes_dict.items())
        batches = [
            file_items[i : i + batch_size]
            for i in range(0, len(file_items), batch_size)
        ]

        logger.info(f"Processing {len(file_items)} files in {len(batches)} batches")

        start_time = time.time()
        job.started_at = start_time

        for batch_index, batch in enumerate(batches):
            if self._stop_processing:
                logger.warning(f"Processing stopped for job {job_id}")
                break

            job.status = "processing"
            job.progress = (batch_index / len(batches)) * 100

            self._notify_progress(
                job_id,
                job.progress,
                f"Processing batch {batch_index + 1}/{len(batches)}",
            )

            batch_results = self._process_batch(batch, batch_index)

            with self._lock:
                for result in batch_results:
                    if result["status"] == "success":
                        if isinstance(job.results, list):
                            job.results.append(result)
                        job.processed_files += 1
                        job.processed_documents += 1
                    else:
                        job.errors.append(
                            {
                                "file": result["file_path"],
                                "error": result.get("error", "Unknown error"),
                            }
                        )

            self.metrics["successful"] = job.processed_files
            self.metrics["failed"] = len(job.errors)

            try:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                if memory_mb > self.metrics["peak_memory_mb"]:
                    self.metrics["peak_memory_mb"] = memory_mb
            except Exception:
                pass

        job.completed_at = time.time()
        job.status = "completed"
        job.progress = 100.0

        self.metrics["total_documents"] += job.total_files
        self.metrics["total_time"] += job.get_duration() or 0
        if job.processed_files > 0:
            self.metrics["avg_time_per_doc"] = (
                self.metrics["total_time"] / self.metrics["total_documents"]
            )

        self._notify_progress(job_id, 100.0, "Processing complete")

        if self.config.save_progress:
            self._save_progress(job)

        return job

    def _save_progress(self, job: BatchJob) -> None:
        """Save job progress to file."""
        try:
            data = {
                "job_id": job.job_id,
                "status": (
                    job.status if isinstance(job.status, str) else job.status.value
                ),
                "processed_files": job.processed_files,
                "total_files": job.total_files,
                "errors": job.errors,
                "results_count": (
                    len(job.results)
                    if isinstance(job.results, list)
                    else len(job.results.keys())
                ),
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
            }
            with open(self.config.progress_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get processing metrics."""
        return {
            **self.metrics,
            "active_job": self._active_job,
            "total_jobs": len(self.jobs),
            "config": self.config.to_dict(),
        }

    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on system resources."""
        try:
            cpu_count = multiprocessing.cpu_count()
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)

            if available_mb > 4096 and cpu_count > 4:
                return 20
            elif available_mb > 2048 and cpu_count > 2:
                return 10
            else:
                return 5
        except Exception:
            return 10


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_processor: Optional[BatchProcessor] = None
_processor_lock = threading.Lock()


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    global _processor
    with _processor_lock:
        if _processor is None:
            _processor = BatchProcessor()
        return _processor
