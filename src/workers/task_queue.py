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
src/workers/task_queue.py
-------------------------
Thread-safe producer/consumer queue with retry logic and dead-letter
handling (Issue #3146).

The queue wraps ``src.db.task_db`` with an in-process ``queue.Queue``
so multiple worker threads can pull jobs concurrently. When a job fails
it is retried up to ``max_retries`` times; once retries are exhausted
the job is moved to DEAD_LETTER status.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

from src.db import task_db

logger = logging.getLogger(__name__)

# ── Sentinel ───────────────────────────────────────────────────

_SHUTDOWN = object()


class TaskQueue:
    """Thread-safe in-memory + SQLite-backed task queue.

    Producers call :meth:`enqueue` to persist a job to SQLite and push its
    ID onto the in-memory queue. Consumers (worker threads) call
    :meth:`dequeue` to atomically claim the next PENDING job from SQLite
    (via ``task_db.claim_next_job``).

    The in-memory queue is only a wake-up signal — the source of truth
    is always the SQLite ``task_jobs`` table. This means the queue
    survives process restarts: on startup, any PROCESSING jobs whose
    worker died can be re-queued by calling :meth:`requeue_stale_processing`.
    """

    def __init__(
        self,
        worker_id: str = "queue-orchestrator",
        max_retries: int = 3,
        poll_interval: float = 0.5,
        db_path: str | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._max_retries = max_retries
        self._poll_interval = poll_interval
        self.db_path = db_path
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._lock = threading.Lock()
        self._running = False
        if db_path is not None:
            task_db.initialize_task_db(Path(db_path))

    # ── Producer ──────────────────────────────────────────────

    def enqueue(
        self,
        payload: dict[str, Any],
        *,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Persist a new job to SQLite and signal the in-memory queue."""
        retries = max_retries if max_retries is not None else self._max_retries
        job = task_db.create_job(payload, max_retries=retries, db_path=self.db_path)
        self._queue.put(job["id"])
        logger.info(
            "Enqueued job %s (payload keys: %s)", job["id"], list(payload.keys())
        )
        return job

    def submit_batch_scan(
        self,
        document_ids: list[str],
        user_id: str,
        priority: int = 0,
    ) -> str:
        """Submit a new batch scanning job."""
        payload = {
            "document_ids": document_ids,
            "user_id": user_id,
            "priority": priority,
            "type": "batch_scan",
        }
        job = self.enqueue(payload)
        return job["id"]

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the current status and details of a job."""
        return task_db.get_job(job_id, db_path=self.db_path)

    # ── Consumer ──────────────────────────────────────────────

    def dequeue(self, timeout: float = 30.0) -> dict[str, Any] | None:
        """Block up to ``timeout`` seconds for the next job.

        Returns the job dict (status=PROCESSING) or ``None`` on timeout.
        """
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            # In-memory queue empty — poll SQLite directly (catches jobs
            # that were enqueued by another process / a previous run).
            return task_db.claim_next_job(self._worker_id, db_path=self.db_path)

        if item is _SHUTDOWN:
            return None
        if isinstance(item, str):
            # We got a job_id from the in-memory queue, but we still need
            # to atomically claim it from SQLite (another worker might
            # have grabbed it first).
            job = task_db.claim_next_job(self._worker_id, db_path=self.db_path)
            if job is not None:
                return job
            # The job was already claimed by another worker; poll SQLite
            # for the next pending one.
            return task_db.claim_next_job(self._worker_id, db_path=self.db_path)
        return None

    # ── Result handlers ───────────────────────────────────────

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        task_db.mark_completed(job_id, result, db_path=self.db_path)
        logger.info("Job %s completed", job_id)

    def fail(self, job_id: str, error: str) -> None:
        """Mark a job as failed. Retries if attempts remain."""
        task_db.mark_failed(job_id, error, db_path=self.db_path)
        logger.warning("Job %s failed: %s", job_id, error[:200])

    def dead_letter(self, job_id: str, error: str) -> None:
        task_db.mark_dead_letter(job_id, error, db_path=self.db_path)
        logger.error("Job %s moved to dead-letter: %s", job_id, error[:200])

    # ── Lifecycle ──────────────────────────────────────────────

    def signal_shutdown(self) -> None:
        self._queue.put(_SHUTDOWN)

    def requeue_stale_processing(self) -> int:
        """Re-queue any PROCESSING jobs (from a crashed previous run).

        Returns the number of jobs re-queued.
        """
        stale = task_db.list_jobs(status="PROCESSING", limit=1000, db_path=self.db_path)
        for job in stale:
            task_db.mark_failed(
                job["id"],
                "Re-queued: stale PROCESSING job (worker crash)",
                db_path=self.db_path,
            )
        return len(stale)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.signal_shutdown()


# ── Module-level singleton ─────────────────────────────────────

_default_queue: TaskQueue | None = None
_singleton_lock = threading.Lock()


def get_default_queue() -> TaskQueue:
    global _default_queue
    if _default_queue is None:
        with _singleton_lock:
            if _default_queue is None:
                _default_queue = TaskQueue()
    return _default_queue
