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
Enhanced Batch Processing with Smart Queue Management

Features:
- Smart priority queue for document processing
- Parallel processing with adaptive worker pool
- Real-time progress tracking
- Pause/resume capability
- Automatic error recovery with retry
- Batch analytics and performance metrics
- Resource optimization
- Scheduled batch jobs
"""

import asyncio  # noqa: F401
import concurrent.futures
import hashlib  # noqa: F401
import json
import queue
import threading
import time
from collections import defaultdict, deque  # noqa: F401
from dataclasses import asdict, dataclass, field  # noqa: F401
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # noqa: F401

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class JobPriority(Enum):
    """Job priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class JobStatus(Enum):
    """Job status states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class BatchJob:
    """Batch job definition."""

    id: str
    name: str
    priority: JobPriority
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    data: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    progress: float = 0.0
    logs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Batch processing result."""

    job_id: str
    status: JobStatus
    processed_count: int
    success_count: int
    failed_count: int
    total_time: float
    average_time: float
    errors: list[str] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerStats:
    """Worker statistics."""

    worker_id: int
    active: bool
    current_job: Optional[str] = None
    jobs_processed: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    success_rate: float = 1.0


# ==============================================================================
# ENHANCED BATCH PROCESSOR
# ==============================================================================


class EnhancedBatchProcessor:
    """
    Enhanced batch processor with smart queue management and real-time tracking.
    """

    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.max_workers = max_workers
        self.queue_size = queue_size

        # Queues
        self.job_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self.completed_jobs: list[BatchJob] = []
        self.active_jobs: dict[str, BatchJob] = {}

        # Worker management
        self.workers: list[dict] = []
        self.worker_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.is_running = False
        self.is_paused = False
        self.pause_event = threading.Event()
        self.pause_event.set()

        # Tracking
        self.job_counter = 0
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "success_rate": 1.0,
        }

        # Callbacks
        self.on_job_complete: Optional[Callable] = None
        self.on_progress_update: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # Initialize
        self._initialize_workers()

    def _initialize_workers(self):
        """Initialize worker pool."""
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.workers = [
            {
                "id": i,
                "active": False,
                "current_job": None,
                "jobs_processed": 0,
                "total_time": 0.0,
            }
            for i in range(self.max_workers)
        ]

    def add_job(
        self, job_data: dict[str, Any], priority: JobPriority = JobPriority.NORMAL
    ) -> str:
        """
        Add a job to the queue.

        Args:
            job_data: Job data to process
            priority: Job priority

        Returns:
            str: Job ID
        """
        job_id = f"job_{int(time.time())}_{self.job_counter}"
        self.job_counter += 1

        job = BatchJob(
            id=job_id,
            name=job_data.get("name", f"Job {self.job_counter}"),
            priority=priority,
            status=JobStatus.PENDING,
            created_at=time.time(),
            data=job_data,
        )

        # Add to queue
        self.job_queue.put((priority.value, job_id, job))
        self.stats["total_jobs"] += 1

        # Start processing if not running
        if not self.is_running:
            self.start_processing()

        return job_id

    def add_batch(
        self, jobs: list[dict[str, Any]], priority: JobPriority = JobPriority.NORMAL
    ) -> list[str]:
        """
        Add multiple jobs to the queue.

        Args:
            jobs: List of job data
            priority: Priority for all jobs

        Returns:
            List[str]: Job IDs
        """
        job_ids = []
        for job_data in jobs:
            job_id = self.add_job(job_data, priority)
            job_ids.append(job_id)
        return job_ids

    def start_processing(self):
        """Start processing jobs."""
        if self.is_running:
            return

        self.is_running = True
        self.is_paused = False
        self.pause_event.set()

        # Start worker threads
        for i in range(self.max_workers):
            threading.Thread(target=self._worker_loop, args=(i,), daemon=True).start()

    def pause_processing(self):
        """Pause processing."""
        if not self.is_running:
            return

        self.is_paused = True
        self.pause_event.clear()
        st.toast("⏸️ Processing paused")

    def resume_processing(self):
        """Resume processing."""
        if not self.is_running:
            return

        self.is_paused = False
        self.pause_event.set()
        st.toast("▶️ Processing resumed")

    def stop_processing(self):
        """Stop processing."""
        self.is_running = False
        self.is_paused = False
        self.pause_event.set()

        if self.worker_pool:
            self.worker_pool.shutdown(wait=False)

        st.toast("⏹️ Processing stopped")

    def cancel_job(self, job_id: str):
        """Cancel a specific job."""
        # Remove from queue
        # This is simplified - real implementation would need to scan queue
        job = self.active_jobs.get(job_id)
        if job:
            job.status = JobStatus.CANCELLED
            self.active_jobs.pop(job_id, None)

    def _worker_loop(self, worker_id: int):
        """Worker thread loop."""
        worker = self.workers[worker_id]

        while self.is_running:
            try:
                # Check if paused
                self.pause_event.wait()

                # Get job from queue with timeout
                try:
                    priority, job_id, job = self.job_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Update worker status
                worker["active"] = True
                worker["current_job"] = job_id
                self.active_jobs[job_id] = job

                # Process job
                start_time = time.time()
                job.status = JobStatus.PROCESSING
                job.started_at = start_time

                try:
                    # Execute job
                    result = self._process_job(job)
                    job.result = result
                    job.status = JobStatus.COMPLETED
                    job.completed_at = time.time()

                    # Update stats
                    worker["jobs_processed"] += 1
                    self.stats["completed_jobs"] += 1

                    # Callback
                    if self.on_job_complete:
                        self.on_job_complete(job)

                except Exception as e:
                    # Handle error
                    job.error = str(e)

                    if job.retry_count < job.max_retries:
                        # Retry
                        job.retry_count += 1
                        job.status = JobStatus.RETRYING
                        job.logs.append(f"Retry {job.retry_count}: {str(e)}")

                        # Re-add to queue with same priority
                        self.job_queue.put((job.priority.value, job_id, job))
                    else:
                        # Failed
                        job.status = JobStatus.FAILED
                        job.completed_at = time.time()
                        self.stats["failed_jobs"] += 1

                        if self.on_error:
                            self.on_error(job, e)

                finally:
                    # Update worker
                    elapsed = time.time() - start_time
                    worker["total_time"] += elapsed
                    worker["current_job"] = None
                    worker["active"] = False

                    # Remove from active jobs
                    self.active_jobs.pop(job_id, None)

                    # Mark queue task done
                    self.job_queue.task_done()

            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                time.sleep(1)

    def _process_job(self, job: BatchJob) -> Any:
        """
        Process a single job.

        Args:
            job: Job to process

        Returns:
            Any: Job result
        """
        # Simulate processing with progress updates
        total_steps = 10

        for i in range(total_steps):
            # Check if cancelled
            if job.status == JobStatus.CANCELLED:
                raise Exception("Job cancelled")

            # Update progress
            job.progress = (i + 1) / total_steps

            # Simulate work
            time.sleep(0.5)

            # Progress callback
            if self.on_progress_update:
                self.on_progress_update(job)

        # Return result based on job data
        job_data = job.data

        # Example processing - replace with actual processing logic
        if "documents" in job_data:
            # Process documents
            result = {
                "processed": len(job_data["documents"]),
                "success": True,
                "timestamp": time.time(),
            }
            return result
        else:
            return {"status": "success", "data": job_data}

    def get_job_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get job status."""
        # Check active jobs
        job = self.active_jobs.get(job_id)
        if job:
            return self._job_to_dict(job)

        # Check completed jobs
        for completed_job in self.completed_jobs:
            if completed_job.id == job_id:
                return self._job_to_dict(completed_job)

        return None

    def get_all_jobs(self) -> list[dict[str, Any]]:
        """Get all jobs."""
        jobs = []

        # Active jobs
        for job in self.active_jobs.values():
            jobs.append(self._job_to_dict(job))

        # Completed jobs (last 100)
        for job in self.completed_jobs[-100:]:
            jobs.append(self._job_to_dict(job))

        return jobs

    def _job_to_dict(self, job: BatchJob) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": job.id,
            "name": job.name,
            "priority": job.priority.value,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "progress": job.progress,
            "retry_count": job.retry_count,
            "error": job.error,
            "logs": job.logs[-10:],  # Last 10 logs
        }

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        total = self.stats["total_jobs"]
        completed = self.stats["completed_jobs"]
        failed = self.stats["failed_jobs"]

        return {
            "total_jobs": total,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "pending_jobs": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "success_rate": completed / total if total > 0 else 1.0,
            "progress": completed / total if total > 0 else 0.0,
            "workers": [
                {
                    "id": w["id"],
                    "active": w["active"],
                    "current_job": w["current_job"],
                    "jobs_processed": w["jobs_processed"],
                }
                for w in self.workers
            ],
            "is_running": self.is_running,
            "is_paused": self.is_paused,
        }

    def get_batch_analytics(self) -> dict[str, Any]:
        """Get batch processing analytics."""
        if not self.completed_jobs:
            return {
                "total_batches": 0,
                "avg_processing_time": 0,
                "success_rate": 0,
                "peak_throughput": 0,
                "error_rate": 0,
            }

        # Calculate metrics
        total_time = sum(
            (j.completed_at or 0) - (j.started_at or 0) for j in self.completed_jobs
        )

        success_rate = len(
            [j for j in self.completed_jobs if j.status == JobStatus.COMPLETED]
        ) / len(self.completed_jobs)

        return {
            "total_batches": len(self.completed_jobs),
            "avg_processing_time": (
                total_time / len(self.completed_jobs) if self.completed_jobs else 0
            ),
            "success_rate": success_rate,
            "error_rate": 1 - success_rate,
            "peak_throughput": self._calculate_peak_throughput(),
            "total_retries": sum(j.retry_count for j in self.completed_jobs),
        }

    def _calculate_peak_throughput(self) -> float:
        """Calculate peak throughput (jobs per minute)."""
        if not self.completed_jobs:
            return 0

        # Look at last 5 minutes
        cutoff = time.time() - 300
        recent = [j for j in self.completed_jobs if (j.completed_at or 0) > cutoff]

        if not recent:
            return 0

        return len(recent) / 5  # Jobs per minute

    def export_results(self) -> dict[str, Any]:
        """Export processing results."""
        return {
            "stats": self.get_stats(),
            "analytics": self.get_batch_analytics(),
            "jobs": self.get_all_jobs(),
            "exported_at": datetime.now().isoformat(),
        }


# ==============================================================================
# BATCH SCHEDULER
# ==============================================================================


class BatchScheduler:
    """Schedule batch processing jobs."""

    def __init__(self, processor: EnhancedBatchProcessor):
        self.processor = processor
        self.schedules: list[dict] = []
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self._load_schedules()

    def _load_schedules(self):
        """Load schedules from storage."""
        try:
            schedule_path = (
                Path(st.session_state.get("data_dir", ".")) / "batch_schedules.json"
            )
            if schedule_path.exists():
                with open(schedule_path, "r") as f:
                    self.schedules = json.load(f)
        except Exception as e:
            print(f"Error loading schedules: {e}")

    def _save_schedules(self):
        """Save schedules to storage."""
        try:
            schedule_path = (
                Path(st.session_state.get("data_dir", ".")) / "batch_schedules.json"
            )
            schedule_path.parent.mkdir(parents=True, exist_ok=True)
            with open(schedule_path, "w") as f:
                json.dump(self.schedules, f, indent=2)
        except Exception as e:
            print(f"Error saving schedules: {e}")

    def add_schedule(self, name: str, cron: str, config: dict[str, Any]):
        """
        Add a scheduled batch job.

        Args:
            name: Schedule name
            cron: Cron expression (simplified)
            config: Job configuration
        """
        schedule = {
            "id": f"schedule_{int(time.time())}",
            "name": name,
            "cron": cron,
            "config": config,
            "enabled": True,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "next_run": self._calculate_next_run(cron),
        }
        self.schedules.append(schedule)
        self._save_schedules()

        # Start scheduler if not running
        if not self.is_running:
            self.start()

    def _calculate_next_run(self, cron: str) -> str:
        """Calculate next run time from cron expression."""
        # Simplified - would use cron parser in production
        now = datetime.now()

        if cron == "* * * * *":  # Every minute
            next_run = now + timedelta(minutes=1)
        elif cron == "0 * * * *":  # Every hour
            next_run = now + timedelta(hours=1)
        elif cron == "0 0 * * *":  # Daily
            next_run = now + timedelta(days=1)
        elif cron == "0 0 * * 0":  # Weekly
            next_run = now + timedelta(weeks=1)
        elif cron == "0 0 1 * *":  # Monthly
            next_run = now + timedelta(days=30)
        else:
            next_run = now + timedelta(minutes=5)

        return next_run.isoformat()

    def start(self):
        """Start scheduler."""
        if self.is_running:
            return

        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self.scheduler_thread.start()

    def stop(self):
        """Stop scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=2)

    def _scheduler_loop(self):
        """Scheduler main loop."""
        while self.is_running:
            try:
                now = datetime.now()

                for schedule in self.schedules:
                    if not schedule.get("enabled", True):
                        continue

                    next_run = datetime.fromisoformat(schedule["next_run"])

                    if now >= next_run:
                        # Execute schedule
                        self._execute_schedule(schedule)

                        # Update next run
                        schedule["last_run"] = now.isoformat()
                        schedule["next_run"] = self._calculate_next_run(
                            schedule["cron"]
                        )
                        self._save_schedules()

                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(60)

    def _execute_schedule(self, schedule: dict):
        """Execute a scheduled job."""
        config = schedule.get("config", {})

        # Add job to processor
        self.processor.add_job(
            job_data=config.get("data", {}),
            priority=JobPriority(config.get("priority", 2)),
        )


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_batch_processor_ui():
    """Render batch processor UI."""
    st.subheader("🚀 Batch Processing Dashboard")

    # Initialize processor
    if "batch_processor" not in st.session_state:
        st.session_state.batch_processor = EnhancedBatchProcessor(max_workers=4)

    processor = st.session_state.batch_processor

    # Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not processor.is_running:
            if st.button("▶️ Start", use_container_width=True):
                processor.start_processing()
                st.rerun()
        else:
            if st.button("⏹️ Stop", use_container_width=True):
                processor.stop_processing()
                st.rerun()

    with col2:
        if processor.is_running and not processor.is_paused:
            if st.button("⏸️ Pause", use_container_width=True):
                processor.pause_processing()
                st.rerun()
        elif processor.is_running and processor.is_paused:
            if st.button("▶️ Resume", use_container_width=True):
                processor.resume_processing()
                st.rerun()

    with col3:
        if st.button("📊 Stats", use_container_width=True):
            st.session_state.show_batch_stats = not st.session_state.get(
                "show_batch_stats", False
            )

    with col4:
        if st.button("🗑️ Clear Completed", use_container_width=True):
            processor.completed_jobs = []
            st.success("✅ Completed jobs cleared")
            st.rerun()

    # Add job
    with st.expander("📝 Add Batch Job", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            job_name = st.text_input("Job Name", "Batch Processing Job")
            priority = st.selectbox(
                "Priority", ["CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND"], index=2
            )
        with col2:
            doc_count = st.number_input("Number of Documents", 1, 1000, 10)
            max_retries = st.number_input("Max Retries", 0, 5, 3)

        if st.button("➕ Add Job", use_container_width=True):
            # Create job data
            job_data = {
                "name": job_name,
                "documents": [f"doc_{i}" for i in range(doc_count)],
                "max_retries": max_retries,
            }

            priority_map = {
                "CRITICAL": JobPriority.CRITICAL,
                "HIGH": JobPriority.HIGH,
                "NORMAL": JobPriority.NORMAL,
                "LOW": JobPriority.LOW,
                "BACKGROUND": JobPriority.BACKGROUND,
            }

            job_id = processor.add_job(job_data, priority_map[priority])
            st.success(f"✅ Job added: {job_id}")
            st.rerun()

    # Statistics
    if st.session_state.get("show_batch_stats", False):
        stats = processor.get_stats()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Jobs", stats["total_jobs"])
        col2.metric("Completed", stats["completed_jobs"])
        col3.metric("Failed", stats["failed_jobs"])
        col4.metric("Pending", stats["pending_jobs"])
        col5.metric("Success Rate", f"{stats['success_rate']:.1%}")

        # Progress bar
        st.progress(
            stats["progress"], text=f"Overall Progress: {stats['progress']:.1%}"
        )

        # Worker status
        if stats["workers"]:
            st.markdown("#### 🧑‍💻 Worker Status")
            worker_cols = st.columns(len(stats["workers"]))
            for col, worker in zip(worker_cols, stats["workers"]):
                status = "🟢" if worker["active"] else "⚪"
                col.markdown(f"{status} Worker #{worker['id']}")
                col.caption(f"Jobs: {worker['jobs_processed']}")
                if worker["current_job"]:
                    col.caption(f"Current: {worker['current_job'][:8]}...")

    # Active Jobs
    active_jobs = processor.active_jobs
    if active_jobs:
        st.markdown("#### 🔄 Active Jobs")

        for job_id, job in list(active_jobs.items())[:10]:
            with st.expander(f"Job: {job.name} ({job.status.value})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"ID: {job_id}")
                    st.caption(f"Priority: {job.priority.name}")
                    st.caption(f"Retries: {job.retry_count}/{job.max_retries}")
                with col2:
                    st.progress(job.progress, text=f"Progress: {job.progress:.1%}")
                    if job.started_at:
                        elapsed = time.time() - job.started_at
                        st.caption(f"Elapsed: {elapsed:.1f}s")

                if job.error:
                    st.error(f"Error: {job.error}")

                if job.logs:
                    st.caption("Recent Logs:")
                    for log in job.logs[-3:]:
                        st.text(f"• {log}")

                if st.button(f"❌ Cancel", key=f"cancel_{job_id}"):  # noqa: F541
                    processor.cancel_job(job_id)
                    st.rerun()

    # Completed Jobs
    if processor.completed_jobs:
        st.markdown("#### ✅ Completed Jobs")

        completed_df = pd.DataFrame(
            [
                {
                    "ID": j.id[:8],
                    "Name": j.name[:30],
                    "Status": j.status.value,
                    "Progress": f"{j.progress:.1%}",
                    "Time": f"{(j.completed_at or 0) - (j.started_at or 0):.1f}s",
                }
                for j in processor.completed_jobs[-20:]
            ]
        )

        st.dataframe(completed_df, use_container_width=True, hide_index=True)


def render_batch_analytics():
    """Render batch analytics dashboard."""
    st.subheader("📊 Batch Analytics")

    if "batch_processor" not in st.session_state:
        st.info("No batch data available")
        return

    processor = st.session_state.batch_processor
    analytics = processor.get_batch_analytics()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Batches", analytics["total_batches"])
    col2.metric("Success Rate", f"{analytics['success_rate']:.1%}")
    col3.metric("Avg Time", f"{analytics['avg_processing_time']:.1f}s")
    col4.metric("Peak Throughput", f"{analytics['peak_throughput']:.1f}/min")

    # Charts
    if processor.completed_jobs:
        # Create processing time chart
        jobs = processor.completed_jobs[-50:]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Processing Times",
                "Success/Fail",
                "Job Duration Distribution",
                "Retry Counts",
            ),
        )

        # Processing times
        times = [(j.completed_at or 0) - (j.started_at or 0) for j in jobs]
        fig.add_trace(go.Bar(y=times, name="Time", marker_color="blue"), row=1, col=1)

        # Success/Fail pie chart
        success_count = len([j for j in jobs if j.status == JobStatus.COMPLETED])
        fail_count = len([j for j in jobs if j.status == JobStatus.FAILED])
        fig.add_trace(
            go.Pie(labels=["Success", "Failed"], values=[success_count, fail_count]),
            row=1,
            col=2,
        )

        # Duration distribution
        durations = [t for t in times if t > 0]
        fig.add_trace(go.Histogram(x=durations, nbinsx=20), row=2, col=1)

        # Retry counts
        retries = [j.retry_count for j in jobs]
        fig.add_trace(
            go.Bar(
                x=list(range(max(retries) + 1)),
                y=[retries.count(i) for i in range(max(retries) + 1)],
            ),
            row=2,
            col=2,
        )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_batch_scheduler_ui():
    """Render batch scheduler UI."""
    st.subheader("🗓️ Batch Scheduler")

    if "batch_processor" not in st.session_state:
        st.warning("Please initialize batch processor first")
        return

    processor = st.session_state.batch_processor

    # Initialize scheduler
    if "batch_scheduler" not in st.session_state:
        st.session_state.batch_scheduler = BatchScheduler(processor)

    scheduler = st.session_state.batch_scheduler

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if not scheduler.is_running:
            if st.button("▶️ Start Scheduler", use_container_width=True):
                scheduler.start()
                st.rerun()
        else:
            if st.button("⏹️ Stop Scheduler", use_container_width=True):
                scheduler.stop()
                st.rerun()

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Add schedule
    with st.expander("📝 Add Schedule", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            schedule_name = st.text_input("Schedule Name", "Daily Batch")
            cron = st.selectbox(
                "Frequency",
                ["* * * * *", "0 * * * *", "0 0 * * *", "0 0 * * 0", "0 0 1 * *"],
                format_func=lambda x: {
                    "* * * * *": "Every Minute",
                    "0 * * * *": "Every Hour",
                    "0 0 * * *": "Daily",
                    "0 0 * * 0": "Weekly",
                    "0 0 1 * *": "Monthly",
                }.get(x, x),
            )
        with col2:
            priority = st.selectbox("Priority", ["NORMAL", "HIGH", "LOW"])
            doc_count = st.number_input("Documents per batch", 1, 1000, 50)
        with col3:
            enabled = st.checkbox("Enabled", value=True)  # noqa: F841
            if st.button("📅 Create Schedule", use_container_width=True):
                config = {
                    "data": {
                        "documents": [f"scheduled_doc_{i}" for i in range(doc_count)]
                    },
                    "priority": priority,
                }
                scheduler.add_schedule(schedule_name, cron, config)
                st.success("✅ Schedule created")
                st.rerun()

    # Display schedules
    if scheduler.schedules:
        st.markdown("#### 📋 Active Schedules")

        df = pd.DataFrame(
            [
                {
                    "Name": s["name"],
                    "Frequency": s["cron"],
                    "Next Run": s["next_run"][:16] if s["next_run"] else "N/A",
                    "Last Run": s["last_run"][:16] if s["last_run"] else "Never",
                    "Enabled": "✅" if s["enabled"] else "❌",
                }
                for s in scheduler.schedules
            ]
        )

        st.dataframe(df, use_container_width=True, hide_index=True)


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_batch_processor():
    """Initialize batch processor."""
    if "batch_processor_initialized" not in st.session_state:
        st.session_state.batch_processor_initialized = True

        # Create processor
        processor = EnhancedBatchProcessor(max_workers=4)
        st.session_state.batch_processor = processor

        # Create scheduler
        scheduler = BatchScheduler(processor)
        st.session_state.batch_scheduler = scheduler
