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
Batch Processor UI Components.

Provides UI elements for batch processing with progress tracking.
"""

import time  # noqa: F401
from datetime import datetime
from typing import Any, Dict, List  # noqa: F401

import pandas as pd
import streamlit as st

from src.core.batch_processor import BatchConfig  # noqa: F401
from src.core.batch_processor import BatchProcessor  # noqa: F401
from src.core.batch_processor import BatchJob, get_batch_processor


def render_batch_processor_ui() -> None:
    """Render batch processor UI."""
    st.markdown("### 🚀 Batch Document Processor")

    processor = get_batch_processor()

    # Configuration
    with st.expander("⚙️ Batch Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            batch_size = st.number_input(
                "📦 Batch Size",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of documents per batch",
            )

        with col2:
            max_workers = st.number_input(
                "👷 Workers",
                min_value=1,
                max_value=8,
                value=4,
                help="Number of parallel workers",
            )

        with col3:
            recommended = processor.get_recommended_batch_size()
            st.metric(
                "📊 Recommended Batch Size",
                recommended,
                help="Based on system resources",
            )

        col1, col2 = st.columns(2)
        with col1:
            use_parallel = st.checkbox(
                "⚡ Enable Parallel Processing",
                value=True,
                help="Process documents in parallel",
            )
        with col2:
            save_progress = st.checkbox(
                "💾 Save Progress", value=True, help="Save progress to file"
            )

    # File upload
    uploaded_files = st.file_uploader(
        "📂 Upload Documents",
        type=["pdf", "docx", "txt", "md", "markdown", "mdown"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if uploaded_files:
        file_count = len(uploaded_files)
        total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
        st.info(f"📁 {file_count} files ({total_size:.1f} MB)")

        # Show file list
        with st.expander("📋 File List", expanded=False):
            df = pd.DataFrame(
                [
                    {"Filename": f.name, "Size": f"{f.size / 1024:.1f} KB"}
                    for f in uploaded_files
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Process button
        if st.button(
            "🚀 Start Batch Processing", type="primary", use_container_width=True
        ):
            # Prepare file bytes dict
            file_bytes_dict = {f.name: f.getvalue() for f in uploaded_files}

            # Update config
            processor.config.batch_size = batch_size
            processor.config.max_workers = max_workers
            processor.config.use_parallel = use_parallel
            processor.config.save_progress = save_progress

            # Process
            with st.spinner(f"Processing {len(file_bytes_dict)} documents..."):
                job = processor.process_documents(file_bytes_dict)

                if job:
                    st.success(
                        f"✅ Processing complete! {job.processed_files}/{job.total_files} successful"
                    )
                    st.session_state["batch_job_id"] = job.job_id
                    st.rerun()

    # Show active job
    active_job = processor.get_active_job()
    if active_job and active_job.status != "completed":
        render_job_progress(active_job)


def render_job_progress(job: BatchJob) -> None:
    """Render job progress."""
    st.subheader(f"📊 Job Progress: {job.job_id}")

    # Progress bar
    progress = job.get_progress_percentage()
    st.progress(progress / 100)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Processed", f"{job.processed_files}/{job.total_files}")
    with col2:
        st.metric("✅ Success", len(job.results))
    with col3:
        st.metric("❌ Failed", len(job.errors))
    with col4:
        duration = job.get_duration()
        if duration:
            st.metric("⏱️ Duration", f"{duration:.1f}s")
        else:
            st.metric("⏱️ Duration", "Processing...")

    # Status
    status_colors = {
        "pending": "🟡",
        "processing": "🔵",
        "completed": "🟢",
        "failed": "🔴",
    }
    st.markdown(
        f"**Status:** {status_colors.get(job.status, '⚪')} {job.status.upper()}"
    )

    # Show errors if any
    if job.errors:
        with st.expander(f"❌ Errors ({len(job.errors)})", expanded=False):
            for error in job.errors:
                st.error(
                    f"**{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}"
                )

    # Metrics chart
    if job.processed_files > 0:
        st.subheader("📈 Processing Metrics")

        # Success rate
        success_rate = (
            (len(job.results) / job.processed_files) * 100
            if job.processed_files > 0
            else 0
        )
        st.metric("✅ Success Rate", f"{success_rate:.1f}%")
        st.progress(success_rate / 100)


def render_batch_analytics() -> None:
    """Render batch processing analytics."""
    st.markdown("### 📊 Batch Processing Analytics")

    processor = get_batch_processor()
    metrics = processor.get_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Total Documents", metrics["total_documents"])
    with col2:
        st.metric("✅ Successful", metrics["successful"])
    with col3:
        st.metric("❌ Failed", metrics["failed"])
    with col4:
        st.metric("⏱️ Total Time", f"{metrics['total_time']:.1f}s")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚡ Avg Time/Doc", f"{metrics['avg_time_per_doc']:.2f}s")
    with col2:
        st.metric("💾 Peak Memory", f"{metrics['peak_memory_mb']:.0f} MB")

    st.divider()

    # Show configuration
    with st.expander("⚙️ Current Configuration", expanded=False):
        config = metrics.get("config", {})
        for key, value in config.items():
            st.caption(f"**{key}:** {value}")

    # Show all jobs
    with st.expander("📋 Job History", expanded=False):
        # Get jobs from processor
        jobs = getattr(processor, "_jobs", {})
        if jobs:
            job_data = []
            for job_id, job in jobs.items():
                job_data.append(
                    {
                        "Job ID": job_id[:12],
                        "Status": job.status,
                        "Files": f"{job.processed_files}/{job.total_files}",
                        "Duration": (
                            f"{job.get_duration():.1f}s" if job.get_duration() else "-"
                        ),
                        "Created": datetime.fromtimestamp(job.created_at).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    }
                )
            df = pd.DataFrame(job_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No jobs found")


def render_batch_scheduler_ui() -> None:
    """Render batch scheduler UI."""
    st.markdown("### ⏰ Batch Scheduler")

    st.info("Schedule batch processing for later execution.")

    col1, col2 = st.columns(2)

    with col1:
        schedule_time = st.time_input("⏱️ Schedule Time", value=datetime.now().time())

    with col2:
        schedule_date = st.date_input("📅 Schedule Date", value=datetime.now().date())

    if st.button("📅 Schedule Job", use_container_width=True):
        st.success(f"✅ Job scheduled for {schedule_date} at {schedule_time}")
