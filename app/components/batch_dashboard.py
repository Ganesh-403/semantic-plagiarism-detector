"""
Batch Processing Dashboard Component for Streamlit.

Provides a comprehensive UI for managing and monitoring batch
plagiarism detection jobs with real-time progress tracking.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.core.batch_processor import (
    BatchProcessor,
    BatchJob,
    BatchStatus,
    BatchPriority,
    BatchConfig,
)
from src.core.batch_history import BatchHistory


def render_batch_dashboard(processor: BatchProcessor, history: BatchHistory):
    """
    Render the main batch processing dashboard.

    Args:
        processor: BatchProcessor instance
        history: BatchHistory instance
    """
    st.title("📊 Batch Processing Dashboard")
    st.markdown("Manage and monitor large-scale plagiarism detection jobs.")

    tab_create, tab_active, tab_history, tab_analytics = st.tabs(
        ["➕ Create Job", "⚡ Active Jobs", "📜 History", "📈 Analytics"]
    )

    with tab_create:
        _render_create_job(processor)

    with tab_active:
        _render_active_jobs(processor)

    with tab_history:
        _render_history(history)

    with tab_analytics:
        _render_analytics(history)


def _render_create_job(processor: BatchProcessor):
    """Render job creation form."""
    st.subheader("Create New Batch Job")

    with st.form("batch_create"):
        col1, col2 = st.columns(2)
        with col1:
            job_name = st.text_input(
                "Job Name", placeholder="e.g., Q3 Assignment Check"
            )
            priority = st.selectbox("Priority", ["low", "normal", "high", "urgent"])
        with col2:
            threshold = st.slider("Similarity Threshold", 0.50, 0.99, 0.59, 0.01)
            max_workers = st.number_input("Parallel Workers", 1, 16, 4)

        uploaded_files = st.file_uploader(
            "Upload PDFs for batch processing",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple PDF files for batch analysis",
        )

        submitted = st.form_submit_button(
            "🚀 Create Batch Job", use_container_width=True, type="primary"
        )

        if submitted and uploaded_files:
            config = BatchConfig(
                max_workers=max_workers, similarity_threshold=threshold
            )
            job = processor.create_job(
                name=job_name or f"Batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                document_paths=[f.name for f in uploaded_files],
                priority=BatchPriority(priority),
                metadata={"uploaded_count": len(uploaded_files)},
            )
            st.success(f"✅ Created job **{job.job_id}**: {job.name}")
            st.json(job.to_dict())
        elif submitted:
            st.warning("Please upload at least 2 PDF files.")


def _render_active_jobs(processor: BatchProcessor):
    """Render active jobs monitoring."""
    st.subheader("Active Batch Jobs")

    stats = processor.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pending", stats.get("pending", 0))
    col2.metric("Processing", stats.get("processing", 0))
    col3.metric("Completed", stats.get("completed", 0))
    col4.metric("Failed", stats.get("failed", 0))

    jobs = processor.list_jobs()
    if not jobs:
        st.info("No batch jobs found. Create one in the Create Job tab.")
        return

    for job in jobs:
        with st.expander(
            f"{'🟢' if job.status == BatchStatus.COMPLETED else '🟡' if job.status == BatchStatus.PROCESSING else '⚪'} {job.name} ({job.job_id})",
            expanded=(job.status == BatchStatus.PROCESSING),
        ):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Status:** {job.status.value}")
            col2.write(f"**Priority:** {job.priority.value}")
            col3.write(f"**Progress:** {job.progress:.1f}%")

            st.progress(job.progress / 100.0)

            st.write(f"Documents: {job.processed_documents}/{job.total_documents}")
            st.write(f"Flagged Pairs: {job.flagged_pairs}")

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            if btn_col1.button("▶️ Process", key=f"process_{job.job_id}"):
                processor.process_job(job.job_id)
                st.rerun()
            if btn_col2.button("⏸️ Pause", key=f"pause_{job.job_id}"):
                processor.pause_job(job.job_id)
                st.rerun()
            if btn_col3.button("❌ Cancel", key=f"cancel_{job.job_id}"):
                processor.cancel_job(job.job_id)
                st.rerun()


def _render_history(history: BatchHistory):
    """Render batch history."""
    st.subheader("Batch Processing History")

    with st.expander("🔍 Search & Filter", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            search_query = st.text_input("Search by name", key="hist_search")
        with col2:
            status_filter = st.selectbox(
                "Status", ["All", "completed", "failed", "cancelled"], key="hist_status"
            )
        with col3:
            limit = st.number_input("Max results", 10, 100, 20, key="hist_limit")

        if st.button("🔍 Search", key="hist_search_btn"):
            records = history.search_jobs(
                query=search_query or None,
                status=status_filter if status_filter != "All" else None,
                limit=limit,
            )
        else:
            records = history.get_recent_jobs(limit=limit)

    if not records:
        st.info("No history records found.")
        return

    df = pd.DataFrame([r.to_dict() for r in records])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", len(records))
    col2.metric("Total Documents", df["document_count"].sum())
    col3.metric("Total Flagged", df["flagged_count"].sum())
    col4.metric(
        "Avg Duration",
        f"{df['duration_seconds'].mean():.1f}s"
        if df["duration_seconds"].mean()
        else "N/A",
    )

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("⬇️ Export CSV", csv, "batch_history.csv", "text/csv")


def _render_analytics(history: BatchHistory):
    """Render analytics visualizations."""
    st.subheader("Batch Processing Analytics")

    stats = history.get_statistics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", stats.get("total_jobs", 0))
    col2.metric("Total Documents Processed", stats.get("total_documents", 0))
    col3.metric("Total Flagged", stats.get("total_flagged", 0))
    col4.metric("Avg Duration", f"{stats.get('avg_duration_seconds', 0):.1f}s")

    # Status distribution pie chart
    by_status = stats.get("by_status", {})
    if by_status:
        fig_status = px.pie(
            values=list(by_status.values()),
            names=list(by_status.keys()),
            title="Jobs by Status",
            color_discrete_map={
                "completed": "#22c55e",
                "failed": "#ef4444",
                "pending": "#94a3b8",
                "processing": "#60a5fa",
            },
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # Daily summary
    daily = history.get_daily_summary(days=14)
    if daily:
        df_daily = pd.DataFrame(daily)
        fig_daily = px.bar(
            df_daily,
            x="date",
            y=["jobs", "documents"],
            title="Daily Activity (Last 14 Days)",
        )
        st.plotly_chart(fig_daily, use_container_width=True)
