"""
Tests for Batch Processing Engine.

Comprehensive test suite covering batch job creation, processing,
history tracking, and export functionality.
"""

import os
import json
import tempfile
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.core.batch_processor import (
    BatchProcessor,
    BatchJob,
    BatchConfig,
    BatchStatus,
    BatchPriority,
)
from src.core.batch_history import BatchHistory, HistoryRecord
from src.utils.batch_export import BatchExporter, ReportFormatter, ExportConfig


class TestBatchJob:
    """Tests for BatchJob data class."""

    def test_create_job(self):
        """Test basic job creation."""
        job = BatchJob(
            job_id="test-001",
            name="Test Job",
            document_paths=["doc1.pdf", "doc2.pdf"],
            total_documents=2,
        )
        assert job.job_id == "test-001"
        assert job.name == "Test Job"
        assert job.status == BatchStatus.PENDING
        assert job.total_documents == 2
        assert job.progress == 0.0

    def test_job_to_dict(self):
        """Test job serialization."""
        job = BatchJob(job_id="t-001", name="Test", document_paths=[])
        d = job.to_dict()
        assert d["job_id"] == "t-001"
        assert d["status"] == "pending"
        assert d["priority"] == "normal"

    def test_update_progress(self):
        """Test progress update."""
        job = BatchJob(
            job_id="t-002", name="Test", document_paths=[], total_documents=10
        )
        job.update_progress(5, flagged=2, high=1)
        assert job.progress == 50.0
        assert job.processed_documents == 5
        assert job.flagged_pairs == 2
        assert job.high_severity_count == 1


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    def setup_method(self):
        self.config = BatchConfig(max_workers=2)
        self.processor = BatchProcessor(self.config)

    def test_create_job(self):
        """Test job creation."""
        job = self.processor.create_job("Test", ["a.pdf", "b.pdf"])
        assert job.job_id in self.processor.jobs
        assert job.total_documents == 2

    def test_get_job(self):
        """Test job retrieval."""
        job = self.processor.create_job("Test", ["a.pdf"])
        retrieved = self.processor.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_list_jobs(self):
        """Test job listing."""
        self.processor.create_job("Job1", ["a.pdf"])
        self.processor.create_job("Job2", ["b.pdf", "c.pdf"])
        jobs = self.processor.list_jobs()
        assert len(jobs) == 2

    def test_cancel_job(self):
        """Test job cancellation."""
        job = self.processor.create_job("Test", ["a.pdf"])
        result = self.processor.cancel_job(job.job_id)
        assert result is True
        assert job.status == BatchStatus.CANCELLED

    def test_cancel_nonexistent_job(self):
        """Test cancelling nonexistent job."""
        result = self.processor.cancel_job("nonexistent")
        assert result is False

    def test_pause_resume(self):
        """Test pause and resume."""
        job = self.processor.create_job("Test", ["a.pdf"])
        job.status = BatchStatus.PROCESSING
        assert self.processor.pause_job(job.job_id) is True
        assert job.status == BatchStatus.PAUSED
        assert self.processor.resume_job(job.job_id) is True
        assert job.status == BatchStatus.PROCESSING

    def test_get_statistics(self):
        """Test statistics."""
        self.processor.create_job("J1", ["a.pdf"])
        self.processor.create_job("J2", ["b.pdf", "c.pdf"])
        stats = self.processor.get_statistics()
        assert stats["total_jobs"] == 2
        assert stats["total_documents"] == 3

    def test_clear_completed(self):
        """Test clearing completed jobs."""
        job = self.processor.create_job("Test", ["a.pdf"])
        job.status = BatchStatus.COMPLETED
        count = self.processor.clear_completed()
        assert count == 1
        assert len(self.processor.jobs) == 0

    def test_export_results(self):
        """Test result export."""
        job = self.processor.create_job("Test", ["a.pdf"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = self.processor.export_results(job.job_id, path)
            assert result is True
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestBatchHistory:
    """Tests for BatchHistory."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.history = BatchHistory(self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_record_job(self):
        """Test recording a job."""
        data = {
            "job_id": "h-001",
            "name": "Test",
            "status": "completed",
            "document_count": 5,
            "flagged_count": 2,
            "created_at": datetime.now().isoformat(),
        }
        assert self.history.record_job(data) is True

    def test_get_job(self):
        """Test retrieving a job."""
        data = {
            "job_id": "h-002",
            "name": "Test2",
            "status": "completed",
            "document_count": 3,
            "created_at": datetime.now().isoformat(),
        }
        self.history.record_job(data)
        record = self.history.get_job("h-002")
        assert record is not None
        assert record.name == "Test2"

    def test_get_recent(self):
        """Test recent jobs."""
        for i in range(5):
            self.history.record_job(
                {
                    "job_id": f"h-{i}",
                    "name": f"Job {i}",
                    "status": "completed",
                    "created_at": datetime.now().isoformat(),
                }
            )
        recent = self.history.get_recent_jobs(limit=3)
        assert len(recent) == 3

    def test_search(self):
        """Test search functionality."""
        self.history.record_job(
            {
                "job_id": "s-001",
                "name": "Assignment Check",
                "status": "completed",
                "document_count": 10,
                "created_at": datetime.now().isoformat(),
            }
        )
        results = self.history.search_jobs(query="Assignment")
        assert len(results) == 1

    def test_statistics(self):
        """Test statistics."""
        self.history.record_job(
            {
                "job_id": "st-001",
                "name": "Test",
                "status": "completed",
                "document_count": 10,
                "flagged_count": 3,
                "created_at": datetime.now().isoformat(),
            }
        )
        stats = self.history.get_statistics()
        assert stats["total_jobs"] == 1
        assert stats["total_documents"] == 10


class TestBatchExporter:
    """Tests for BatchExporter."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.exporter = BatchExporter(ExportConfig(output_dir=self.tmp_dir))

    def test_export_json(self):
        """Test JSON export."""
        data = {"test": "value", "count": 42}
        path = self.exporter.export_json(data, "test.json")
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["results"]["test"] == "value"

    def test_export_csv(self):
        """Test CSV export."""
        data = [{"name": "Test1", "value": 1}, {"name": "Test2", "value": 2}]
        path = self.exporter.export_csv(data, "test.csv")
        assert os.path.exists(path)

    def test_generate_text_summary(self):
        """Test text summary generation."""
        results = {"total_jobs": 10, "completed": 8, "total_documents": 50}
        summary = self.exporter.generate_text_summary(results)
        assert "Total Jobs: 10" in summary
        assert "Total Documents: 50" in summary


class TestReportFormatter:
    """Tests for ReportFormatter."""

    def test_format_duration(self):
        """Test duration formatting."""
        assert ReportFormatter.format_duration(None) == "N/A"
        assert ReportFormatter.format_duration(30.5) == "30.5s"
        assert ReportFormatter.format_duration(125) == "2m 5s"
        assert ReportFormatter.format_duration(3661) == "1h 1m"

    def test_status_emoji(self):
        """Test status emoji."""
        assert ReportFormatter.format_status_emoji("completed") == "✅"
        assert ReportFormatter.format_status_emoji("failed") == "❌"
        assert ReportFormatter.format_status_emoji("unknown") == "❓"
