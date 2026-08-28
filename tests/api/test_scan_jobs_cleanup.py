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
tests/api/test_scan_jobs_cleanup.py
-----------------------------------
Tests for auto-expiration and cleanup for in-memory scan_jobs (Issue #3223).
"""

from datetime import datetime, timedelta, timezone

from src.api.routers.analysis import (
    SCAN_JOB_TTL_SECONDS,
    cleanup_expired_scan_jobs,
    scan_jobs,
)


def setup_function():
    """Clear scan_jobs before every test."""
    scan_jobs.clear()


def teardown_function():
    """Clear scan_jobs after every test."""
    scan_jobs.clear()


def test_cleanup_expired_scan_jobs_removes_old_completed_jobs():
    """Verify that jobs completed more than 2 hours ago are removed."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(seconds=7300)).isoformat()
    recent_time = (now - timedelta(seconds=60)).isoformat()

    scan_jobs["job_old"] = {
        "job_id": "job_old",
        "status": "completed",
        "filename": "old.pdf",
        "created_at": old_time,
        "completed_at": old_time,
        "result": {"score": 0.1},
        "error": None,
    }
    scan_jobs["job_recent"] = {
        "job_id": "job_recent",
        "status": "completed",
        "filename": "recent.pdf",
        "created_at": recent_time,
        "completed_at": recent_time,
        "result": {"score": 0.2},
        "error": None,
    }

    evicted = cleanup_expired_scan_jobs(max_age_seconds=7200)

    assert evicted == 1
    assert "job_old" not in scan_jobs
    assert "job_recent" in scan_jobs


def test_cleanup_expired_scan_jobs_removes_old_failed_jobs():
    """Verify that failed jobs older than max_age_seconds are removed."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=3)).isoformat()

    scan_jobs["job_failed_old"] = {
        "job_id": "job_failed_old",
        "status": "failed",
        "filename": "broken.pdf",
        "created_at": old_time,
        "completed_at": old_time,
        "result": None,
        "error": "Extraction error",
    }

    evicted = cleanup_expired_scan_jobs(max_age_seconds=7200)

    assert evicted == 1
    assert "job_failed_old" not in scan_jobs


def test_cleanup_preserves_queued_and_processing_jobs():
    """Verify that in-flight jobs (queued or processing) are never evicted by age."""
    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(hours=5)).isoformat()

    scan_jobs["job_processing"] = {
        "job_id": "job_processing",
        "status": "processing",
        "filename": "in_progress.pdf",
        "created_at": old_time,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    scan_jobs["job_queued"] = {
        "job_id": "job_queued",
        "status": "queued",
        "filename": "queued.pdf",
        "created_at": old_time,
        "completed_at": None,
        "result": None,
        "error": None,
    }

    evicted = cleanup_expired_scan_jobs(max_age_seconds=7200)

    assert evicted == 0
    assert "job_processing" in scan_jobs
    assert "job_queued" in scan_jobs


def test_cleanup_custom_ttl():
    """Verify custom max_age_seconds threshold."""
    now = datetime.now(timezone.utc)
    time_30s_ago = (now - timedelta(seconds=30)).isoformat()

    scan_jobs["job_short_ttl"] = {
        "job_id": "job_short_ttl",
        "status": "completed",
        "filename": "doc.pdf",
        "created_at": time_30s_ago,
        "completed_at": time_30s_ago,
        "result": {},
        "error": None,
    }

    # TTL 60s should keep it
    assert cleanup_expired_scan_jobs(max_age_seconds=60) == 0
    assert "job_short_ttl" in scan_jobs

    # TTL 10s should evict it
    assert cleanup_expired_scan_jobs(max_age_seconds=10) == 1
    assert "job_short_ttl" not in scan_jobs


def test_cleanup_lru_capacity_eviction():
    """Verify capacity-based eviction removes oldest completed jobs when exceeding max_capacity."""
    now = datetime.now(timezone.utc)

    for i in range(5):
        t = (now - timedelta(minutes=10 - i)).isoformat()
        scan_jobs[f"job_{i}"] = {
            "job_id": f"job_{i}",
            "status": "completed",
            "filename": f"doc_{i}.pdf",
            "created_at": t,
            "completed_at": t,
            "result": {},
            "error": None,
        }

    # Max capacity of 3 with max_age of 24h should evict the 2 oldest jobs (job_0 and job_1)
    evicted = cleanup_expired_scan_jobs(max_age_seconds=86400, max_capacity=3)
    assert evicted == 2
    assert "job_0" not in scan_jobs
    assert "job_1" not in scan_jobs
    assert "job_2" in scan_jobs
    assert "job_3" in scan_jobs
    assert "job_4" in scan_jobs


def test_scan_job_ttl_seconds_default_value():
    """Verify SCAN_JOB_TTL_SECONDS defaults to 2 hours (7200 seconds)."""
    assert SCAN_JOB_TTL_SECONDS == 7200
