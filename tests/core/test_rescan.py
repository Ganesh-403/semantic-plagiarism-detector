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
tests/core/test_rescan.py
--------------------------
Unit tests for the scheduled/event-driven plagiarism rescan pipeline
(`src.core.processing.rescan_recent_documents`).

Covers the acceptance criteria from the "continuous monitoring" issue:
  * With two documents crossing the threshold, a rescan produces exactly
    one new incident.
  * A rerun after the incident exists creates zero new rows and does not
    re-fire the webhook alert.
  * The FAISS index is guarded by the same ``FAISSLock`` used by manual
    scans, so a scheduled rescan and a concurrent manual scan cannot
    corrupt the index.
  * The run is recorded so the scheduler is restart-safe.

FAISS itself and the clock are both mocked, per the issue's request for
"unit tests with a mocked index and clock".
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.concurrency import ConcurrencyTimeoutError, faiss_write_lock
from src.core.processing import RESCAN_JOB_NAME, rescan_recent_documents
from src.db.incidents import get_all_incidents, get_last_scheduler_run, incident_exists

FIXED_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _fake_chunks(doc_name: str, text: str = "some paragraph text") -> tuple:
    """A minimal (texts, embeddings) pair shaped like corpus_db's real return value."""
    return ([text], np.zeros((1, 384), dtype=np.float32))


@pytest.fixture(autouse=True)
def _patch_common(monkeypatch, tmp_path):
    """Common patches shared by every test in this module.

    * Points the FAISS index/lock at a tmp path so tests never touch (or
      leave stray ``.lock`` files beside) the real repo-root index.
    * Runs "background" webhook dispatch synchronously so assertions don't
      need to race a thread pool.
    * Replaces the actual FAISS index load/search with lightweight fakes —
      the pipeline under test only cares about the *matches* FAISS would
      have returned, not FAISS itself.
    """
    fake_index_path = tmp_path / "corpus.index"
    monkeypatch.setattr("src.core.app_config.FAISS_INDEX_PATH", fake_index_path)

    monkeypatch.setattr(
        "src.core.faiss_index.load_or_rebuild_index",
        lambda filepath: (MagicMock(name="fake_index"), [MagicMock()], False),
    )

    # run_background normally submits to a thread pool; execute inline so
    # tests can assert on the dispatch call deterministically.
    def _run_inline(func, *args, **kwargs):
        func(*args, **kwargs)
        return MagicMock()

    monkeypatch.setattr("src.core.synchronization.run_background", _run_inline)

    dispatch_mock = MagicMock(return_value=True)
    monkeypatch.setattr("src.core.webhook.dispatch_plagiarism_alert", dispatch_mock)

    yield {"dispatch_mock": dispatch_mock, "fake_index_path": fake_index_path}


def test_rescan_with_no_recent_documents_is_a_noop(mock_db, monkeypatch, _patch_common):
    """No documents in the grace period => no incidents, run still recorded."""
    monkeypatch.setattr("src.db.corpus_db.get_documents_since", lambda since_iso: [])

    result = rescan_recent_documents(
        grace_period=60, threshold=0.75, now=FIXED_NOW, db_path=mock_db
    )

    assert result.documents_scanned == 0
    assert result.new_incidents == []
    assert result.total_flags == 0
    _patch_common["dispatch_mock"].assert_not_called()

    last_run = get_last_scheduler_run(RESCAN_JOB_NAME, mock_db)
    assert last_run is not None
    assert last_run["documents_scanned"] == 0
    assert last_run["new_incidents"] == 0


def test_rescan_two_documents_crossing_threshold_creates_one_incident(
    mock_db, monkeypatch, _patch_common
):
    """Acceptance criterion: exactly one incident for one genuinely-new match."""
    monkeypatch.setattr(
        "src.db.corpus_db.get_documents_since", lambda since_iso: ["studentB.pdf"]
    )
    monkeypatch.setattr(
        "src.db.corpus_db.get_chunks_for_documents",
        lambda filenames: {"studentB.pdf": _fake_chunks("studentB.pdf")},
    )
    monkeypatch.setattr(
        "src.core.faiss_index.find_plagiarised_chunks",
        lambda *a, **kw: [
            {
                "source_doc": "studentB.pdf",
                "source_chunk_text": "identical paragraph",
                "match_doc": "studentA.pdf",
                "match_chunk_text": "identical paragraph",
                "similarity": 0.95,
            }
        ],
    )

    result = rescan_recent_documents(
        grace_period=60, threshold=0.75, now=FIXED_NOW, db_path=mock_db
    )

    assert result.documents_scanned == 1
    assert len(result.new_incidents) == 1
    incident = result.new_incidents[0]
    assert {incident["doc_a"], incident["doc_b"]} == {"studentA.pdf", "studentB.pdf"}

    assert incident_exists("studentA.pdf", "studentB.pdf", mock_db) is True
    assert len(get_all_incidents(mock_db)) == 1

    _patch_common["dispatch_mock"].assert_called_once()
    called_args = _patch_common["dispatch_mock"].call_args.args
    assert set(called_args[:2]) == {"studentA.pdf", "studentB.pdf"}


def test_rescan_rerun_creates_zero_new_rows_and_does_not_renotify(
    mock_db, monkeypatch, _patch_common
):
    """Acceptance criterion: rerunning after the incident exists is a no-op."""
    monkeypatch.setattr(
        "src.db.corpus_db.get_documents_since", lambda since_iso: ["studentB.pdf"]
    )
    monkeypatch.setattr(
        "src.db.corpus_db.get_chunks_for_documents",
        lambda filenames: {"studentB.pdf": _fake_chunks("studentB.pdf")},
    )
    monkeypatch.setattr(
        "src.core.faiss_index.find_plagiarised_chunks",
        lambda *a, **kw: [
            {
                "source_doc": "studentB.pdf",
                "source_chunk_text": "identical paragraph",
                "match_doc": "studentA.pdf",
                "match_chunk_text": "identical paragraph",
                "similarity": 0.95,
            }
        ],
    )

    first = rescan_recent_documents(
        grace_period=60, threshold=0.75, now=FIXED_NOW, db_path=mock_db
    )
    assert len(first.new_incidents) == 1
    assert _patch_common["dispatch_mock"].call_count == 1

    second = rescan_recent_documents(
        grace_period=60, threshold=0.75, now=FIXED_NOW, db_path=mock_db
    )

    assert second.new_incidents == []
    # The pair is still flagged (total_flags), just not a *new* incident.
    assert second.total_flags == 1
    assert len(get_all_incidents(mock_db)) == 1
    # No second webhook fire for a pair we already alerted on.
    assert _patch_common["dispatch_mock"].call_count == 1


def test_rescan_below_threshold_creates_no_incident(
    mock_db, monkeypatch, _patch_common
):
    monkeypatch.setattr(
        "src.db.corpus_db.get_documents_since", lambda since_iso: ["studentB.pdf"]
    )
    monkeypatch.setattr(
        "src.db.corpus_db.get_chunks_for_documents",
        lambda filenames: {"studentB.pdf": _fake_chunks("studentB.pdf")},
    )
    monkeypatch.setattr(
        "src.core.faiss_index.find_plagiarised_chunks",
        lambda *a, **kw: [
            {
                "source_doc": "studentB.pdf",
                "source_chunk_text": "somewhat similar",
                "match_doc": "studentA.pdf",
                "match_chunk_text": "somewhat similar-ish",
                "similarity": 0.40,
            }
        ],
    )

    result = rescan_recent_documents(
        grace_period=60, threshold=0.75, now=FIXED_NOW, db_path=mock_db
    )

    assert result.new_incidents == []
    assert result.total_flags == 0
    assert len(get_all_incidents(mock_db)) == 0
    _patch_common["dispatch_mock"].assert_not_called()


def test_rescan_restart_safe_last_run_is_queryable(mock_db, monkeypatch, _patch_common):
    """The scheduler can consult the last-completed run after a process restart."""
    monkeypatch.setattr(
        "src.db.corpus_db.get_documents_since", lambda since_iso: ["studentB.pdf"]
    )
    monkeypatch.setattr(
        "src.db.corpus_db.get_chunks_for_documents",
        lambda filenames: {"studentB.pdf": _fake_chunks("studentB.pdf")},
    )
    monkeypatch.setattr(
        "src.core.faiss_index.find_plagiarised_chunks", lambda *a, **kw: []
    )

    assert get_last_scheduler_run(RESCAN_JOB_NAME, mock_db) is None

    rescan_recent_documents(
        grace_period=60, threshold=0.75, now=FIXED_NOW, db_path=mock_db
    )

    last_run = get_last_scheduler_run(RESCAN_JOB_NAME, mock_db)
    assert last_run is not None
    assert last_run["documents_scanned"] == 1
    assert last_run["last_run_at"].startswith("2026-06-01")


def test_rescan_holds_faiss_lock_and_excludes_manual_scans(
    mock_db, monkeypatch, _patch_common
):
    """Acceptance criterion: manual scans and the scheduler cannot both hold
    the FAISS lock at once (lock correctness / mutual exclusion)."""
    lock_path = f"{_patch_common['fake_index_path']}.lock"

    monkeypatch.setattr(
        "src.db.corpus_db.get_documents_since", lambda since_iso: ["studentB.pdf"]
    )
    monkeypatch.setattr(
        "src.db.corpus_db.get_chunks_for_documents",
        lambda filenames: {"studentB.pdf": _fake_chunks("studentB.pdf")},
    )

    lock_held_event = threading.Event()

    def _slow_find_plagiarised_chunks(*args, **kwargs):
        # Signal that we're inside the locked section, then hold it briefly
        # so a concurrent "manual scan" attempt can observe the lock held.
        lock_held_event.set()
        time.sleep(0.35)
        return []

    monkeypatch.setattr(
        "src.core.faiss_index.find_plagiarised_chunks",
        _slow_find_plagiarised_chunks,
    )

    rescan_thread = threading.Thread(
        target=rescan_recent_documents,
        kwargs=dict(grace_period=60, threshold=0.75, now=FIXED_NOW, db_path=mock_db),
    )
    rescan_thread.start()

    assert lock_held_event.wait(timeout=2.0), "rescan never entered the locked section"

    # Simulate a concurrent manual scan trying to acquire the same lock
    # while the scheduled rescan still holds it — it must NOT succeed
    # immediately, proving the two paths are mutually exclusive.
    with pytest.raises(ConcurrencyTimeoutError):
        with faiss_write_lock(lock_path=lock_path, timeout=0.1):
            pass  # pragma: no cover - should never be reached

    rescan_thread.join(timeout=5.0)
    assert not rescan_thread.is_alive()

    # After the rescan releases the lock, a manual-scan-style acquisition
    # succeeds normally.
    with faiss_write_lock(lock_path=lock_path, timeout=2.0):
        pass
