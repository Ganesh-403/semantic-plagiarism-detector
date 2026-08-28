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
test_sqlite_timeout_concurrency_documentation_issue_2721.py
--------------------------------------------------------------
Comprehensive unit test suite for Issue #2721:
Validating SQLITE_TIMEOUT = 15.0 configuration, busy timeout resolution mechanics,
and concurrency behavior during multi-threaded simulated database operations.

This suite ensures that:
1. `SQLITE_TIMEOUT` in `src/db/auth.py` and `DEFAULT_SQLITE_TIMEOUT` in `src/db/connection.py`
   remain strictly set to 15.0 seconds.
2. `resolve_busy_timeout_ms(15.0)` resolves correctly to 15000 ms.
3. PRAGMA busy_timeout is properly configured on SQLite connection objects.
4. Concurrent multi-threaded reads/writes leverage the 15.0s busy timeout without premature locking crashes.
5. High-concurrency lock contention under WAL mode handles passive checkpointing sweeps.
"""

import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from src.db.auth import SQLITE_TIMEOUT as AUTH_SQLITE_TIMEOUT
from src.db.connection import (
    DEFAULT_SQLITE_TIMEOUT,
    MIN_BUSY_TIMEOUT_MS,
    apply_busy_timeout,
    create_connection,
    get_connection,
    resolve_busy_timeout_ms,
)

# ---------------------------------------------------------------------------
# Section 1: Constant & Configuration Assertions
# ---------------------------------------------------------------------------


def test_sqlite_timeout_constants_value():
    """Assert that both module constants equal 15.0 seconds as documented."""
    assert AUTH_SQLITE_TIMEOUT == 15.0
    assert DEFAULT_SQLITE_TIMEOUT == 15.0


def test_resolve_busy_timeout_ms_conversions():
    """Assert busy timeout ms resolution for 15.0 seconds and custom values."""
    assert resolve_busy_timeout_ms(15.0) == 15000
    assert resolve_busy_timeout_ms(1.0) == 1000
    assert resolve_busy_timeout_ms(0.05) == MIN_BUSY_TIMEOUT_MS  # Floor check (~100ms)


def test_resolve_busy_timeout_ms_validation():
    """Assert validation errors for invalid or negative timeout values."""
    with pytest.raises(ValueError):
        resolve_busy_timeout_ms(0)

    with pytest.raises(ValueError):
        resolve_busy_timeout_ms(-5.0)

    with pytest.raises(ValueError):
        resolve_busy_timeout_ms(float("nan"))

    with pytest.raises(TypeError):
        resolve_busy_timeout_ms("15.0")  # type: ignore

    with pytest.raises(TypeError):
        resolve_busy_timeout_ms(True)  # type: ignore


# ---------------------------------------------------------------------------
# Section 2: Connection PRAGMA Configuration Assertions
# ---------------------------------------------------------------------------


def test_connection_pragma_busy_timeout_applied():
    """Verify that create_connection applies PRAGMA busy_timeout = 15000."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        conn = create_connection(db_path, timeout=15.0)
        cursor = conn.execute("PRAGMA busy_timeout;")
        row = cursor.fetchone()
        assert row is not None
        # PRAGMA busy_timeout returns busy timeout in milliseconds
        assert row[0] == 15000
        conn.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_apply_busy_timeout_helper():
    """Verify direct apply_busy_timeout helper function behavior."""
    conn = sqlite3.connect(":memory:")
    try:
        applied_ms = apply_busy_timeout(conn, 15.0)
        assert applied_ms == 15000
        cursor = conn.execute("PRAGMA busy_timeout;")
        assert cursor.fetchone()[0] == 15000
    finally:
        conn.close()


def test_get_connection_context_manager():
    """Verify get_connection context manager yields configured connection."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        with get_connection(db_path, timeout=15.0) as conn:
            cursor = conn.execute("PRAGMA busy_timeout;")
            assert cursor.fetchone()[0] == 15000
    finally:
        Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Section 3: Concurrent Multi-Threaded Database Lock Simulations
# ---------------------------------------------------------------------------


def _worker_write_task(db_path: str, worker_id: int, lock_hold_duration: float):
    """Simulate a worker writing to SQLite database under lock contention."""
    with get_connection(db_path, timeout=15.0) as conn:
        conn.execute(
            "INSERT INTO audit_log (worker_id, timestamp) VALUES (?, ?);",
            (worker_id, time.time()),
        )
        conn.commit()
        if lock_hold_duration > 0:
            time.sleep(lock_hold_duration)
    return worker_id


def test_concurrent_multi_threaded_writes_with_15s_timeout():
    """Simulate concurrent threads writing to SQLite with 15.0s busy timeout.
    Verifies that threads wait for locks rather than throwing immediate database locked errors.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize schema
        with get_connection(db_path, timeout=15.0) as conn:
            conn.execute(
                "CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, worker_id INT, timestamp REAL);"
            )
            conn.commit()

        worker_count = 10
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_worker_write_task, db_path, i, 0.05)
                for i in range(worker_count)
            ]
            completed_workers = [f.result() for f in as_completed(futures)]

        assert len(completed_workers) == worker_count

        # Verify all records were committed cleanly
        with get_connection(db_path, timeout=15.0) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM audit_log;")
            count = cursor.fetchone()[0]
            assert count == worker_count
    finally:
        Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Section 4: Bulk FAISS Vector Index & WAL Checkpoint Lock Simulations
# ---------------------------------------------------------------------------


def test_simulated_bulk_faiss_sync_write_lock_contention():
    """Simulate heavy background write transaction (e.g. FAISS vector sync) while
    a concurrent read/write query executes with 15.0s timeout.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        with get_connection(db_path, timeout=15.0) as conn:
            conn.execute(
                "CREATE TABLE embeddings (id INT PRIMARY KEY, vector_data TEXT);"
            )
            conn.commit()

        def _bulk_faiss_sync():
            with get_connection(db_path, timeout=15.0) as conn:
                conn.execute("BEGIN EXCLUSIVE TRANSACTION;")
                for i in range(50):
                    conn.execute(
                        "INSERT INTO embeddings (id, vector_data) VALUES (?, ?);",
                        (i, f"vector_embedding_{i}"),
                    )
                time.sleep(0.3)  # Hold exclusive transaction for 300ms
                conn.commit()

        def _concurrent_read():
            time.sleep(0.05)  # Let FAISS sync start first
            with get_connection(db_path, timeout=15.0) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings;")
                return cursor.fetchone()[0]

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(_bulk_faiss_sync)
            f2 = executor.submit(_concurrent_read)

            f1.result()
            inserted_count = f2.result()

        assert inserted_count == 50
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_wal_checkpoint_lock_contention_simulation():
    """Simulate WAL mode passive checkpointing sweep lock contention handling."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        with get_connection(db_path, timeout=15.0) as conn:
            conn.execute(
                "CREATE TABLE document_nodes (id INT PRIMARY KEY, content TEXT);"
            )
            conn.commit()

        def _wal_writer():
            with get_connection(db_path, timeout=15.0) as conn:
                for i in range(100):
                    conn.execute(
                        "INSERT INTO document_nodes VALUES (?, ?);", (i, f"node_{i}")
                    )
                conn.commit()
                # Run explicit passive checkpoint sweep
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")

        def _concurrent_worker():
            time.sleep(0.02)
            with get_connection(db_path, timeout=15.0) as conn:
                conn.execute(
                    "INSERT INTO document_nodes VALUES (?, ?);",
                    (999, "concurrent_node"),
                )
                conn.commit()

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(_wal_writer)
            f2 = executor.submit(_concurrent_worker)
            f1.result()
            f2.result()

        with get_connection(db_path, timeout=15.0) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM document_nodes;")
            assert cursor.fetchone()[0] == 101
    finally:
        Path(db_path).unlink(missing_ok=True)
