import os
import queue
import sqlite3
import threading
from typing import List

import pytest


# --- Target Database Bootstrapper & WAL Configurator ---
def init_isolated_wal_db(db_path: str) -> None:
    """Initializes schema and forces WAL journal mode with a transaction timeout."""
    # busy_timeout is set to 5000ms (5 seconds) to allow busy threads to retry gracefully
    conn = sqlite3.connect(db_path, timeout=5.0)
    cursor = conn.cursor()

    # Enable Write-Ahead Logging mode for parallel reading and writing
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")

    # Establish baseline structures matching criteria profiles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL
        );
    """)

    # Pre-seed incidents table to support simultaneous parallel extraction reads
    for i in range(10):
        cursor.execute("INSERT INTO incidents (title) VALUES (?);", (f"Pre-existing Alert {i}",))

    conn.commit()
    conn.close()


# --- Worker Function executing Concurrent Reads & Writes ---
def stress_worker_pipeline(db_path: str, worker_id: int, ops_count: int, error_queue: queue.Queue) -> None:
    """Executes consecutive reads and writes within isolated thread database connections."""
    try:
        # Each thread must instantiate its own isolated connection pool anchor link
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        for i in range(ops_count):
            # Operation 1: High-contention write insertion block
            cursor.execute(
                "INSERT INTO documents (content) VALUES (?);",
                (f"Thread {worker_id} - Document Content Batch payload item chunk {i}",)
            )
            conn.commit()

            # Operation 2: Concurrent table extraction read lookup
            cursor.execute("SELECT COUNT(*) FROM incidents;")
            _ = cursor.fetchone()

        conn.close()
    except Exception as exc:
        # Route any unexpected operational errors back to the tracking validator container
        error_queue.put(f"Worker {worker_id} failed: {str(exc)}")


# --- Stress Verification Suite ---

@pytest.fixture(scope="function")
def sqlite_stress_file(tmp_path):
    """Provides a temporary on-disk file path isolated from current working directories."""
    db_file = os.path.join(tmp_path, "concurrency_stress_test.db")
    init_isolated_wal_db(db_file)
    yield db_file

    # Cleanup trailing WAL and SHM shared memory state journals upon run completion
    for ext in ["", "-wal", "-shm"]:
        target = db_file + ext
        if os.path.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass


def test_sqlite_wal_concurrency_stress_under_high_contention(sqlite_stress_file):
    """
    Scenario: Spin up 20 threads executing 50 sequential reads and writes
              simultaneously to stress busy timeout retry loops.
    Acceptance Criteria:
    - 0 Deadlocks encountered.
    - 100% Successful business processing operations completed.
    """
    db_path = sqlite_stress_file
    threads_count = 20
    operations_per_thread = 50

    error_sharing_queue = queue.Queue()
    thread_pool: list[threading.Thread] = []

    # 1. Spawn 20 separate threads performing operations concurrently
    for i in range(threads_count):
        t = threading.Thread(
            target=stress_worker_pipeline,
            args=(db_path, i, operations_per_thread, error_sharing_queue)
        )
        thread_pool.append(t)

    # 2. Synchronize activation triggers to hit the database simultaneously
    for t in thread_pool:
        t.start()

    # 3. Wait for all concurrent threads to conclude their routines
    for t in thread_pool:
        t.join()

    # 4. Extract any thrown errors from the tracking queue container
    collected_errors = []
    while not error_sharing_queue.empty():
        collected_errors.append(error_sharing_queue.get())

    # 5. Core Assertions: Enforce absolute zero failure thresholds
    assert len(collected_errors) == 0, (
        f"Contention bottlenecks caused operations to fail! "
        f"Errors detected: {collected_errors}"
    )

    # 6. Verify data integrity: Expected total database size should match inputs exactly
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents;")
    total_inserted_docs = cursor.fetchone()[0]
    conn.close()

    expected_total_records = threads_count * operations_per_thread
    assert total_inserted_docs == expected_total_records, (
        f"Data drift detected! Expected {expected_total_records} logs but got {total_inserted_docs}"
    )
