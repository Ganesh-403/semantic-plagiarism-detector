import sqlite3
import tempfile
import threading
import pathlib
import pytest

def worker_insert(db_path, thread_id, iterations=50):
    """Worker function to execute concurrent insert transactions."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    try:
        for i in range(iterations):
            cursor.execute(
                "INSERT INTO test_wal (thread_id, payload) VALUES (?, ?)",
                (thread_id, f"data-{thread_id}-{i}")
            )
            conn.commit()
    finally:
        conn.close()

def test_wal_concurrency_mode():
    """Verify SQLite behavior in WAL mode under 10 concurrent writer threads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(pathlib.Path(tmpdir) / "wal_test.db")
        
        # Initialize database and enable WAL mode
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("CREATE TABLE test_wal (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id INTEGER, payload TEXT);")
        conn.commit()
        conn.close()

        num_threads = 10
        iterations_per_thread = 20
        threads = []

        # Spawn 10 concurrent writer threads
        for t_id in range(num_threads):
            thread = threading.Thread(target=worker_insert, args=(db_path, t_id, iterations_per_thread))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all records were inserted successfully without locked database exceptions
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_wal;")
        total_rows = cursor.fetchone()[0]
        conn.close()

        expected_total = num_threads * iterations_per_thread
        assert total_rows == expected_total, f"Expected {expected_total} rows, but found {total_rows}"