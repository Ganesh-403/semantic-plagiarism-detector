"""Stress the application's SQLite configuration with real concurrent connections."""

from concurrent.futures import ThreadPoolExecutor
import threading

from src.db.connection import (
    DEFAULT_SQLITE_TIMEOUT,
    create_connection,
    get_connection,
    resolve_busy_timeout_ms,
)


def test_sqlite_wal_concurrency_stress_under_high_contention(tmp_path):
    db_path = tmp_path / "concurrency.db"
    with get_connection(db_path) as conn:
        conn.execute(
            "CREATE TABLE documents (worker INTEGER, operation INTEGER, PRIMARY KEY(worker, operation))"
        )
        conn.execute("CREATE TABLE incidents (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO incidents VALUES (?)", [(i,) for i in range(10)])
        conn.commit()

    workers, operations = 20, 50
    start = threading.Barrier(workers, timeout=30)

    def worker(worker_id):
        # PRAGMAs such as synchronous and busy_timeout are per connection.
        # Use the production factory for every worker, not a separate test setup.
        with get_connection(db_path) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
            assert conn.execute("PRAGMA busy_timeout").fetchone()[
                0
            ] == resolve_busy_timeout_ms(DEFAULT_SQLITE_TIMEOUT)
            start.wait()
            for operation in range(operations):
                conn.execute(
                    "INSERT INTO documents VALUES (?, ?)", (worker_id, operation)
                )
                conn.commit()
                assert (
                    conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0] == 10
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # result() propagates every worker failure instead of only counting rows.
        futures = [pool.submit(worker, worker_id) for worker_id in range(workers)]
        for future in futures:
            future.result(timeout=60)

    with get_connection(db_path, read_only=True) as conn:
        rows = conn.execute(
            "SELECT worker, COUNT(*) FROM documents GROUP BY worker ORDER BY worker"
        ).fetchall()
        assert [tuple(row) for row in rows] == [(i, operations) for i in range(workers)]
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_wal_reader_remains_available_while_writer_waits(tmp_path):
    path = tmp_path / "held-writer.db"
    with get_connection(path) as conn:
        conn.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY)")
        conn.commit()

    attempted = threading.Event()
    # Configure both connections before acquiring the write lock.
    writer = create_connection(path)
    waiting_writer = create_connection(path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO entries VALUES (1)")
        with get_connection(path, read_only=True) as reader:
            # WAL readers see committed data and do not block on the writer.
            assert reader.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0

        def insert_second():
            attempted.set()
            waiting_writer.execute("INSERT INTO entries VALUES (2)")
            waiting_writer.commit()

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(insert_second)
            assert attempted.wait(5)
            writer.commit()
            pending.result(timeout=DEFAULT_SQLITE_TIMEOUT + 5)
        assert [
            tuple(row) for row in writer.execute("SELECT id FROM entries ORDER BY id")
        ] == [(1,), (2,)]
    finally:
        writer.close()
        waiting_writer.close()
