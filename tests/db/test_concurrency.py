from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


from src.db.migrations import migrate_corpus_database


def connect(path) -> sqlite3.Connection:
    """Open an SQLite connection with multi-thread support enabled."""
    connection = sqlite3.connect(str(path), check_same_thread=False, timeout=15.0)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def test_concurrent_readers_and_writers_no_database_locked(tmp_path):
    """
    Spawn 20 worker threads executing concurrent read and write operations
    on a temp database. Assert zero sqlite3.OperationalError: database is locked.
    (Issue #1379)
    """
    db_path = tmp_path / "concurrency-test.db"

    with connect(db_path) as conn:
        migrate_corpus_database(conn)

    errors: list[Exception] = []
    lock = threading.Lock()

    def worker_task(worker_id: int) -> None:
        try:
            with connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO documents (filename, file_hash, upload_date)
                    VALUES (?, ?, ?)
                    """,
                    (
                        f"doc_{worker_id}.pdf",
                        f"hash_{worker_id}",
                        "2026-01-01T00:00:00",
                    ),
                )

                row = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE filename = ?",
                    (f"doc_{worker_id}.pdf",),
                ).fetchone()
                assert row[0] == 1

                conn.execute(
                    """
                    INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        worker_id,
                        f"doc_{worker_id}.pdf",
                        0,
                        f"Test chunk content for worker {worker_id}",
                        b"\x00\x00\x00\x00",
                    ),
                )

                conn.execute("SELECT COUNT(*) FROM chunks").fetchone()

        except sqlite3.OperationalError as exc:
            with lock:
                errors.append(exc)
        except Exception as exc:
            with lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(worker_task, i) for i in range(20)]
        for future in as_completed(futures):
            future.result()

    locked_errors = [
        e for e in errors if "database is locked" in str(e).lower()
    ]
    assert len(locked_errors) == 0, (
        f"Got {len(locked_errors)} 'database is locked' errors: {locked_errors}"
    )

    with connect(db_path) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert doc_count == 20, f"Expected 20 documents, got {doc_count}"
        assert chunk_count == 20, f"Expected 20 chunks, got {chunk_count}"


def test_concurrent_reads_only_no_database_locked(tmp_path):
    """
    Pre-populate the database and spawn 20 threads doing read-only queries.
    """
    db_path = tmp_path / "concurrency-readonly.db"

    with connect(db_path) as conn:
        migrate_corpus_database(conn)
        conn.execute(
            "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
            ("shared.pdf", "shared-hash", "2026-01-01T00:00:00"),
        )
        for i in range(50):
            conn.execute(
                """
                INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (i, "shared.pdf", i, f"chunk {i}", b"\x00\x00"),
            )

    errors: list[Exception] = []
    lock = threading.Lock()

    def reader_task(_worker_id: int) -> None:
        try:
            with connect(db_path) as conn:
                for _ in range(10):
                    conn.execute("SELECT COUNT(*) FROM documents").fetchone()
                    conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
                    conn.execute(
                        "SELECT * FROM chunks WHERE filename = ? LIMIT 5",
                        ("shared.pdf",),
                    ).fetchall()
        except sqlite3.OperationalError as exc:
            with lock:
                errors.append(exc)
        except Exception as exc:
            with lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(reader_task, i) for i in range(20)]
        for future in as_completed(futures):
            future.result()

    locked_errors = [
        e for e in errors if "database is locked" in str(e).lower()
    ]
    assert len(locked_errors) == 0, (
        f"Got {len(locked_errors)} 'database is locked' errors during read-only test"
    )
