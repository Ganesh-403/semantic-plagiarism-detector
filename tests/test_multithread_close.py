"""Close real pooled SQLite connections concurrently and reopen them."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import sqlite3
import pytest
from src.db import corpus_db


def test_concurrent_global_close_is_idempotent(mock_db):
    barrier = Barrier(8)
    def open_connection(_):
        with corpus_db._connect() as connection:
            connection.execute("SELECT 1")
        barrier.wait(timeout=10)
        return connection
    with ThreadPoolExecutor(max_workers=8) as pool:
        connections = list(pool.map(open_connection, range(8)))
        assert len({id(connection) for connection in connections}) == 8
        list(pool.map(lambda _: corpus_db.close_connections(all_threads=True), range(16)))
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    with corpus_db._connect() as fresh:
        assert fresh.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_current_thread_close_can_be_repeated(mock_db):
    with corpus_db._connect() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    corpus_db.close_connections()
    corpus_db.close_connections()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
