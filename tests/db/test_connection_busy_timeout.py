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
tests/db/test_connection_busy_timeout.py
----------------------------------------
Regression tests for issue #2176.

``create_connection()`` used to run a hardcoded ``PRAGMA busy_timeout = 5000``
right after ``sqlite3.connect(timeout=...)``, which threw away whatever timeout
the caller had asked for. These tests pin the pragma to the caller's value so
the regression cannot come back unnoticed.
"""

import sqlite3

import pytest

from src.db.connection import (
    DEFAULT_SQLITE_TIMEOUT,
    MIN_BUSY_TIMEOUT_MS,
    apply_busy_timeout,
    create_connection,
    get_connection,
    resolve_busy_timeout_ms,
)


@pytest.fixture
def db_path(tmp_path):
    """Path to a fresh SQLite database file with one table in it."""
    path = tmp_path / "busy_timeout.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()
    return path


def _read_busy_timeout(conn):
    """Return the connection's currently configured busy timeout in ms."""
    return conn.execute("PRAGMA busy_timeout").fetchone()[0]


# ── resolve_busy_timeout_ms ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seconds, expected_ms",
    [
        (1.0, 1000),
        (5.0, 5000),
        (15.0, 15000),
        (30.0, 30000),
        (0.5, 500),
        (2, 2000),
    ],
)
def test_resolve_busy_timeout_converts_seconds_to_milliseconds(seconds, expected_ms):
    assert resolve_busy_timeout_ms(seconds) == expected_ms


def test_resolve_busy_timeout_applies_lower_bound():
    """Very small timeouts are floored so locks are still briefly waited on."""
    assert resolve_busy_timeout_ms(0.001) == MIN_BUSY_TIMEOUT_MS
    assert resolve_busy_timeout_ms(0.05) == MIN_BUSY_TIMEOUT_MS


@pytest.mark.parametrize("bad_timeout", [0, 0.0, -1, -15.0])
def test_resolve_busy_timeout_rejects_non_positive(bad_timeout):
    with pytest.raises(ValueError):
        resolve_busy_timeout_ms(bad_timeout)


@pytest.mark.parametrize("bad_timeout", [float("inf"), float("nan")])
def test_resolve_busy_timeout_rejects_non_finite(bad_timeout):
    with pytest.raises(ValueError):
        resolve_busy_timeout_ms(bad_timeout)


@pytest.mark.parametrize("bad_timeout", ["15", None, True, [15]])
def test_resolve_busy_timeout_rejects_non_numeric(bad_timeout):
    with pytest.raises(TypeError):
        resolve_busy_timeout_ms(bad_timeout)


# ── apply_busy_timeout ─────────────────────────────────────────────────────────


def test_apply_busy_timeout_sets_pragma(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        applied = apply_busy_timeout(conn, 20.0)
        assert applied == 20000
        assert _read_busy_timeout(conn) == 20000
    finally:
        conn.close()


def test_apply_busy_timeout_rejects_invalid_timeout(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(ValueError):
            apply_busy_timeout(conn, 0)
    finally:
        conn.close()


# ── create_connection ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("timeout, expected_ms", [(1.0, 1000), (30.0, 30000)])
def test_create_connection_honours_timeout_argument(db_path, timeout, expected_ms):
    """The caller's timeout must survive; previously it was clamped to 5000 ms."""
    conn = create_connection(db_path, timeout=timeout)
    try:
        assert _read_busy_timeout(conn) == expected_ms
    finally:
        conn.close()


def test_create_connection_default_timeout_matches_constant(db_path):
    conn = create_connection(db_path)
    try:
        expected = int(DEFAULT_SQLITE_TIMEOUT * 1000)
        assert _read_busy_timeout(conn) == expected
    finally:
        conn.close()


def test_create_connection_never_uses_the_old_hardcoded_value(db_path):
    """A 30 s request must not silently become the historical 5 s default."""
    conn = create_connection(db_path, timeout=30.0)
    try:
        assert _read_busy_timeout(conn) != 5000
    finally:
        conn.close()


def test_read_only_connection_honours_timeout(db_path):
    conn = create_connection(db_path, timeout=25.0, read_only=True)
    try:
        assert _read_busy_timeout(conn) == 25000
    finally:
        conn.close()


def test_create_connection_still_enables_wal_and_foreign_keys(db_path):
    conn = create_connection(db_path, timeout=10.0)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize("bad_timeout", [0, -5.0])
def test_create_connection_rejects_invalid_timeout(db_path, bad_timeout):
    with pytest.raises(ValueError):
        create_connection(db_path, timeout=bad_timeout)


def test_create_connection_validates_before_creating_directories(tmp_path):
    """An invalid timeout must not leave a stray parent directory behind."""
    target = tmp_path / "not_created" / "db.sqlite"

    with pytest.raises(ValueError):
        create_connection(target, timeout=-1)

    assert not target.parent.exists()


# ── get_connection ─────────────────────────────────────────────────────────────


def test_get_connection_propagates_timeout(db_path):
    with get_connection(db_path, timeout=12.0) as conn:
        assert _read_busy_timeout(conn) == 12000


def test_get_connection_closes_connection(db_path):
    with get_connection(db_path, timeout=12.0) as conn:
        pass

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


# ── behavioural check ──────────────────────────────────────────────────────────


def test_busy_timeout_actually_delays_lock_failure(db_path):
    """A short timeout should surface as a lock error, a long one should not.

    The writer below holds an exclusive lock. A second connection configured
    with a sub-second busy timeout gives up, while one configured with a
    multi-second timeout would keep waiting -- which is exactly the behaviour
    the hardcoded pragma was suppressing.
    """
    holder = create_connection(db_path, timeout=10.0)
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("INSERT INTO sample (value) VALUES ('locked')")

    impatient = create_connection(db_path, timeout=0.2)
    try:
        assert _read_busy_timeout(impatient) == 200
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            impatient.execute("INSERT INTO sample (value) VALUES ('blocked')")
            impatient.commit()
    finally:
        impatient.close()
        holder.rollback()
        holder.close()
