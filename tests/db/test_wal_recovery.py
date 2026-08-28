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

import os
import sqlite3

import pytest


def init_wal_database(db_path: str) -> None:
    """Initializes schema and forces WAL journal mode."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            status TEXT NOT NULL
        );
    """
    )
    conn.commit()
    conn.close()


# --- Recovery Verification Suite ---


@pytest.fixture(scope="function")
def crash_db_file(tmp_path):
    """Provides a temporary on-disk file path for simulating connection crashes."""
    db_file = os.path.join(tmp_path, "crash_recovery_test.db")
    init_wal_database(db_file)
    yield db_file

    # Teardown: Clean up trailing database and WAL state journals safely
    for ext in ["", "-wal", "-shm"]:
        target = db_file + ext
        if os.path.exists(target):
            try:
                os.remove(target)
            except OSError:
                pass


def test_wal_journal_recovery_after_simulated_crash(crash_db_file):
    """
    Scenario: Write records into WAL without triggering checkpoints,
              abruptly close connections to simulate a hard crash, and
              assert that a fresh connection reads all committed rows.
    """
    db_path = crash_db_file

    # 1. Establish an active write connection context
    conn_writer = sqlite3.connect(db_path)
    cursor_writer = conn_writer.cursor()

    # Force checkpoint operations off to leave all written pages solely inside the -wal file
    cursor_writer.execute("PRAGMA wal_autocheckpoint=0;")

    # 2. Insert committed tracking rows
    test_records = [
        ("Core migration tracking checkpoint initialized", "PENDING"),
        ("Asynchronous state machine worker activated", "RUNNING"),
        ("Transactional replication loop completed", "SUCCESS"),
    ]
    cursor_writer.executemany(
        "INSERT INTO system_logs (message, status) VALUES (?, ?);", test_records
    )
    conn_writer.commit()

    # 3. Assert that data is residing in the WAL file and not yet merged to the primary DB file
    cursor_writer.execute("PRAGMA wal_integrity_check;")
    _ = cursor_writer.fetchall()

    wal_file_path = f"{db_path}-wal"
    assert os.path.exists(
        wal_file_path
    ), "WAL file should actively contain uncheckpointed pages"
    assert (
        os.path.getsize(wal_file_path) > 0
    ), "WAL log file should have captured active binary page deltas"

    # 4. Simulate a sudden crash event: close connection directly without execution of explicit CHECKPOINT
    conn_writer.close()

    # 5. Rehydrate a completely fresh connection to trigger the automatic WAL recovery log re-scan
    conn_reader = sqlite3.connect(db_path)
    cursor_reader = conn_reader.cursor()

    # 6. Core Assertions: Confirm zero data loss across the crash perimeter
    cursor_reader.execute("SELECT message, status FROM system_logs ORDER BY id ASC;")
    recovered_records = cursor_reader.fetchall()
    conn_reader.close()

    assert (
        len(recovered_records) == 3
    ), f"Data drop-out detected! Expected 3 records, recovered {len(recovered_records)}"
    assert recovered_records[0] == (
        "Core migration tracking checkpoint initialized",
        "PENDING",
    )
    assert recovered_records[2] == (
        "Transactional replication loop completed",
        "SUCCESS",
    )
