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
tests/db/test_incidents_self_plagiarism_logging.py
--------------------------------------------------
Regression tests for the self-plagiarism suppression path in
``sync_flagged_incidents`` (issue #3786).

``src/db/incidents.py`` called ``logger.info(...)`` when it skipped a pair
belonging to one student, but the module never imported ``logging`` and never
defined ``logger``. The only ``except`` guarding that block catches
``sqlite3.Error``, which does not catch ``NameError``, so the exception escaped
the function.

That made the failure disproportionate: passing
``allow_self_plagiarism_flags=False`` — the whole point of the option — aborted
the entire sync, so every *other* flag in the batch was lost too, rather than
just the one pair being skipped.

These tests cover the logging call itself and, more importantly, that the rest
of the batch still lands.
"""

import logging
import sqlite3

import pytest

from src.db import incidents as incidents_module
from src.db.incidents import get_all_incidents, log_incident, sync_flagged_incidents

DOCUMENTS = [
    # (filename, file_hash, upload_date, class_section, student_name, title)
    ("alice_draft1.pdf", "h1", "2026-08-01", "CS101", "Alice Smith", "Essay 1"),
    ("alice_draft2.pdf", "h2", "2026-08-02", "CS101", "Alice Smith", "Essay 1"),
    ("bob_essay.pdf", "h3", "2026-08-03", "CS101", "Bob Jones", "Essay 1"),
    ("carol_essay.pdf", "h4", "2026-08-04", "CS101", "Carol White", "Essay 1"),
]


@pytest.fixture
def db(mock_db):
    """A corpus DB seeded with two same-student drafts and two other students."""
    with sqlite3.connect(mock_db) as conn:
        conn.executemany(
            """
            INSERT INTO documents (
                filename, file_hash, upload_date,
                class_section, student_name, assignment_title
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            DOCUMENTS,
        )
        conn.commit()
    return mock_db


def _flag(doc_a, doc_b, similarity=0.95):
    return {
        "doc_a": doc_a,
        "doc_b": doc_b,
        "similarity": similarity,
        "severity": "High",
    }


SELF_PAIR = _flag("alice_draft1.pdf", "alice_draft2.pdf")
CROSS_PAIR = _flag("bob_essay.pdf", "carol_essay.pdf", similarity=0.88)


# ── the regression itself ──────────────────────────────────────────────────────


def test_module_defines_a_logger():
    """`logger` was referenced but never defined anywhere in the module."""
    assert isinstance(incidents_module.logger, logging.Logger)


def test_logger_is_namespaced_to_the_module():
    """Matches the convention in the rest of src/db."""
    assert incidents_module.logger.name == "src.db.incidents"


def test_suppressing_a_self_plagiarism_pair_does_not_raise(db):
    """This raised NameError before the logger existed."""
    results = sync_flagged_incidents(
        [SELF_PAIR], db_path=db, allow_self_plagiarism_flags=False
    )

    assert results == []


def test_suppression_logs_the_skipped_pair(db, caplog):
    with caplog.at_level(logging.INFO, logger="src.db.incidents"):
        sync_flagged_incidents(
            [SELF_PAIR], db_path=db, allow_self_plagiarism_flags=False
        )

    assert "alice_draft1.pdf" in caplog.text
    assert "alice_draft2.pdf" in caplog.text
    assert "Alice Smith" in caplog.text


def test_suppression_log_is_info_level(db, caplog):
    with caplog.at_level(logging.INFO, logger="src.db.incidents"):
        sync_flagged_incidents(
            [SELF_PAIR], db_path=db, allow_self_plagiarism_flags=False
        )

    assert [r.levelno for r in caplog.records] == [logging.INFO]


def test_nothing_is_logged_when_no_pair_is_skipped(db, caplog):
    with caplog.at_level(logging.INFO, logger="src.db.incidents"):
        sync_flagged_incidents(
            [CROSS_PAIR], db_path=db, allow_self_plagiarism_flags=False
        )

    assert "Skipping self-plagiarism" not in caplog.text


# ── the batch must survive the skip ────────────────────────────────────────────


def test_remaining_flags_still_sync_after_a_skip(db):
    """The crash lost the whole batch, not just the suppressed pair."""
    results = sync_flagged_incidents(
        [SELF_PAIR, CROSS_PAIR], db_path=db, allow_self_plagiarism_flags=False
    )

    assert len(results) == 1
    assert {results[0].document_a, results[0].document_b} == {
        "bob_essay.pdf",
        "carol_essay.pdf",
    }


def test_a_skip_in_the_middle_does_not_drop_later_flags(db):
    """Ordering must not matter — the skip is a `continue`, not a bail-out."""
    other = _flag("alice_draft1.pdf", "bob_essay.pdf", similarity=0.80)

    results = sync_flagged_incidents(
        [other, SELF_PAIR, CROSS_PAIR],
        db_path=db,
        allow_self_plagiarism_flags=False,
    )

    assert len(results) == 2


def test_only_the_suppressed_pair_is_absent_from_the_database(db):
    sync_flagged_incidents(
        [SELF_PAIR, CROSS_PAIR], db_path=db, allow_self_plagiarism_flags=False
    )

    stored = get_all_incidents(db_path=db)
    pairs = {frozenset((i.document_a, i.document_b)) for i in stored}

    assert frozenset(("bob_essay.pdf", "carol_essay.pdf")) in pairs
    assert frozenset(("alice_draft1.pdf", "alice_draft2.pdf")) not in pairs


# ── the flag must still be honoured in both directions ─────────────────────────


def test_same_student_pair_is_kept_when_suppression_is_off(db):
    results = sync_flagged_incidents(
        [SELF_PAIR], db_path=db, allow_self_plagiarism_flags=True
    )

    assert len(results) == 1


def test_different_students_are_never_suppressed(db):
    results = sync_flagged_incidents(
        [CROSS_PAIR], db_path=db, allow_self_plagiarism_flags=False
    )

    assert len(results) == 1


def test_documents_with_no_student_name_are_not_suppressed(mock_db):
    """A NULL student_name must not compare equal to another NULL."""
    with sqlite3.connect(mock_db) as conn:
        conn.executemany(
            """
            INSERT INTO documents (
                filename, file_hash, upload_date,
                class_section, student_name, assignment_title
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("anon_a.pdf", "ha", "2026-08-01", "CS101", None, "Essay"),
                ("anon_b.pdf", "hb", "2026-08-02", "CS101", None, "Essay"),
            ],
        )
        conn.commit()

    results = sync_flagged_incidents(
        [_flag("anon_a.pdf", "anon_b.pdf")],
        db_path=mock_db,
        allow_self_plagiarism_flags=False,
    )

    assert len(results) == 1


# ── log_incident wraps the same path ───────────────────────────────────────────


def test_log_incident_returns_none_for_a_suppressed_pair(db):
    assert (
        log_incident(SELF_PAIR, db_path=db, allow_self_plagiarism_flags=False) is None
    )


def test_log_incident_still_records_a_cross_student_pair(db):
    result = log_incident(CROSS_PAIR, db_path=db, allow_self_plagiarism_flags=False)

    assert result is not None
    assert {result.document_a, result.document_b} == {
        "bob_essay.pdf",
        "carol_essay.pdf",
    }
    assert get_all_incidents(db_path=db)
