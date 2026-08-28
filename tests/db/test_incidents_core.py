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
tests/db/test_incidents_core.py
-------------------------------
Core CRUD operation tests for the plagiarism incidents database.

Utilizes the centralized db_connection pytest fixture (Issue #2725).
"""

import sqlite3
from datetime import datetime


class TestIncidentCreation:
    """Test suite for creating individual incidents."""

    def test_create_single_incident(self, db_connection: sqlite3.Connection):
        """Verify a single incident can be created and retrieved."""
        incident_id = "TEST-001"
        doc_a = "alice_essay.pdf"
        doc_b = "bob_essay.pdf"
        similarity = 0.92
        severity = "High"
        timestamp = datetime.utcnow().isoformat()

        db_connection.execute(
            """
            INSERT INTO plagiarism_incidents
            (incident_id, document_a, document_b, similarity, severity, timestamp, threshold_at_time_of_flag, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                doc_a,
                doc_b,
                similarity,
                severity,
                timestamp,
                0.59,
                "Pending",
            ),
        )
        db_connection.commit()

        cursor = db_connection.execute(
            "SELECT * FROM plagiarism_incidents WHERE incident_id = ?", (incident_id,)
        )
        row = cursor.fetchone()

        assert row is not None
        assert row["document_a"] == doc_a
        assert row["document_b"] == doc_b
        assert row["similarity"] == similarity
        assert row["severity"] == severity

    def test_create_incident_with_null_threshold(
        self, db_connection: sqlite3.Connection
    ):
        """Verify incident creation handles NULL threshold gracefully."""
        db_connection.execute(
            """
            INSERT INTO plagiarism_incidents
            (incident_id, document_a, document_b, similarity, severity, timestamp, threshold_at_time_of_flag, review_status)
            VALUES ('NULL-001', 'a.pdf', 'b.pdf', 0.80, 'High', '2024-01-01', NULL, 'Pending')
            """
        )
        db_connection.commit()

        cursor = db_connection.execute(
            "SELECT threshold_at_time_of_flag FROM plagiarism_incidents WHERE incident_id = 'NULL-001'"
        )
        row = cursor.fetchone()

        assert row["threshold_at_time_of_flag"] is None


class TestIncidentRetrieval:
    """Test suite for retrieving and querying incidents."""

    def test_get_incidents_by_document(
        self, populated_db_connection: sqlite3.Connection
    ):
        """Verify retrieving all incidents involving a specific document."""
        conn = populated_db_connection

        # Insert a specific incident to query
        conn.execute(
            """
            INSERT INTO plagiarism_incidents
            (incident_id, document_a, document_b, similarity, severity, timestamp, threshold_at_time_of_flag, review_status)
            VALUES ('QUERY-001', 'target_doc.pdf', 'other.pdf', 0.99, 'High', '2024-02-01', 0.59, 'Pending')
            """
        )
        conn.commit()

        cursor = conn.execute(
            """
            SELECT * FROM plagiarism_incidents
            WHERE document_a = ? OR document_b = ?
            """,
            ("target-doc.pdf", "target-doc.pdf"),
        )
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0]["incident_id"] == "QUERY-001"

    def test_get_high_severity_incidents(
        self, populated_db_connection: sqlite3.Connection
    ):
        """Verify filtering by High severity returns correct subset."""
        conn = populated_db_connection

        cursor = conn.execute(
            "SELECT COUNT(*) FROM plagiarism_incidents WHERE severity = 'High'"
        )
        high_count = cursor.fetchone()[0]

        # Based on populated_db_connection logic (sim >= 0.80 is High)
        # i ranges 0-49, sim = 0.50 + i*0.01. High when sim >= 0.80 -> i >= 30
        # So 20 incidents should be High (i=30 to 49)
        assert high_count == 20
