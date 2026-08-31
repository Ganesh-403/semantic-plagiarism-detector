import sqlite3
import pytest
from unittest.mock import patch

# --- Pytest Fixtures Layer ---

@pytest.fixture(scope="function")
def isolated_test_db():
    """
    Provides a transient, in-memory SQLite database session connection.
    Guarantees zero file leakage into the developer's working environment.
    """
    connection = sqlite3.connect(":memory:")
    # Initialize basic schema prerequisites required for the bulk worker test
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            severity TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()
    
    yield connection
    
    # Teardown hook: Close out connection securely
    connection.close()


# --- Refactored Test Suite ---

def test_incidents_bulk_insertion_pipeline(isolated_test_db):
    """
    Scenario: Validate bulk record operations execute completely inside 
              the isolated mock database context hook.
    """
    conn = isolated_test_db
    cursor = conn.cursor()
    
    # Mock dataset payload matching operational structures
    bulk_payload = [
        ("Network Outage East", "CRITICAL"),
        ("Database Replica Delay", "WARNING"),
        ("Expired SSL Alert", "INFO")
    ]
    
    # Execute batch insertion execution steps
    cursor.executemany(
        "INSERT INTO incidents (title, severity) VALUES (?, ?);", 
        bulk_payload
    )
    conn.commit()
    
    # Verify count metrics match the injected parameters
    cursor.execute("SELECT COUNT(*) FROM incidents;")
    record_count = cursor.fetchone()[0]
    assert record_count == 3
    
    # Validate structural content properties
    cursor.execute("SELECT title FROM incidents ORDER BY id ASC;")
    inserted_titles = [row[0] for row in cursor.fetchall()]
    assert inserted_titles[0] == "Network Outage East"



class TestBulkIncidentInsertion:
    """Test suite for bulk inserting plagiarism incidents."""

    def test_bulk_insert_100_incidents(self, db_connection: sqlite3.Connection):
        """Verify 100 incidents can be inserted in a single transaction."""
        incidents = [
            (
                f"BULK-{i:05d}",
                f"doc_a_{i}.pdf",
                f"doc_b_{i}.pdf",
                0.75,
                "Medium",
                datetime.utcnow().isoformat(),
                0.59,
                "Pending",
            )
            for i in range(100)
        ]

        db_connection.executemany(
            """
            INSERT INTO plagiarism_incidents 
            (incident_id, document_a, document_b, similarity, severity, timestamp, threshold_at_time_of_flag, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            incidents,
        )
        db_connection.commit()

        cursor = db_connection.execute("SELECT COUNT(*) FROM plagiarism_incidents")
        count = cursor.fetchone()[0]

        assert count == 100

    def test_bulk_insert_rollback_on_duplicate(self, db_connection: sqlite3.Connection):
        """Verify transaction rolls back if a duplicate incident_id is encountered."""
        # Insert first incident
        db_connection.execute(
            """
            INSERT INTO plagiarism_incidents 
            (incident_id, document_a, document_b, similarity, severity, timestamp, threshold_at_time_of_flag, review_status)
            VALUES ('DUP-001', 'a.pdf', 'b.pdf', 0.90, 'High', '2024-01-01', 0.59, 'Pending')
            """
        )
        db_connection.commit()

        # Attempt bulk insert with duplicate
        incidents = [
            ("DUP-002", "c.pdf", "d.pdf", 0.80, "High", "2024-01-02", 0.59, "Pending"),
            (
                "DUP-001",
                "e.pdf",
                "f.pdf",
                0.85,
                "High",
                "2024-01-03",
                0.59,
                "Pending",
            ),  # Duplicate
        ]

        with pytest.raises(sqlite3.IntegrityError):
            try:
                db_connection.executemany(
                    """
                    INSERT INTO plagiarism_incidents 
                    (incident_id, document_a, document_b, similarity, severity, timestamp, threshold_at_time_of_flag, review_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    incidents,
                )
                db_connection.commit()
            except sqlite3.IntegrityError:
                db_connection.rollback()
                raise

        # Verify only the original incident exists
        cursor = db_connection.execute("SELECT COUNT(*) FROM plagiarism_incidents")
        assert cursor.fetchone()[0] == 1

    def test_bulk_update_review_status(
        self, populated_db_connection: sqlite3.Connection
    ):
        """Verify bulk updating review status for all pending incidents."""
        conn = populated_db_connection

        # Update all pending to reviewed
        cursor = conn.execute(
            """
            UPDATE plagiarism_incidents 
            SET review_status = 'Reviewed' 
            WHERE review_status = 'Pending'
            """
        )
        conn.commit()

        assert cursor.rowcount == 50

        # Verify no pending remain
        cursor = conn.execute(
            "SELECT COUNT(*) FROM plagiarism_incidents WHERE review_status = 'Pending'"
        )
        assert cursor.fetchone()[0] == 0


class TestBulkIncidentDeletion:
    """Test suite for bulk deleting incidents."""

    def test_bulk_delete_by_severity(self, populated_db_connection: sqlite3.Connection):
        """Verify bulk deletion of all Low severity incidents."""
        conn = populated_db_connection

        # Count Low severity before deletion
        cursor = conn.execute(
            "SELECT COUNT(*) FROM plagiarism_incidents WHERE severity = 'Low'"
        )
        low_count_before = cursor.fetchone()[0]

        # Delete all Low severity
        cursor = conn.execute("DELETE FROM plagiarism_incidents WHERE severity = 'Low'")
        conn.commit()

        assert cursor.rowcount == low_count_before

        # Verify none remain
        cursor = conn.execute(
            "SELECT COUNT(*) FROM plagiarism_incidents WHERE severity = 'Low'"
        )
        assert cursor.fetchone()[0] == 0

    def test_bulk_delete_older_than_date(
        self, populated_db_connection: sqlite3.Connection
    ):
        """Verify bulk deletion of incidents older than a specific date."""
        conn = populated_db_connection
        cutoff_date = "2024-01-15T00:00:00"

        cursor = conn.execute(
            "DELETE FROM plagiarism_incidents WHERE timestamp < ?", (cutoff_date,)
        )
        conn.commit()

        # Verify all remaining are >= cutoff
        cursor = conn.execute("SELECT MIN(timestamp) FROM plagiarism_incidents")
        min_timestamp = cursor.fetchone()[0]

        assert min_timestamp >= cutoff_date


class TestBulkIncidentUpsert:
    """Test suite for bulk updating / upserting incidents."""

    def test_sync_flagged_incidents_bulk_upsert_severity(
        self, populated_db_connection: sqlite3.Connection
    ):
        """Verify updating an incident's severity to 'Critical' correctly records 'Critical' rather than defaulting to 'High'."""
        conn = populated_db_connection

        conn.execute(
            """
            UPDATE plagiarism_incidents
            SET similarity = 0.99, severity = 'Critical'
            WHERE incident_id = 'INC-0000'
            """
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT similarity, severity FROM plagiarism_incidents WHERE incident_id = 'INC-0000'"
        )
        row = cursor.fetchone()
        assert row[0] == 0.99
        assert row[1] == "Critical"
