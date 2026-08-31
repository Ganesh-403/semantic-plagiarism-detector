import sqlite3


def get_recent_audit_events(db_connection: sqlite3.Connection):
    """
    Retrieves recent audit events using sqlite3.Row and dict conversion.
    """
    db_connection.row_factory = sqlite3.Row
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM audit_events ORDER BY timestamp DESC LIMIT 50")
    rows = cursor.fetchall()
    return [dict(row) for row in rows]


def get_security_audit_logs(db_connection: sqlite3.Connection):
    """
    Refactored to use sqlite3.Row and dict mapping, matching get_recent_audit_events.
    """
    db_connection.row_factory = sqlite3.Row
    cursor = db_connection.cursor()
    cursor.execute(
        "SELECT id, event_type, user_id, timestamp, details FROM security_logs ORDER BY timestamp DESC LIMIT 50"
    )
    rows = cursor.fetchall()
    return [dict(row) for row in rows]
