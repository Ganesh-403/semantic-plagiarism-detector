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
