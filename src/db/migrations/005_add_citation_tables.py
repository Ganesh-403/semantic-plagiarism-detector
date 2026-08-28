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
Migration 005: Add Bibliography Citation Tables
-----------------------------------------------
Creates the `citations` and `document_citations` tables to support
structural plagiarism analysis and citation graph extraction (Issue #1958).
"""


def migrate(connection):
    """Execute the migration SQL."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS citations (
            hash TEXT PRIMARY KEY,
            author TEXT,
            year TEXT,
            title TEXT,
            raw_text TEXT
        )
    """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_citations (
            doc_name TEXT NOT NULL,
            citation_hash TEXT NOT NULL,
            is_ghost INTEGER DEFAULT 0,
            PRIMARY KEY (doc_name, citation_hash),
            FOREIGN KEY (citation_hash) REFERENCES citations(hash) ON DELETE CASCADE
        )
    """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_doc_citations_doc
        ON document_citations(doc_name)
    """
    )
