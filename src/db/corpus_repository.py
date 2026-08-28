"""Repository helpers for filtering corpus documents by metadata.

Issue #3414: provide parameterized metadata filtering for CorpusRepository.
"""
from __future__ import annotations

from typing import Any

from src.db import corpus_db


class CorpusRepository:
    """Data-access facade for corpus document metadata."""

    def __init__(self, db_path=None) -> None:
        if db_path is not None:
            corpus_db.configure_db_path(db_path)

    def get_documents_by_metadata(
        self,
        class_section: str | None = None,
        student_name: str | None = None,
        assignment_title: str | None = None,
        owner: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return active documents matching any supplied metadata filters.

        Every predicate is parameterized; omitted filters do not contribute a
        WHERE clause. Results are dictionaries so callers can consume the
        complete document metadata without coupling to a schema dataclass.
        """
        filters: list[str] = []
        params: list[str] = []

        for column, value in (
            ("class_section", class_section),
            ("student_name", student_name),
            ("assignment_title", assignment_title),
            ("owner", owner),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                params.append(value)

        query = """
            SELECT filename,
                   file_hash,
                   upload_date,
                   class_section,
                   student_name,
                   assignment_title,
                   pdf_author,
                   pdf_creation_date,
                   pdf_title,
                   tags,
                   detected_language,
                   owner
            FROM documents
        """
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY upload_date DESC"

        with corpus_db._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            columns = [column[0] for column in conn.description]

        return [dict(zip(columns, row)) for row in rows]
