"""Controlled and resumable embedding migration workflow."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclassfrom datetime import datetime, timezone
from typing import Any

import numpy as np

from src.core.embedding_compatibility import get_active_embedding_metadata
from src.core.embedding_model import embed_chunks
from src.core.faiss_index import build_index_from_matrix, save_index
from src.db.corpus_db import (
    _connect,
    get_document_embeddings_for_migration,
    get_documents_with_embeddings,
    update_document_embeddings,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MigrationResult:
    """Summary of one migration run."""

    migration_id: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    status: str
    failures: dict[str, str]


def _ensure_migration_table() -> None:
    """Create the resumable migration state table if needed."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_migrations (
                migration_id TEXT PRIMARY KEY,
                source_model_identifier TEXT,
                source_model_version TEXT,
                target_model_identifier TEXT NOT NULL,
                target_model_version TEXT NOT NULL,
                target_dimension INTEGER NOT NULL,
                status TEXT NOT NULL,
                total_documents INTEGER NOT NULL DEFAULT 0,
                processed_documents INTEGER NOT NULL DEFAULT 0,
                failed_documents INTEGER NOT NULL DEFAULT 0,
                failures TEXT NOT NULL DEFAULT '{}',
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def _migration_id() -> str:
    """Generate a stable migration identifier."""
    return datetime.now(timezone.utc).strftime(
        "embedding_%Y%m%d%H%M%S"
    )


def start_migration(
    migration_id: str | None = None,
) -> str:
    """Create a resumable migration record."""
    _ensure_migration_table()

    metadata = get_active_embedding_metadata()
    migration_id = migration_id or _migration_id()
    documents = get_documents_with_embeddings()
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO embedding_migrations (
                migration_id,
                target_model_identifier,
                target_model_version,
                target_dimension,
                status,
                total_documents,
                started_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
            """,
            (
                migration_id,
                metadata.model_identifier,
                metadata.model_version,
                metadata.dimension,
                len(documents),
                now,
                now,
            ),
        )

    return migration_id


def migrate_embeddings(
    migration_id: str,
    *,
    stop_on_error: bool = False,
) -> MigrationResult:
    """Resume a migration from its last successful document."""
    _ensure_migration_table()

    with _connect() as conn:
        migration = conn.execute(
            """
            SELECT *
            FROM embedding_migrations
            WHERE migration_id = ?
            """,
            (migration_id,),
        ).fetchone()

    if migration is None:
        raise ValueError(f"Unknown embedding migration: {migration_id}")

    metadata = get_active_embedding_metadata()
    documents = get_documents_with_embeddings()

    failures: dict[str, str] = {}

    with _connect() as conn:
        processed_before = int(migration["processed_documents"])

    # Resume by checking the target metadata in the database instead of
    # relying only on a counter. This makes interrupted migrations recoverable.
    pending_documents = []

    with _connect() as conn:
        for filename in documents:
            row = conn.execute(
                """
                SELECT model_identifier, model_version,
                       embedding_dimension, normalization_strategy,
                       vector_schema_version
                FROM chunks
                WHERE filename = ?
                LIMIT 1
                """,
                (filename,),
            ).fetchone()

            if (
                row
                and row["model_identifier"] == metadata.model_identifier
                and row["model_version"] == metadata.model_version
                and row["embedding_dimension"] == metadata.dimension
                and row["normalization_strategy"]
                == metadata.normalization_strategy
                and row["vector_schema_version"]
                == metadata.vector_schema_version
            ):
                continue

            pending_documents.append(filename)

    processed = len(documents) - len(pending_documents)

    for filename in pending_documents:
        try:
            texts, _old_embeddings = get_document_embeddings_for_migration(
                filename
            )

            if not texts:
                continue

            new_embeddings = embed_chunks(texts)

            generated_at = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )

            update_document_embeddings(
                filename,
                new_embeddings,
                model_identifier=metadata.model_identifier,
                model_version=metadata.model_version,
                normalization_strategy=metadata.normalization_strategy,
                vector_schema_version=metadata.vector_schema_version,
                generated_at=generated_at,
            )

            processed += 1

            with _connect() as conn:
                conn.execute(
                    """
                    UPDATE embedding_migrations
                    SET
                        processed_documents = ?,
                        updated_at = ?
                    WHERE migration_id = ?
                    """,
                    (
                        processed,
                        datetime.now(timezone.utc).isoformat(),
                        migration_id,
                    ),
                )

        except Exception as exc:
            failures[filename] = str(exc)
            logger.exception(
                "Embedding migration failed for document %s",
                filename,
            )

            if stop_on_error:
                break

    status = (
        "completed"
        if not failures and processed >= len(documents)
        else "partial"
        if processed > 0
        else "failed"
    )

    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            """
            UPDATE embedding_migrations
            SET
                status = ?,
                processed_documents = ?,
                failed_documents = ?,
                failures = ?,
                updated_at = ?
            WHERE migration_id = ?
            """,
            (
                status,
                processed,
                len(failures),
                json.dumps(failures),
                now,
                migration_id,            ),
        )

    return MigrationResult(
        migration_id=migration_id,
        total_documents=len(documents),
        processed_documents=processed,
        failed_documents=len(failures),
        status=status,
        failures=failures,
    )


def rebuild_index_after_migration(
    embeddings: dict[str, np.ndarray],
    chunked_docs: dict[str, list[str]],
    index_path: str,
) -> None:
    """Rebuild FAISS only after all vectors use the active embedding schema."""
    metadata = get_active_embedding_metadata()

    for document_name, vectors in embeddings.items():
        if vectors.size == 0:
            continue

        if vectors.shape[1] != metadata.dimension:
            raise ValueError(
                f"{document_name} has dimension {vectors.shape[1]}, "
                f"expected {metadata.dimension}."
            )

    matrix = (
        np.vstack(
            [
                vectors
                for vectors in embeddings.values()
                if vectors.size > 0
            ]
        )
        if any(vectors.size > 0 for vectors in embeddings.values())
        else np.empty((0, metadata.dimension), dtype=np.float32)
    )

    index = build_index_from_matrix(matrix)
    save_index(index, index_path)