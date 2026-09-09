"""Persist analyzed uploads atomically, then rebuild the disposable search index."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

from src.core.embedding_compatibility import get_embedding_model_metadata
from src.core.faiss_index import _INDEX_LOCK, build_index_from_matrix, save_index
from src.db.corpus_db import _connect, get_all_embeddings
from src.utils.filename import sanitize_filename


def persist_analyzed_documents(files, chunks, embeddings, *, owner, index_path):
    """Store new documents and vectors together; repeated uploads are idempotent.

    SQLite remains the source of truth if saving the derived index fails.
    A filename collision receives a content-hash suffix instead of overwriting
    another document. No input data is committed if vector validation fails.
    """
    added = {}
    now = datetime.now(timezone.utc).isoformat()
    with _INDEX_LOCK:
        # Reject a corpus built with a different model before committing new rows.
        get_all_embeddings()
        from src.core.embedding_model import get_embedding_model_info

        _, dimension = get_embedding_model_info()
        with _connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            next_id = conn.execute(
                "SELECT COALESCE(MAX(vector_id), -1) + 1 FROM chunks"
            ).fetchone()[0]
            for original, text_chunks in chunks.items():
                if not text_chunks:
                    continue
                matrix = np.asarray(embeddings[original], dtype=np.float32)
                if (
                    matrix.ndim != 2
                    or matrix.shape[0] != len(text_chunks)
                    or matrix.shape[1] != dimension
                    or not np.isfinite(matrix).all()
                ):
                    raise ValueError(f"Invalid embeddings for {original}")
                digest = sha256(files[original]).hexdigest()
                if conn.execute(
                    "SELECT 1 FROM documents WHERE file_hash = ?", (digest,)
                ).fetchone():
                    continue
                name = sanitize_filename(original)
                if conn.execute(
                    "SELECT 1 FROM documents WHERE filename = ?", (name,)
                ).fetchone():
                    path = Path(name)
                    name = f"{path.stem}-{digest}{path.suffix}"
                conn.execute(
                    "INSERT INTO documents (filename, file_hash, upload_date, owner) VALUES (?, ?, ?, ?)",
                    (name, digest, now, owner),
                )
                meta = get_embedding_model_metadata(matrix.shape[1], generated_at=now)
                rows = [
                    (
                        next_id + i,
                        name,
                        i,
                        str(chunk),
                        vector.tobytes(),
                        meta.model_identifier,
                        meta.model_version,
                        meta.dimension,
                        meta.normalization_strategy,
                        meta.generated_at,
                        meta.vector_schema_version,
                    )
                    for i, (chunk, vector) in enumerate(zip(text_chunks, matrix))
                ]
                conn.executemany(
                    """INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text,
                    embedding, model_identifier, model_version, embedding_dimension,
                    normalization_strategy, embedding_generated_at, vector_schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                next_id += len(rows)
                added[original] = name
        if added:
            matrix = get_all_embeddings()
            save_index(build_index_from_matrix(matrix), str(index_path))
    return added
