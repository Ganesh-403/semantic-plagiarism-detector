"""Corpus-level exact and near-duplicate document detection.

The detector uses a normalized-text SHA-256 hash for exact duplicates and a
small MinHash signature for efficient near-duplicate detection. Duplicate
relationships are stored separately from plagiarism incidents so they remain
available for audit/reporting.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Iterable

from src.db import corpus_db


DEFAULT_NEAR_DUPLICATE_THRESHOLD = float(
    os.getenv("CORPUS_NEAR_DUPLICATE_THRESHOLD", "0.92")
)
MINHASH_SIZE = 64
SHINGLE_SIZE = 5


@dataclass(frozen=True)
class DocumentFingerprint:
    """Lightweight fingerprint used for corpus duplicate detection."""

    filename: str
    exact_hash: str
    signature: tuple[int, ...]
    token_count: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "exact_hash": self.exact_hash,
                "signature": list(self.signature),
                "token_count": self.token_count,
            },
            separators=(",", ":"),
        )


def _normalize_text(text: str) -> str:
    """Normalize text so formatting-only changes do not affect fingerprints."""
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> list[str]:
    """Return normalized word/number tokens."""
    return re.findall(r"\b\w+\b", _normalize_text(text), flags=re.UNICODE)


def _shingles(tokens: list[str]) -> Iterable[str]:
    """Generate fixed-size token shingles."""
    if len(tokens) < SHINGLE_SIZE:
        if tokens:
            yield " ".join(tokens)
        return

    for index in range(len(tokens) - SHINGLE_SIZE + 1):
        yield " ".join(tokens[index : index + SHINGLE_SIZE])


def _hash_shingle(shingle: str, seed: int) -> int:
    """Generate a deterministic 64-bit hash for one shingle."""
    payload = f"{seed}:{shingle}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="big",
    )


def build_fingerprint(
    filename: str,
    text: str,
) -> DocumentFingerprint:
    """Build an exact hash plus MinHash signature for a document."""
    normalized = _normalize_text(text)
    tokens = _tokens(normalized)

    exact_hash = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    shingles = list(_shingles(tokens))

    if not shingles:
        signature = (0,) * MINHASH_SIZE
    else:
        signature = tuple(
            min(
                _hash_shingle(shingle, seed)
                for shingle in shingles
            )
            for seed in range(MINHASH_SIZE)
        )

    return DocumentFingerprint(
        filename=filename,
        exact_hash=exact_hash,
        signature=signature,
        token_count=len(tokens),
    )


def estimate_similarity(
    first: DocumentFingerprint,
    second: DocumentFingerprint,
) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not first.signature or not second.signature:
        return 0.0

    if len(first.signature) != len(second.signature):
        return 0.0

    equal = sum(
        a == b
        for a, b in zip(first.signature, second.signature)
    )

    return equal / len(first.signature)


def classify_relationship(
    first: DocumentFingerprint,
    second: DocumentFingerprint,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> tuple[str | None, float]:
    """Classify two documents as exact, near-duplicate, or unrelated."""
    if first.exact_hash == second.exact_hash:
        return "exact_duplicate", 1.0

    similarity = estimate_similarity(first, second)

    if similarity >= threshold:
        return "near_duplicate", similarity

    return None, similarity


def _ensure_store() -> None:
    """Create the duplicate metadata tables if they do not yet exist."""
    with corpus_db._connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_fingerprints (
                filename TEXT PRIMARY KEY,
                exact_hash TEXT NOT NULL,
                minhash_signature TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_duplicate_relationships (
                relationship_id TEXT PRIMARY KEY,
                document_a TEXT NOT NULL,
                document_b TEXT NOT NULL,
                relationship_type TEXT NOT NULL
                    CHECK (
                        relationship_type IN (
                            'exact_duplicate',
                            'near_duplicate'
                        )
                    ),
                similarity REAL NOT NULL,
                family_id TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                UNIQUE(document_a, document_b)
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_a
            ON corpus_duplicate_relationships(document_a)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_b
            ON corpus_duplicate_relationships(document_b)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_family
            ON corpus_duplicate_relationships(family_id)
            """
        )


def _load_fingerprints() -> dict[str, DocumentFingerprint]:
    """Load all stored document fingerprints."""
    _ensure_store()

    with corpus_db._connect() as conn:
        rows = conn.execute(
            """
            SELECT
                filename,
                exact_hash,
                minhash_signature,
                token_count
            FROM document_fingerprints
            """
        ).fetchall()

    fingerprints: dict[str, DocumentFingerprint] = {}

    for row in rows:
        fingerprints[row[0]] = DocumentFingerprint(
            filename=row[0],
            exact_hash=row[1],
            signature=tuple(
                int(value)
                for value in json.loads(row[2])
            ),
            token_count=int(row[3]),
        )

    return fingerprints


def _store_fingerprint(
    fingerprint: DocumentFingerprint,
) -> None:
    """Persist or refresh a document fingerprint."""
    _ensure_store()

    with corpus_db._connect() as conn:
        conn.execute(
            """
            INSERT INTO document_fingerprints (
                filename,
                exact_hash,
                minhash_signature,
                token_count,
                created_at
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(filename) DO UPDATE SET
                exact_hash = excluded.exact_hash,
                minhash_signature = excluded.minhash_signature,
                token_count = excluded.token_count
            """,
            (
                fingerprint.filename,
                fingerprint.exact_hash,
                json.dumps(list(fingerprint.signature)),
                fingerprint.token_count,
            ),
        )


def _find_family_id(
    document_a: str,
    document_b: str,
) -> str:
    """Reuse an existing duplicate family where possible."""
    with corpus_db._connect() as conn:
        row = conn.execute(
            """
            SELECT family_id
            FROM corpus_duplicate_relationships
            WHERE document_a IN (?, ?)
               OR document_b IN (?, ?)
            ORDER BY detected_at ASC
            LIMIT 1
            """,
            (
                document_a,
                document_b,
                document_a,
                document_b,
            ),
        ).fetchone()

    return row[0] if row else str(uuid.uuid4())

def _store_relationship(
    document_a: str,
    document_b: str,
    relationship_type: str,
    similarity: float,
) -> None:
    """Persist one duplicate relationship for audit/reporting."""
    first, second = sorted((document_a, document_b))
    family_id = _find_family_id(first, second)

    with corpus_db._connect() as conn:
        conn.execute(
            """
            INSERT INTO corpus_duplicate_relationships (
                relationship_id,
                document_a,
                document_b,
                relationship_type,
                similarity,
                family_id,
                detected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(document_a, document_b) DO UPDATE SET
                relationship_type = excluded.relationship_type,
                similarity = excluded.similarity,
                family_id = excluded.family_id,
                detected_at = excluded.detected_at
            """,
            (
                str(uuid.uuid4()),
                first,
                second,
                relationship_type,
                round(float(similarity), 6),
                family_id,
            ),
        )


def detect_and_store_duplicates(
    documents: dict[str, str],
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> set[tuple[str, str]]:
    """Detect duplicate relationships and return pairs to exclude from scoring.

    ``documents`` contains the documents currently being processed. Previously
    stored fingerprints are also considered, so a new revision can be
    recognized against the existing corpus.
    """
    if not documents:
        return set()

    stored = _load_fingerprints()

    current = {
        filename: build_fingerprint(filename, text)
        for filename, text in documents.items()
    }

    # Persist the new/updated fingerprints first.
    for fingerprint in current.values():
        _store_fingerprint(fingerprint)

    all_fingerprints = dict(stored)
    all_fingerprints.update(current)

    names = list(all_fingerprints)
    excluded_pairs: set[tuple[str, str]] = set()

    for index, first_name in enumerate(names):
        first = all_fingerprints[first_name]

        for second_name in names[index + 1 :]:
            # Only exclude pairs where at least one document is part of this
            # processing run. Existing historical relationships do not cause
            # unrelated future documents to be skipped.
            if (
                first_name not in current
                and second_name not in current
            ):
                continue

            second = all_fingerprints[second_name]

            relationship_type, similarity = classify_relationship(
                first,
                second,
                threshold=threshold,
            )

            if relationship_type is None:
                continue

            _store_relationship(
                first_name,
                second_name,
                relationship_type,
                similarity,
            )

            if first_name in current and second_name in current:
                excluded_pairs.add(
                    tuple(sorted((first_name, second_name)))
                )

    return excluded_pairs


def get_duplicate_pairs(
    filenames: Iterable[str],
) -> set[tuple[str, str]]:
    """Return stored duplicate relationships touching the supplied documents."""
    names = list(filenames)

    if not names:
        return set()

    _ensure_store()

    placeholders = ",".join("?" for _ in names)

    with corpus_db._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT document_a, document_b
            FROM corpus_duplicate_relationships
            WHERE document_a IN ({placeholders})
               OR document_b IN ({placeholders})
            """,
            tuple(names) + tuple(names),
        ).fetchall()

    return {
        tuple(sorted((row[0], row[1])))
        for row in rows
    }


def get_duplicate_relationships(
    filenames: Iterable[str] | None = None,
) -> list[dict]:
    """Return duplicate relationships for audit/reporting."""
    _ensure_store()

    names = list(filenames or [])

    with corpus_db._connect() as conn:
        if names:
            placeholders = ",".join("?" for _ in names)
            rows = conn.execute(
                f"""
                SELECT
                    relationship_id,
                    document_a,
                    document_b,
                    relationship_type,
                    similarity,
                    family_id,
                    detected_at
                FROM corpus_duplicate_relationships
                WHERE document_a IN ({placeholders})
                   OR document_b IN ({placeholders})
                ORDER BY detected_at DESC
                """,
                tuple(names) + tuple(names),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    relationship_id,
                    document_a,
                    document_b,
                    relationship_type,
                    similarity,
                    family_id,
                    detected_at
                FROM corpus_duplicate_relationships
                ORDER BY detected_at DESC
                """
            ).fetchall()

    return [
        {
            "relationship_id": row[0],
            "document_a": row[1],
            "document_b": row[2],
            "relationship_type": row[3],
            "similarity": row[4],
            "family_id": row[5],
            "detected_at": row[6],
        }
        for row in rows
    ]