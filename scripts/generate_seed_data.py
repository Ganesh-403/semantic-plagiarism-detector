#!/usr/bin/env python3
"""
scripts/generate_seed_data.py
-----------------------------
Generate seed data for development and testing environments.

This script populates the database with sample users, documents, and
plagiarism incidents to facilitate local development and demonstration.

Usage:
    # Generate seed data and write to database
    python scripts/generate_seed_data.py

    # Preview what would be inserted without modifying database (Issue #2020)
    python scripts/generate_seed_data.py --dry-run

    # Specify custom seed directory
    python scripts/generate_seed_data.py --seed-dir /path/to/seeds

Acceptance Criteria (Issue #2020):
- Added --dry-run flag to preview operations without DB writes
- Logs all operations that would be performed in dry-run mode
- Exits successfully after dry-run preview

Examples:
    >>> python scripts/generate_seed_data.py --dry-run
    [DRY RUN] Would create 2 seed users
    [DRY RUN] Would upload 5 sample documents
    [DRY RUN] Would create 3 plagiarism incidents
    [DRY RUN] No database modifications made
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.auth import add_user, init_db
from src.db.corpus_db import add_document, init_corpus_db
from src.db.incidents import init_incident_db, sync_flagged_incidents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Seed data definitions
SEED_USERS = [
    {"username": "admin", "password_env": "ADMIN_BOOTSTRAP_PASSWORD", "role": "admin"},  # pragma: allowlist secret
    {"username": "teacher", "password_env": "SEED_TEACHER_PASSWORD", "role": "teacher"},  # pragma: allowlist secret
]

SEED_DOCUMENTS = [
    {
        "filename": "sample_essay_1.txt",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "class_section": "CS101",
        "student_name": "Alice Johnson",
    },
    {
        "filename": "sample_essay_2.txt",
        "content": "Artificial intelligence includes machine learning, which allows computers to learn patterns from data.",
        "class_section": "CS101",
        "student_name": "Bob Smith",
    },
    {
        "filename": "sample_essay_3.txt",
        "content": "Deep learning is a specialized form of machine learning using neural networks with multiple layers.",
        "class_section": "CS101",
        "student_name": "Charlie Brown",
    },
]

SEED_INCIDENTS = [
    {
        "document_a": "sample_essay_1.txt",
        "document_b": "sample_essay_2.txt",
        "similarity": 0.85,
        "severity": "High",
    },
    {
        "document_a": "sample_essay_2.txt",
        "document_b": "sample_essay_3.txt",
        "similarity": 0.72,
        "severity": "Medium",
    },
]


def record_plagiarism_incident(document_a, document_b, similarity, severity):
    """Store a demonstration incident through the active incident repository."""
    return sync_flagged_incidents([{
        "doc_a": document_a, "doc_b": document_b,
        "similarity": similarity, "severity": severity,
    }])


def build_seed_index(index_path: Path) -> None:
    """Embed previously unindexed samples and rebuild the real corpus index."""
    from src.core.embedding_model import embed_documents
    from src.core.text_chunking import chunk_documents
    from src.core.faiss_index import build_index_from_matrix, save_index
    from src.db.corpus_db import (
        get_document_chunks_count, get_embedding_count, add_chunks,
        get_all_embeddings, get_document_by_hash,
    )

    raw = {}
    for doc in SEED_DOCUMENTS:
        digest = hashlib.sha256(doc["content"].encode("utf-8")).hexdigest()
        filename = get_document_by_hash(digest)
        if filename and not get_document_chunks_count(filename):
            raw[filename] = doc["content"]
    if raw:
        chunks = chunk_documents(raw, min_words=0)
        embeddings = embed_documents(chunks)
        vector_id = get_embedding_count()
        rows = []
        for filename, vectors in embeddings.items():
            if len(vectors) != len(chunks[filename]):
                raise ValueError("Seed embedding/chunk count mismatch")
            for chunk_index, (chunk, vector) in enumerate(zip(chunks[filename], vectors)):
                rows.append((vector_id, filename, chunk_index, str(chunk), vector))
                vector_id += 1
        add_chunks(rows)
    vectors = get_all_embeddings()
    if not vectors.size:
        raise ValueError("Seed corpus contains no embeddings")
    save_index(build_index_from_matrix(vectors), str(index_path))


def generate_seed_data(seed_dir: Path, dry_run: bool = False) -> dict:
    """Generate seed data and optionally write to database.

    Args:
        seed_dir: Directory to store generated seed files
        dry_run: If True, preview operations without DB writes (Issue #2020)

    Returns:
        Dictionary with summary of operations performed or previewed
    """
    summary = {
        "users_created": 0,
        "documents_created": 0,
        "incidents_created": 0,
        "dry_run": dry_run,
        "errors": 0,
    }

    # Initialize databases (always needed to check schema)
    if not dry_run:
        logger.info("Initializing databases...")
        init_db()
        init_corpus_db()
        init_incident_db()
    else:
        logger.info("[DRY RUN] Skipping database initialization")

    # Create seed directory
    seed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create seed users
    logger.info(
        f"{'[DRY RUN] Would create' if dry_run else 'Creating'} {len(SEED_USERS)} seed users..."
    )
    for user_data in SEED_USERS:
        if dry_run:
            logger.info(
                f"[DRY RUN] Would create user: {user_data['username']} "
                f"(role: {user_data['role']})"
            )
            summary["users_created"] += 1
        else:
            password = os.environ.get(user_data["password_env"])
            if not password:
                logger.info("Skipping %s; set %s to create this demo account",
                            user_data["username"], user_data["password_env"])
                continue
            try:
                add_user(
                    username=user_data["username"],
                    password=password,
                    role=user_data["role"],
                )
                summary["users_created"] += 1
                logger.info(f"✓ Created user: {user_data['username']}")
            except ValueError as exc:
                if "already exists" not in str(exc):
                    summary["errors"] += 1
                logger.warning("Could not create user %s: %s", user_data["username"], exc)
            except Exception:
                summary["errors"] += 1
                logger.exception("Could not create seed user %s", user_data["username"])

    # 2. Create seed documents
    logger.info(
        f"{'[DRY RUN] Would upload' if dry_run else 'Uploading'} {len(SEED_DOCUMENTS)} sample documents..."
    )
    for doc_data in SEED_DOCUMENTS:
        filename = doc_data["filename"]
        content = doc_data["content"]
        file_path = seed_dir / filename

        if dry_run:
            logger.info(
                f"[DRY RUN] Would create document: {filename} "
                f"({len(content)} chars, student: {doc_data['student_name']})"
            )
            summary["documents_created"] += 1
        else:
            # Write file to disk
            file_path.write_text(content, encoding="utf-8")

            # Calculate file hash
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Add to database
            try:
                add_document(
                    filename=filename,
                    file_hash=file_hash,
                    class_section=doc_data.get("class_section"),
                    student_name=doc_data.get("student_name"),
                )
                summary["documents_created"] += 1
                logger.info(f"✓ Created document: {filename}")
            except Exception as exc:
                summary["errors"] += 1
                logger.warning(f"Failed to create document {filename}: {exc}")

    # 3. Create seed incidents
    logger.info(
        f"{'[DRY RUN] Would create' if dry_run else 'Creating'} {len(SEED_INCIDENTS)} plagiarism incidents..."
    )
    for incident_data in SEED_INCIDENTS:
        if dry_run:
            logger.info(
                f"[DRY RUN] Would create incident: {incident_data['document_a']} <-> "
                f"{incident_data['document_b']} (similarity: {incident_data['similarity']:.1%})"
            )
            summary["incidents_created"] += 1
        else:
            try:
                record_plagiarism_incident(
                    document_a=incident_data["document_a"],
                    document_b=incident_data["document_b"],
                    similarity=incident_data["similarity"],
                    severity=incident_data["severity"],
                )
                summary["incidents_created"] += 1
                logger.info(
                    f"✓ Created incident: {incident_data['document_a']} <-> "
                    f"{incident_data['document_b']}"
                )
            except Exception as exc:
                summary["errors"] += 1
                logger.warning(f"Failed to create incident: {exc}")

    # Summary
    if dry_run:
        logger.info("=" * 70)
        logger.info("[DRY RUN] Summary:")
        logger.info(f"  - Would create {summary['users_created']} users")
        logger.info(f"  - Would upload {summary['documents_created']} documents")
        logger.info(f"  - Would create {summary['incidents_created']} incidents")
        logger.info("[DRY RUN] No database modifications were made")
        logger.info("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("Seed data generation complete:")
        logger.info(f"  - Created {summary['users_created']} users")
        logger.info(f"  - Uploaded {summary['documents_created']} documents")
        logger.info(f"  - Created {summary['incidents_created']} incidents")
        logger.info(f"  - Seed files stored in: {seed_dir}")
        logger.info("=" * 70)

    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate seed data for development and testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=ROOT_DIR / "data" / "seeds",
        help="Directory to store generated seed files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview seed data generation without modifying the database. "
            "Logs all operations that would be performed. (Issue #2020)"
        ),
    )

    parser.add_argument(
        "--state-dir", type=Path,
        help="Optional isolated directory for corpus.db, users.db and corpus.index. "
             "Without this option, populate the configured application databases.",
    )
    parser.add_argument(
        "--skip-index", action="store_true",
        help="Create metadata only, without downloading or running the embedding model.",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for seed data generation.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("Seed Data Generator")
    logger.info("=" * 70)
    logger.info(f"Seed directory: {args.seed_dir}")

    if args.dry_run:
        logger.info("[DRY RUN MODE] No database modifications will be made")
        logger.info("-" * 70)

    try:
        from src.core.app_config import FAISS_INDEX_PATH
        index_path = Path(FAISS_INDEX_PATH)
        if args.state_dir and not args.dry_run:
            from src.db import auth, corpus_db, incidents
            args.state_dir.mkdir(parents=True, exist_ok=True)
            auth.configure_db_path(args.state_dir / "users.db")
            corpus_db.configure_db_path(args.state_dir / "corpus.db")
            incidents.DEFAULT_DB_PATH = str(args.state_dir / "corpus.db")
            index_path = args.state_dir / "corpus.index"
        summary = generate_seed_data(
            seed_dir=args.seed_dir,
            dry_run=args.dry_run,
        )

        if summary["errors"]:
            logger.error("Seed generation had %s failed operations", summary["errors"])
            return 1
        if not args.dry_run and not args.skip_index:
            build_seed_index(index_path)
        logger.info("✓ Seed data generation completed successfully")
        return 0

    except Exception as exc:
        logger.error(f"✗ Seed data generation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
