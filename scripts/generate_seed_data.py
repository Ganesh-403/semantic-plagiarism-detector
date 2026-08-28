#!/usr/bin/env python3
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
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.auth import add_user, init_db
from src.db.corpus_db import add_document, init_corpus_db
from src.db.incidents import init_incident_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Seed data definitions
SEED_USERS = [
    {"username": "admin", "password": "admin123", "role": "admin"},
    {"username": "teacher", "password": "teacher123", "role": "teacher"},
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
            try:
                add_user(
                    username=user_data["username"],
                    password=user_data["password"],
                    role=user_data["role"],
                )
                summary["users_created"] += 1
                logger.info(f"✓ Created user: {user_data['username']}")
            except Exception as exc:
                logger.warning(f"User {user_data['username']} may already exist: {exc}")

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
                logger.warning(f"Document {filename} may already exist: {exc}")

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
                from src.db.incidents import record_plagiarism_incident

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
        summary = generate_seed_data(  # noqa: F841
            seed_dir=args.seed_dir,
            dry_run=args.dry_run,
        )

        logger.info("✓ Seed data generation completed successfully")
        return 0

    except Exception as exc:
        logger.error(f"✗ Seed data generation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
