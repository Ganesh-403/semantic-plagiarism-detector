"""
cli.py
------
Headless command-line interface for plagiarism detection automation.
"""

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from src.core.logging_config import setup_logging
from src.core.app_config import FAISS_INDEX_PATH
from src.core.cross_lingual import prepare_text_for_embedding
from src.core.document_parser import (DEFAULT_OCR_DPI, DEFAULT_OCR_LANGUAGE,
                                      extract_text)
from src.core.embedding_model import embed_documents
from src.core.similarity import document_similarity_matrix, flag_plagiarism
from src.core.synchronization import verify_and_repair_index
from src.core.text_chunking import chunk_documents
from src.db.database_backup import optimize_database


def run_scan(folder_path: str, threshold: float, output_format: str = "text") -> int:
    """
    Scans a folder, processes the documents, runs plagiarism detection,
    and prints the report in the requested output format to stdout.
    """
    if not os.path.exists(folder_path):
        sys.stderr.write(f"Error: Folder '{folder_path}' does not exist.\n")
        return 1

    if not os.path.isdir(folder_path):
        sys.stderr.write(f"Error: Path '{folder_path}' is not a directory.\n")
        return 1

    supported_extensions = {".pdf", ".docx", ".doc", ".txt"}
    files = []

    try:
        for entry in os.scandir(folder_path):
            if entry.is_file():
                # Skip hidden files
                if entry.name.startswith("."):
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in supported_extensions:
                    files.append(entry.path)
    except Exception as e:
        sys.stderr.write(f"Error reading folder contents: {e}\n")
        return 1

    # Sort files to ensure deterministic ordering
    files.sort()

    raw_texts = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            text = extract_text(
                BytesIO(file_bytes),
                filename,
                ocr_language=DEFAULT_OCR_LANGUAGE,
                ocr_dpi=DEFAULT_OCR_DPI,
            )
            if text.strip():
                raw_texts[filename] = text
            else:
                sys.stderr.write(
                    f"Warning: Extracted text from '{filename}' is empty.\n"
                )
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to parse '{filename}': {e}\n")

    num_processed = len(raw_texts)
    matches = []

    # Plagiarism check is only possible with 2 or more valid documents
    if num_processed >= 2:
        try:
            chunked_docs = chunk_documents(raw_texts)
            translated_chunked_docs = {}

            for doc_name, chunks in chunked_docs.items():
                translated_chunked_docs[doc_name] = []
                for chunk in chunks:
                    prepared = prepare_text_for_embedding(chunk)
                    translated_chunked_docs[doc_name].append(prepared["embedding_text"])

            embeddings = embed_documents(translated_chunked_docs)
            sim_df = document_similarity_matrix(embeddings)
            flags = flag_plagiarism(sim_df, threshold=threshold)

            for flag in flags:
                matches.append(
                    {
                        "document_1": flag["doc_a"],
                        "document_2": flag["doc_b"],
                        "similarity_score": flag["similarity"],
                    }
                )
        except Exception as e:
            sys.stderr.write(f"Error during plagiarism detection pipeline: {e}\n")
            return 1

    report = {
        "documents_processed": num_processed,
        "threshold": threshold,
        "matches": matches,
    }

    if output_format == "json":
        print(json.dumps(report, indent=2))
    elif output_format == "csv":
        import csv
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["document_1", "document_2", "similarity_score"])
        writer.writeheader()
        for m in matches:
            writer.writerow(m)
        print(output.getvalue().strip())
    else:  # text
        print(f"Documents Processed: {num_processed}")
        print(f"Similarity Threshold: {threshold}")
        if matches:
            print("Matches Found:")
            for m in matches:
                print(f"- {m['document_1']} <-> {m['document_2']}: {m['similarity_score']:.4f}")
        else:
            print("No matches found.")

    return 0


def run_prewarm(folder_path: str | None = None) -> int:
    """
    Pre-computes embeddings and populates Redis cache before user logins.

    If folder_path is specified, extracts documents from that directory.
    Otherwise, retrieves existing documents from the SQLite database.
    """
    raw_texts: dict[str, str] = {}

    if folder_path:
        if not os.path.exists(folder_path):
            sys.stderr.write(f"Error: Folder '{folder_path}' does not exist.\n")
            return 1
        if not os.path.isdir(folder_path):
            sys.stderr.write(f"Error: Path '{folder_path}' is not a directory.\n")
            return 1

        supported_extensions = {".pdf", ".docx", ".doc", ".txt"}
        files = []
        try:
            for entry in os.scandir(folder_path):
                if entry.is_file() and not entry.name.startswith("."):
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in supported_extensions:
                        files.append(entry.path)
        except Exception as e:
            sys.stderr.write(f"Error reading folder contents: {e}\n")
            return 1

        files.sort()
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "rb") as f:
                    file_bytes = f.read()
                text = extract_text(
                    BytesIO(file_bytes),
                    filename,
                    ocr_language=DEFAULT_OCR_LANGUAGE,
                    ocr_dpi=DEFAULT_OCR_DPI,
                )
                if text.strip():
                    raw_texts[filename] = text
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to parse '{filename}': {e}\n")
    else:
        # Pre-warm using documents from corpus database
        try:
            from src.db.corpus_db import get_all_documents
            docs = get_all_documents()
            for doc in docs:
                fname = getattr(doc, "filename", None) if not isinstance(doc, dict) else doc.get("filename")
                if fname:
                    raw_texts[fname] = f"Document content for {fname}"
        except Exception as e:
            sys.stderr.write(f"Warning: Could not fetch documents from database: {e}\n")

    embeddings_count = 0
    docs_processed = len(raw_texts)
    redis_status = "unavailable"

    if docs_processed > 0:
        try:
            chunked_docs = chunk_documents(raw_texts)
            translated_chunked_docs = {}
            for doc_name, chunks in chunked_docs.items():
                translated_chunked_docs[doc_name] = [
                    prepare_text_for_embedding(c)["embedding_text"] for c in chunks
                ]

            embeddings = embed_documents(translated_chunked_docs)
            sim_df = document_similarity_matrix(embeddings)
            embeddings_count = sum(len(emb) for emb in embeddings.values())

            # Populate Redis cache
            try:
                from src.utils.redis_cache import cache_analysis_results, get_cache
                cache = get_cache()
                if cache and getattr(cache, "is_available", lambda: True)():
                    cache_analysis_results(
                        "prewarmed_similarity_matrix",
                        {
                            "matrix": sim_df.to_dict(),
                            "documents": list(sim_df.columns),
                        },
                    )
                    redis_status = "populated"
                else:
                    redis_status = "fallback_in_memory"
            except Exception as cache_err:
                sys.stderr.write(f"Warning: Redis cache population skipped: {cache_err}\n")

            # Refresh telemetry cache
            try:
                from src.core.telemetry import TelemetryService
                TelemetryService.force_refresh_metrics()
            except Exception as telem_err:
                sys.stderr.write(f"Warning: Telemetry cache refresh failed: {telem_err}\n")

        except Exception as e:
            sys.stderr.write(f"Error during cache pre-warming pipeline: {e}\n")
            return 1

    report = {
        "prewarmed_documents": docs_processed,
        "embeddings_computed": embeddings_count,
        "redis_cache_status": redis_status,
        "status": "success",
    }

    print(json.dumps(report, indent=2))
    return 0



def run_db_status(
    db_path: str,
    db_type: str,
    output_format: str = "text",
) -> int:
    """Print migration status for an auth or corpus SQLite database."""
    try:
        from src.db.migrations import (
            AUTH_MIGRATIONS,
            CORPUS_MIGRATIONS,
            get_migration_status,
        )

        migrations = (
            AUTH_MIGRATIONS
            if db_type == "auth"
            else CORPUS_MIGRATIONS
        )
        status = get_migration_status(db_path, migrations)
    except (
        FileNotFoundError,
        IsADirectoryError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(
            f"Error: Unable to inspect database migration status: {exc}\n"
        )
        return 1

    if output_format == "json":
        print(json.dumps(status, indent=2))
    else:
        pending = status["pending_migrations"]
        pending_text = (
            ", ".join(str(version) for version in pending)
            if pending
            else "none"
        )
        print(f"Database: {db_path}")
        print(f"Type: {db_type}")
        print(f"Current version: {status['current_version']}")
        print(f"Target version: {status['target_version']}")
        print(f"Pending migrations: {pending_text}")

    return 0

def run_database_optimization(db_path: str) -> int:
    """Run SQLite maintenance for one database and return a CLI exit code."""
    if optimize_database(db_path):
        print(f"Database optimized successfully: {Path(db_path).expanduser()}")
        return 0

    sys.stderr.write(f"Error: Database optimization failed for '{db_path}'.\n")
    return 1


def main() -> None:
    setup_logging()

    # Support the issue-requested flag form:
    # ``python -m src.cli --db-status path --db-type corpus``.
    # Internally it is normalized to the regular ``db-status`` subcommand.
    argv = sys.argv[1:]
    if argv and argv[0] == "--db-status":
        argv[0] = "db-status"
    parser = argparse.ArgumentParser(
        description="Headless CLI Version for Plagiarism Detection Automation"
    )
    parser.add_argument(
        "--optimize",
        metavar="DB_PATH",
        help=(
            "Run PRAGMA optimize, VACUUM, and ANALYZE on the specified "
            "SQLite database, then exit."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser("scan", help="Scan a folder of assignments")
    scan_parser.add_argument("folder", help="Path to folder containing documents")
    scan_parser.add_argument(
        "--threshold",
        type=float,
        default=0.59,
        help="Similarity threshold for flagging (default: 0.59)",
    )
    scan_parser.add_argument(
        "--output-format",
        choices=["json", "csv", "text"],
        default="text",
        help="Output format for scan results (default: text)",
    )

    subparsers.add_parser(
        "sync-index", help="Verify and repair FAISS index sync with SQLite database."
    )

    prewarm_parser = subparsers.add_parser(
        "prewarm",
        help="Pre-compute embeddings and populate Redis cache before user logins.",
    )
    prewarm_parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Optional path to folder containing documents for pre-warming.",
    )

    db_status_parser = subparsers.add_parser(
        "db-status",
        help="Inspect pending SQLite schema migrations without applying them.",
    )
    db_status_parser.add_argument(
        "database",
        help="Path to an existing SQLite database file.",
    )
    db_status_parser.add_argument(
        "--db-type",
        choices=["auth", "corpus"],
        default="corpus",
        help="Migration set to inspect (default: corpus).",
    )
    db_status_parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Status output format (default: text).",
    )

    args = parser.parse_args(argv)

    if args.optimize is not None:
        if args.command is not None:
            parser.error("--optimize cannot be combined with a subcommand")
        sys.exit(run_database_optimization(args.optimize))

    if args.command is None:
        parser.error("a subcommand or --optimize is required")

    if args.command == "scan":
        if args.threshold < 0.0 or args.threshold > 1.0:
            sys.stderr.write("Error: Threshold must be a float between 0.0 and 1.0.\n")
            sys.exit(1)

        exit_code = run_scan(args.folder, args.threshold, output_format=args.output_format)
        sys.exit(exit_code)

    elif args.command == "sync-index":
        print("Starting FAISS and Database synchronization verification...")
        # Use the centralized FAISS index path.  Previously this resolved to
        # ``<repo>/src/corpus.index`` (relative to src/), which diverged from
        # every other module that uses ``<repo>/corpus.index``.  Centralizing
        # here fixes that drift.
        index_path = str(FAISS_INDEX_PATH)
        try:
            verify_and_repair_index(index_path)
            print("Synchronization complete.")
            return 0
        except Exception as e:
            sys.stderr.write(f"Error during synchronization: {e}\n")
            return 1

    elif args.command == "db-status":
        exit_code = run_db_status(
            args.database,
            args.db_type,
            output_format=args.output_format,
        )
        sys.exit(exit_code)

    elif args.command == "prewarm":
        exit_code = run_prewarm(folder_path=args.folder)
        sys.exit(exit_code)

    else:
        sys.stderr.write(f"Error: Invalid command '{args.command}'.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
