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

from src.core.cross_lingual import prepare_text_for_embedding
from src.core.document_parser import (DEFAULT_OCR_DPI, DEFAULT_OCR_LANGUAGE,
                                      extract_text)
from src.core.embedding_model import embed_documents
from src.core.similarity import document_similarity_matrix, flag_plagiarism
from src.core.synchronization import verify_and_repair_index
from src.core.text_chunking import chunk_documents


def run_scan(folder_path: str, threshold: float) -> int:
    """
    Scans a folder, processes the documents, runs plagiarism detection,
    and prints a JSON report to stdout.
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

    print(json.dumps(report, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless CLI Version for Plagiarism Detection Automation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan a folder of assignments")
    scan_parser.add_argument("folder", help="Path to folder containing documents")
    scan_parser.add_argument(
        "--threshold",
        type=float,
        default=0.59,
        help="Similarity threshold for flagging (default: 0.59)",
    )

    subparsers.add_parser(
        "sync-index", help="Verify and repair FAISS index sync with SQLite database."
    )

    args = parser.parse_args()

    if args.command == "scan":
        if args.threshold < 0.0 or args.threshold > 1.0:
            sys.stderr.write("Error: Threshold must be a float between 0.0 and 1.0.\n")
            sys.exit(1)

        exit_code = run_scan(args.folder, args.threshold)
        sys.exit(exit_code)

    elif args.command == "sync-index":
        print("Starting FAISS and Database synchronization verification...")
        index_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "corpus.index")
        )
        try:
            verify_and_repair_index(index_path)
            print("Synchronization complete.")
            return 0
        except Exception as e:
            sys.stderr.write(f"Error during synchronization: {e}\n")
            return 1

    else:
        sys.stderr.write(f"Error: Invalid command '{args.command}'.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
