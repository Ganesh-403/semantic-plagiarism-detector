#!/usr/bin/env python3
"""
Quickstart example for semantic plagiarism detection.

This self-contained script demonstrates the core pipeline programmatically:
1. extract_text()         -- pull raw text out of a document (TXT, PDF, DOCX, ...)
2. chunk_text()           -- split each document into semantic chunks
3. embed_documents()      -- embed the chunks with a SentenceTransformer model
4. compute_similarity_matrix() -- compute a document-level similarity matrix
5. flag_plagiarism()      -- flag document pairs above the plagiarism threshold

Execution instructions
----------------------
1. Install the project dependencies (from the repository root):

       pip install -r requirements.txt

   The first run downloads the ``paraphrase-multilingual-MiniLM-L12-v2``
   embedding model (~420 MB) and caches it locally.

2. Prepare two text files you want to compare (TXT, PDF, DOCX, and other
   supported formats all work):

       echo "This is the first document." > doc1.txt
       echo "This is the first document." > doc2.txt

3. Run the script from the repository root:

       python examples/basic_plagiarism_check.py doc1.txt doc2.txt

   Expected output includes the similarity matrix and any flagged pairs.

The script must be run from the repository root so that the ``src/`` package
is importable (the script adds the repository root to ``sys.path``, but the
``examples/`` directory itself must be reachable as well).
"""

import sys
from pathlib import Path

# Add the repository root to sys.path so the example can be run from any
# working directory (mirrors scripts/run_benchmark.py and scripts/coverage_report.py).
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import PLAGIARISM_THRESHOLD
from src.core.document_parser import extract_text
from src.core.embedding_model import embed_documents
from src.core.similarity import compute_similarity_matrix, flag_plagiarism
from src.core.text_chunking import chunk_text


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python examples/basic_plagiarism_check.py <file1> <file2>")
        sys.exit(1)

    file1 = Path(sys.argv[1])
    file2 = Path(sys.argv[2])

    for path in (file1, file2):
        if not path.is_file():
            print(f"Error: file not found: {path}")
            sys.exit(1)

    text1 = extract_text(file1.read_bytes(), file1.name)
    text2 = extract_text(file2.read_bytes(), file2.name)

    chunked_documents = {
        file1.name: chunk_text(text1),
        file2.name: chunk_text(text2),
    }

    embeddings = embed_documents(chunked_documents)

    similarity_matrix = compute_similarity_matrix(embeddings)

    print("\nSemantic Similarity Matrix")
    print(similarity_matrix)

    flagged = flag_plagiarism(
        similarity_matrix,
        threshold=PLAGIARISM_THRESHOLD,
        chunked_docs=chunked_documents,
        embeddings=embeddings,
    )

    print(f"\nPlagiarism threshold: {PLAGIARISM_THRESHOLD}")
    if flagged:
        print(f"Flagged pairs: {len(flagged)}")
        for record in flagged:
            print(
                f"  {record['doc_a']} <-> {record['doc_b']}: "
                f"{record['similarity']:.4f} ({record['severity']})"
            )
    else:
        print("No pairs exceeded the plagiarism threshold.")


if __name__ == "__main__":
    main()
