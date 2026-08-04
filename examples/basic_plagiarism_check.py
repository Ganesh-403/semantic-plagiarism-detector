#!/usr/bin/env python3
"""
Quickstart example for semantic plagiarism detection.

Usage:
    python examples/basic_plagiarism_check.py <file1> <file2>

Example:
    python examples/basic_plagiarism_check.py essay1.txt essay2.txt

This example demonstrates how to:
1. Extract text from two documents.
2. Split the text into semantic chunks.
3. Generate semantic embeddings.
4. Compute a document similarity matrix.
"""

from pathlib import Path
import sys

from src.core.document_parser import extract_text
from src.core.embedding_model import embed_documents
from src.core.similarity import compute_similarity_matrix
from src.core.text_chunking import chunk_text


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python examples/basic_plagiarism_check.py <file1> <file2>")
        sys.exit(1)

    file1 = Path(sys.argv[1])
    file2 = Path(sys.argv[2])

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


if __name__ == "__main__":
    main()
