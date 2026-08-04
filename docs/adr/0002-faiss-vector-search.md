# ADR 0002: Use FAISS for Vector Similarity Search

## Status

Accepted

## Context

The Semantic Plagiarism Detector compares documents using semantic embeddings rather than exact text matching. As the number of document chunks increases, performing a brute-force similarity search over all embeddings becomes inefficient.

The project requires a solution that:

- Supports high-dimensional embedding vectors.
- Performs fast similarity searches.
- Integrates well with Python.
- Scales as the document corpus grows.
- Works alongside the existing SQLite-based metadata storage.

## Decision

The project uses **FAISS (Facebook AI Similarity Search)** as the vector search engine.

FAISS stores semantic embedding vectors and performs efficient nearest-neighbor searches to identify potentially similar document chunks.

Document metadata, filenames, chunk information, and other persistent data continue to be stored in SQLite, while FAISS is responsible only for vector indexing and retrieval.

## Consequences

### Positive

- Fast similarity search over large embedding collections.
- Optimized for high-dimensional vector operations.
- Widely adopted and actively maintained.
- Clean separation between vector indexing (FAISS) and metadata storage (SQLite).
- Supports future scaling with more advanced FAISS index types if needed.

### Negative

- Adds an external dependency to the project.
- Requires synchronization between the FAISS index and SQLite metadata.
- Contributors must understand both FAISS and SQLite when modifying indexing logic.
