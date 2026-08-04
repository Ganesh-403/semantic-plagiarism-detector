# FAISS-to-SQLite ID Mapping

## Overview

The plagiarism detector uses **FAISS** (Facebook AI Similarity Search) for fast vector similarity search and **SQLite** for persistent storage of document chunks, embeddings, and metadata. A shared integer ID scheme ties the two systems together.

## The Core Idea

**The FAISS vector index row ID is the same integer as the SQLite `vector_id` primary key, which is also the list index into the in-memory registry (`List[ChunkRecord]`).**

All three representations are built in the same order — documents in insertion order, chunks within each document in ascending chunk index — so the positional index is always identical across all three.

## Key Data Structures

### `ChunkRecord` (`src/core/faiss_index.py`)

```python
@dataclass
class ChunkRecord:
    doc_name: str       # document filename, e.g. "essay1.pdf"
    chunk_index: int    # 0-based position of this chunk within the document
    chunk_text: str     # the raw text of this chunk
```

A `List[ChunkRecord]` (the **registry**) serves as the lookup table: `registry[faiss_id]` returns the `ChunkRecord` for that vector.

### SQLite `chunks` Table (`src/db/corpus_db.py`)

```sql
CREATE TABLE chunks (
    vector_id   INTEGER PRIMARY KEY,   -- FAISS vector index (0, 1, 2, ...)
    filename    TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT    NOT NULL,
    embedding   BLOB    NOT NULL       -- float32 embedding via .tobytes()
);
```

## How the Mapping Works

### 1. Indexing Flow

```
Documents (PDF/DOCX/TXT)
        │
        ▼
  chunk_documents()        ──►  Dict[doc_name, List[chunk_text]]
        │
        ▼
  embed_chunks()           ──►  Dict[doc_name, np.ndarray (N_chunks x 384)]
        │
        ▼
  build_index(embeddings, chunked_docs)
        │
        ├── Iterates documents in insertion order
        ├── For each doc, iterates chunks in order
        ├── Appends each vector → all_vectors list
        ├── Appends ChunkRecord(doc_name, i, text) → registry list
        │
        ▼
  matrix = np.vstack(all_vectors)   shape: (total_chunks, 384)
  index.add(matrix)                  FAISS assigns IDs 0, 1, 2, ... N-1

  Result: FAISS ID `i` == registry[i] == matrix[i]
```

### 2. Storage Flow

```python
# Each chunk stored with its vector_id
add_chunks([
    (vector_id, filename, chunk_index, chunk_text, embedding_np_array),
    ...
])
```

### 3. Query Flow

```python
scores, indices = index.search(query_vector, top_k)
# indices[0] = [FAISS_ID_1, FAISS_ID_2, ...]

for score, faiss_id in zip(scores[0], indices[0]):
    record = registry[faiss_id]              # direct list-index lookup
    print(record.doc_name, record.chunk_text)
```

### 4. Recovery Flow

```python
def load_or_rebuild_index(filepath):
    matrix = get_all_embeddings()      # SELECT ... ORDER BY vector_id ASC
    registry = get_chunk_registry()    # SELECT ... ORDER BY vector_id ASC
    assert matrix.shape[0] == len(registry)   # safety check
    # Use or rebuild the on-disk FAISS index
```

Both `get_all_embeddings()` and `get_chunk_registry()` use `ORDER BY vector_id ASC`, ensuring the row order matches the FAISS index order.

### 5. Document Deletion

When a document is deleted, its chunks are removed via `ON DELETE CASCADE`, leaving gaps in `vector_id`.

`_compact_vector_ids()` (in `corpus_db.py`) reassigns sequential IDs:

```python
def _compact_vector_ids():
    # Read all remaining chunks in order
    # Delete all chunks
    # Re-insert with fresh sequential IDs: 0, 1, 2, ...
```

After compaction, the on-disk FAISS index is stale and must be rebuilt:

```python
embeddings_matrix = get_all_embeddings()
new_index = build_index_from_matrix(embeddings_matrix)
save_index(new_index, _INDEX_PATH)
```

## Consistency Guarantees

| Check | Location | What it verifies |
|---|---|---|
| Matrix rows == registry length | `load_or_rebuild_index()` | `matrix.shape[0] == len(registry)` |
| ORDER BY vector_id ASC | `get_all_embeddings()` and `get_chunk_registry()` | Both use the same ordering |
| Vector ID compaction | `_compact_vector_ids()` | Removes gaps after deletion |
| Index rebuild after delete | `streamlit_app.py` | `build_index_from_matrix()` + `save_index()` |

## File Reference

| File | Key Functions |
|---|---|
| `src/core/faiss_index.py` | `ChunkRecord`, `build_index()`, `search_similar_chunks()`, `load_or_rebuild_index()`, `build_index_from_matrix()` |
| `src/db/corpus_db.py` | `add_chunks()`, `get_chunk_registry()`, `get_all_embeddings()`, `_compact_vector_ids()`, `delete_document()` |
| `app/streamlit_app.py` | Document management UI, FAISS search UI |
