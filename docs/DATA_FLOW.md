# Data Flow

This document describes how document data moves through the Semantic Plagiarism Detection System, from the initial file upload to text extraction, text chunking, semantic embedding, and FAISS indexing. It also documents the data format used at each processing stage.

## Processing Pipeline

### 1. File Upload

The processing pipeline begins when one or more documents are uploaded through the Streamlit application or API. Uploaded files are received as raw byte streams and stored in a dictionary mapping filenames to their corresponding byte content.

**Input Format**

```text
Dict[str, bytes]
```

Example:

```python
{"report.pdf": b"...", "assignment.docx": b"..."}
```

**Output Format**

```text
bytes
```

---

### 2. Text Extraction

Each uploaded file is passed to the document parser, which extracts readable text from supported document formats. If OCR is enabled, scanned documents are processed before text extraction.

The extracted text is stored using the document filename as the key.

**Transformation**

```text
bytes
        ↓
extract_text()
        ↓
str
```

**Output Format**

```python
{"report.pdf": "Extracted document text..."}
```

---

### 3. Text Chunking

The extracted document text is divided into smaller chunks before semantic embedding.

The current processing pipeline uses `chunk_text()` through `chunk_documents()`.

Characteristics:

- Default chunk size: **500 characters**
- Default overlap: **50 characters**
- Preserves word boundaries
- Filters very small chunks
- Uses character-based fallback chunking for languages without whitespace (for example CJK languages)

**Transformation**

```text
str
      ↓
chunk_documents()
      ↓
list[str]
```

Example:

```python
["Introduction...", "Methodology...", "Experimental Results..."]
```

---

### 4. Embedding Generation

Each text chunk is converted into a semantic vector using the SentenceTransformer model.

**Embedding Model**

```
paraphrase-multilingual-MiniLM-L12-v2
```

Features:

- Multilingual support
- 384-dimensional embeddings
- L2-normalized vectors
- Batch embedding generation

The embedding stage produces one vector for every text chunk.

**Transformation**

```text
list[str]
        ↓
embed_documents()
        ↓
np.ndarray
```

Output shape:

```text
(number_of_chunks, 384)
```

---

### 5. FAISS Indexing

After embeddings are generated, all chunk vectors are inserted into a FAISS index for efficient similarity search.

Each vector is associated with metadata using a `ChunkRecord`, which stores:

- Document name
- Chunk index
- Original chunk text
- Optional metadata

The system automatically selects the index type based on the number of vectors.

| Number of Vectors | FAISS Index |
|------------------:|------------|
| Less than 5000 | IndexFlatIP |
| 5000 or more | IndexIVFFlat |

Since embeddings are L2-normalized, the project uses inner-product search, which is equivalent to cosine similarity.

**Transformation**

```text
np.ndarray
        ↓
build_index()
        ↓
FAISS Index
```

---

## Data Flow Diagram

```mermaid
flowchart TD

A[Uploaded Document]
B[Raw File Bytes<br/>bytes]
C[extract_text()]
D[Extracted Text<br/>str]
E[chunk_documents()]
F[Text Chunks<br/>list[str]]
G[embed_documents()]
H[SentenceTransformer<br/>paraphrase-multilingual-MiniLM-L12-v2]
I[Embeddings<br/>np.ndarray (N × 384)]
J[build_index()]
K[FAISS Index]
L[IndexFlatIP / IndexIVFFlat]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
G --> H
H --> I
I --> J
J --> K
K --> L
```

---

## Data Format Summary

| Processing Stage | Input Format | Output Format |
|------------------|--------------|---------------|
| File Upload | File | `bytes` |
| Text Extraction | `bytes` | `str` |
| Text Chunking | `str` | `list[str]` |
| Embedding Generation | `list[str]` | `np.ndarray (N × 384)` |
| FAISS Indexing | `np.ndarray` | `FAISS Index` |

---

## Pipeline Summary

The complete processing pipeline transforms uploaded document data through several stages before semantic similarity analysis.

```text
Uploaded Document
        │
        ▼
Raw File Bytes (bytes)
        │
        ▼
Extract Text
        │
        ▼
Extracted Text (str)
        │
        ▼
Chunk Documents
        │
        ▼
Text Chunks (list[str])
        │
        ▼
Generate Embeddings
        │
        ▼
NumPy Array (N × 384)
        │
        ▼
Build FAISS Index
        │
        ▼
Semantic Search
```
