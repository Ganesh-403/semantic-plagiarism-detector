# NLP Architecture & Similarity Algorithm Guide

## Overview
(explain the app scores every document pair using two independent signals —
 lexical and semantic — then blends them into one hybrid score)

## Lexical Similarity (src/core/lexical_similarity.py)
- Jaccard similarity formula: |A ∩ B| / |A ∪ B| over stop-word-filtered
  token sets (see jaccard_similarity())
- TF-IDF + Cosine similarity: the production path used in
  lexical_similarity_matrix() — a single TfidfVectorizer is fit across all
  documents, then sklearn's cosine_similarity() is applied to the TF-IDF
  matrix
- Note that stop-words (the, and, is, …) are removed first so common words
  can't inflate the score (issue #222)

## Semantic Similarity (src/core/embedding_model.py + similarity.py)
- Model: paraphrase-multilingual-MiniLM-L12-v2, 384-dimensional embeddings
- Embeddings are L2-normalized at encode time, so cosine similarity reduces
  to a plain dot product (explain why that matters for speed)
- Document-level score: mean-pool chunk embeddings, then cosine similarity
  (document_similarity_matrix())
- Chunk-level score: max pairwise cosine similarity across all chunk pairs
  (chunk_max_similarity()) — used to catch localized/partial plagiarism

## Hybrid Score (hybrid_similarity_matrix())
- Formula: hybrid = w × semantic + (1 - w) × lexical, default w = 0.7
- Explain why semantic is weighted higher (catches paraphrasing, not just
  copy-paste) and give a worked numeric example

## FAISS ANN Index (src/core/faiss_index.py)
- Vector dimension: 384 (must match the embedding model's output)
- IndexFlatIP: exact brute-force inner-product search, used when
  vector count < 5,000 (the _IVF_THRESHOLD constant)
- IndexIVFFlat: approximate search with Voronoi cells (nlist, nprobe
  parameters), auto-selected once vector count ≥ 5,000
- Explain why inner product == cosine similarity here (same L2-normalization
  reason as above)

## Severity Tiers (src/core/config.py)
- Table of the three boundaries: plagiarism (0.59), medium (0.75), high (0.90)
- Explain severity_from_score() logic: below plagiarism → Low (not flagged),
  plagiarism–medium → Low (flagged), medium–high → Medium, ≥ high → High

## Extending the Algorithm
(short code example: how to plug in a new similarity signal or change the
 hybrid weight w)