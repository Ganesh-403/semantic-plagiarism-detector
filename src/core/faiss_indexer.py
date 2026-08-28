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
src/core/faiss_indexer.py
-------------------------
Builds and searches FAISS vector index for fast similarity lookup.
"""

from typing import Dict, List, Tuple, Union

import faiss
import numpy as np


def build_index(
    doc_embeddings: dict[str, np.ndarray] | np.ndarray | list[np.ndarray],
    chunked_docs: dict[str, list[str]] = None,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Build a FAISS IndexFlatIP (Inner Product / Cosine Similarity) index.

    Returns:
        Tuple of (FAISS index object, list of metadata records per index entry)
    """
    registry = []

    # Case 1: Matrix / List input
    if isinstance(doc_embeddings, (np.ndarray, list)):
        embeddings_arr = np.array(doc_embeddings).astype("float32")
        if embeddings_arr.ndim == 1 or embeddings_arr.size == 0:
            return None, []

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_arr)
        dim = embeddings_arr.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_arr)

        for i in range(len(embeddings_arr)):
            registry.append({"doc_id": i, "chunk_index": i})
        return index, registry

    # Case 2: Dict input mapping doc_name -> embedding_array
    if not doc_embeddings:
        return None, []

    all_vectors = []
    for doc_name, emb in doc_embeddings.items():
        if isinstance(emb, np.ndarray) and emb.size > 0:
            if emb.ndim == 1:
                emb = np.expand_dims(emb, axis=0)
            for chunk_idx in range(emb.shape[0]):
                all_vectors.append(emb[chunk_idx])
                registry.append({"doc_name": doc_name, "chunk_index": chunk_idx})

    if not all_vectors:
        return None, []

    vectors_arr = np.vstack(all_vectors).astype("float32")
    faiss.normalize_L2(vectors_arr)

    dim = vectors_arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_arr)

    return index, registry
