"""
src/core/processing.py
----------------------
Standalone pipeline functions for document analysis, independent of the
Streamlit UI layer.  Used both by the synchronous upload path and by the
background RQ worker.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import PLAGIARISM_THRESHOLD
from src.core.document_parser import extract_text
from src.core.embedding_model import embed_documents
from src.core.faiss_index import build_index, ChunkRecord
from src.core.similarity import document_similarity_matrix, flag_plagiarism
from src.core.text_chunking import chunk_documents
from src.core.ai_detector import detect_documents_ai_probability

logger = logging.getLogger(__name__)


PipelineResult = Tuple[
    Dict[str, str],             # raw_texts
    Dict[str, List[str]],       # chunked_docs
    Dict[str, np.ndarray],      # embeddings
    pd.DataFrame,               # sim_df
    pd.DataFrame,               # chunk_sim_df
    Any,                        # faiss_index
    List[ChunkRecord],          # registry
    Dict[str, Dict[str, Any]],  # ai_probabilities
    List[Dict[str, Any]],       # flags
]


def run_full_pipeline(
    file_bytes_dict: Dict[str, bytes],
    *,
    ocr_language: str = "eng",
    ocr_dpi: int = 300,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    threshold: float = PLAGIARISM_THRESHOLD,
    ignore_phrases: Optional[str] = None,
    url_text: Optional[str] = None,
    url_filename: Optional[str] = None,
) -> PipelineResult:
    """Execute the full document upload pipeline outside of Streamlit.

    This is the same logic as ``streamlit_app.run_pipeline()`` but without
    the ``@st.cache_data`` decorator and ``st.warning`` calls, making it
    suitable for background workers and API-driven usage.

    Returns:
        A tuple containing all pipeline outputs including the final flags list.
    """
    import psutil

    raw_texts: Dict[str, str] = {}
    failed_files: List[str] = []
    failure_details: List[str] = []

    for name, data in file_bytes_dict.items():
        if not data:
            continue
        try:
            raw_texts[name] = extract_text(
                io.BytesIO(data), name, ocr_language=ocr_language, ocr_dpi=ocr_dpi
            )
        except Exception as exc:
            failed_files.append(name)
            failure_details.append(f"{name}: {exc}")

    if url_text and url_filename:
        raw_texts[url_filename] = url_text

    if failed_files:
        from src.errors import OCRFileBatchError
        raise OCRFileBatchError(failed_files, failure_details)

    if ignore_phrases and ignore_phrases.strip():
        from src.core.document_parser import remove_ignore_phrases
        raw_texts = {
            name: remove_ignore_phrases(text, ignore_phrases)
            for name, text in raw_texts.items()
        }

    chunked_docs = chunk_documents(
        raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    translated_chunked_docs: Dict[str, List[str]] = {}
    for doc_name, chunks in chunked_docs.items():
        prepared_list: List[str] = []
        for chunk in chunks:
            prepared_list.append(chunk)
        translated_chunked_docs[doc_name] = prepared_list

    embeddings = embed_documents(translated_chunked_docs)
    sim_df = document_similarity_matrix(embeddings)

    names = list(embeddings.keys())
    n = len(names)
    chunk_mat = np.zeros((n, n))

    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                chunk_mat[i, j] = 1.0
            elif j > i:
                ea, eb = embeddings[na], embeddings[nb]
                score = (
                    float(np.max(cosine_similarity(ea, eb)))
                    if ea.size and eb.size
                    else 0.0
                )
                chunk_mat[i, j] = score
                chunk_mat[j, i] = score

    chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)

    memory = psutil.virtual_memory()
    if memory.percent >= 85:
        logger.warning(
            "High memory usage detected (%d%%). Large FAISS indexes may cause "
            "instability or OOM crashes.",
            memory.percent,
        )

    faiss_index, registry = build_index(embeddings, chunked_docs)
    ai_probabilities = detect_documents_ai_probability(chunked_docs)

    flags = flag_plagiarism(
        sim_df,
        threshold=threshold,
        chunked_docs=chunked_docs,
        embeddings=embeddings,
    )

    return (
        raw_texts,
        chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
        flags,
    )
