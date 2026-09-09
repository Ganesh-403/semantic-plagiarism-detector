"""
Document Execution Pipeline Module.

Encapsulates document processing, chunking orchestration, vector embedding,
FAISS index construction, similarity matrix calculation, AI detection,
and cross-lingual plagiarism detection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
import pandas as pd
import streamlit as st

from src.core.ai_detector import detect_documents_ai_probability
from src.core.document_parser import (
    extract_text,
    prepare_text_for_embedding,
)
from src.core.embedding_model import embed_chunks, embed_documents
from src.core.faiss_index import (
    build_index,
    build_index_from_matrix,
)
from src.core.similarity import (
    cosine_similarity,
    document_similarity_matrix,
)
from src.core.text_chunking import chunk_documents

logger = logging.getLogger(__name__)


class ChunkRecord:
    """Record container representing an extracted text chunk."""

    def __init__(self, doc_name, chunk_index, chunk_text, chunk_id=None, metadata=None):
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.chunk_text = chunk_text
        self.chunk_id = chunk_id
        self.metadata = metadata or {}


PipelineChunkRecord = ChunkRecord


# ============================================================================
# CROSS-LINGUAL INTEGRATION
# ============================================================================


def _get_cross_lingual_mode() -> bool:
    """Get cross-lingual mode from session state."""
    try:
        return st.session_state.get("cross_lingual_mode_toggle", False)
    except:  # noqa: E722
        return False


def _process_chunks_cross_lingual(
    chunked_docs: Dict[str, List[str]], cross_lingual_mode: bool = False
) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]]]:
    """
    Process chunks with cross-lingual translation if enabled.

    Returns:
        Tuple of (processed_chunks, metadata)
    """
    if not cross_lingual_mode:
        # Return original chunks with empty metadata
        metadata = {doc_name: [] for doc_name in chunked_docs}
        return chunked_docs, metadata

    try:
        from src.core.cross_lingual import prepare_documents_for_embedding

        return prepare_documents_for_embedding(chunked_docs)
    except ImportError:
        logger.warning(
            "Cross-lingual module not available. Falling back to standard processing."
        )
        metadata = {doc_name: [] for doc_name in chunked_docs}
        return chunked_docs, metadata


# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================


def run_pipeline(file_bytes_dict: dict, ocr_language: str, ocr_dpi: int,
                 chunk_size: int, chunk_overlap: int, cross_lingual_mode: bool = False) -> tuple:
    """Parse documents once and return the shared eight-field UI result."""
    import io
    raw_texts = {
        name: extract_text(io.BytesIO(data), name, ocr_language=ocr_language, ocr_dpi=ocr_dpi)
        for name, data in file_bytes_dict.items()
    }
    result = run_extraction_pipeline(tuple(raw_texts.items()), chunk_size, chunk_overlap, cross_lingual_mode)
    return (raw_texts, *result)


def run_extraction_pipeline(
    raw_texts_items: tuple,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    cross_lingual_mode: bool = False,
):
    """
    Cached extraction pipeline for text dictionary processing.

    Parameters
    ----------
    raw_texts_items : tuple
        Tuple of (filename, text) pairs.
    chunk_size : int
        Target character chunk size.
    chunk_overlap : int
        Chunk character overlap.
    cross_lingual_mode : bool
        Enable cross-lingual detection.

    Returns
    -------
    tuple
        (chunked_docs, embeddings, sim_df, chunk_sim_df, faiss_index, registry, ai_probabilities)
    """
    raw_texts_dict = dict(raw_texts_items)
    chunked_docs = chunk_documents(
        raw_texts_dict, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # ===== CROSS-LINGUAL PROCESSING =====
    translation_metadata = {}

    if cross_lingual_mode:
        try:
            from src.core.cross_lingual import prepare_documents_for_embedding

            logger.info("Processing documents with cross-lingual translation...")
            translated_chunked_docs, metadata = prepare_documents_for_embedding(
                chunked_docs
            )

            # Store metadata for UI
            translation_metadata = metadata

            # Use translated chunks for embedding
            processed_chunked_docs = translated_chunked_docs
            logger.info(
                f"Cross-lingual processing complete for {len(processed_chunked_docs)} documents"
            )

        except ImportError as e:
            logger.warning(f"Cross-lingual module unavailable: {e}")
            # Fallback to standard processing
            processed_chunked_docs = {}
            for doc_name, chunks in chunked_docs.items():
                processed_chunked_docs[doc_name] = []
                for chunk in chunks:
                    prepared = prepare_text_for_embedding(chunk.text if hasattr(chunk, "text") else chunk)
                    processed_chunked_docs[doc_name].append(prepared["embedding_text"])
    else:
        # Standard processing without translation
        processed_chunked_docs = {}
        for doc_name, chunks in chunked_docs.items():
            processed_chunked_docs[doc_name] = []
            for chunk in chunks:
                prepared = prepare_text_for_embedding(chunk.text if hasattr(chunk, "text") else chunk)
                processed_chunked_docs[doc_name].append(prepared["embedding_text"])
    # ===================================

    embeddings = embed_documents(processed_chunked_docs)
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
    faiss_index, registry = build_index(embeddings, processed_chunked_docs)
    ai_probabilities = detect_documents_ai_probability(processed_chunked_docs)

    # Store translation metadata in session state for UI
    if cross_lingual_mode and translation_metadata:
        try:
            st.session_state["translation_metadata"] = translation_metadata
        except:  # noqa: E722
            pass

    return (
        processed_chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
    )


def run_pipeline_with_tracking(
    file_bytes_dict: dict,
    ocr_language: str,
    ocr_dpi: int,
    chunk_size: int,
    chunk_overlap: int,
    cross_lingual_mode: bool = False,
) -> tuple:
    """
    Run pipeline with performance tracking and cross-lingual support.
    Wrapper around run_pipeline for backward compatibility.
    """
    return run_pipeline(
        file_bytes_dict,
        ocr_language,
        ocr_dpi,
        chunk_size,
        chunk_overlap,
        cross_lingual_mode=cross_lingual_mode,
    )
