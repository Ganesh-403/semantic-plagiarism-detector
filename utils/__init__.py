# utils/__init__.py
from .document_parser import (
    extract_text_from_pdf,
    extract_texts_from_pdfs,
    extract_text,
    extract_texts,
)
from .text_chunking import chunk_document, chunk_documents
from .embedding_model import embed_chunks, embed_documents, get_document_embedding
from .similarity import (
    document_similarity_matrix,
    chunk_similarity_matrix,
    flag_plagiarism,
    find_most_similar_chunks,
    PLAGIARISM_THRESHOLD,
)
from .heatmap import plot_similarity_heatmap, plot_chunk_similarity_comparison
from .faiss_index import (
    build_index,
    search_similar_chunks,
    find_plagiarised_chunks,
    save_index,
    load_index,
    ChunkRecord,
)

__all__ = [
    "extract_text_from_pdf", "extract_texts_from_pdfs",
    "extract_text", "extract_texts",
    "chunk_document", "chunk_documents",
    "embed_chunks", "embed_documents", "get_document_embedding",
    "document_similarity_matrix", "chunk_similarity_matrix",
    "flag_plagiarism", "find_most_similar_chunks", "PLAGIARISM_THRESHOLD",
    "plot_similarity_heatmap", "plot_chunk_similarity_comparison",
    "build_index", "search_similar_chunks", "find_plagiarised_chunks",
    "save_index", "load_index", "ChunkRecord",
]
