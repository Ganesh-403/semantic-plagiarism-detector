from .document_parser import (extract_text, extract_text_from_pdf,
                              extract_texts, extract_texts_from_pdfs,
                              sanitize_zero_width_characters)

from .embedding_model import (embed_chunks, embed_documents,
                              get_document_embedding)
from .faiss_index import (ChunkRecord, build_index, build_index_from_matrix,
                          find_plagiarised_chunks, load_index, save_index,
                          search_similar_chunks)
from .similarity import (PLAGIARISM_THRESHOLD, calculate_paragraph_similarity_breakdown,
                         chunk_similarity_matrix, document_similarity_matrix,
                         find_most_similar_chunks, flag_plagiarism)
from .text_chunking import chunk_document, chunk_documents
from .translator import translate_text
from .webhook import send_plagiarism_alert


__all__ = [
    "extract_text_from_pdf",
    "extract_texts_from_pdfs",
    "extract_text",
    "extract_texts",
    "chunk_document",
    "chunk_documents",
    "embed_chunks",
    "embed_documents",
    "get_document_embedding",
    "document_similarity_matrix",
    "chunk_similarity_matrix",
    "flag_plagiarism",
    "find_most_similar_chunks",
    "calculate_paragraph_similarity_breakdown",
    "PLAGIARISM_THRESHOLD",
    "build_index",
    "search_similar_chunks",
    "find_plagiarised_chunks",
    "save_index",
    "load_index",
    "ChunkRecord",
    "build_index_from_matrix",
    "translate_text",
    "send_plagiarism_alert",
    "dispatch_plagiarism_alert",
    "sanitize_zero_width_characters",
]
