"""Public compatibility exports, loaded on demand without startup side effects."""

from importlib import import_module

_EXPORTS = {
    "BrandingConfig": (".config", "BrandingConfig"),
    "get_branding_config": (".config", "get_branding_config"),
    "load_branding_config": (".config", "load_branding_config"),
    "reload_branding_config": (".config", "reload_branding_config"),
    "check_ocr_dependencies": (".document_parser", "check_ocr_dependencies"),
    "extract_text": (".document_parser", "extract_text"),
    "extract_text_from_pdf": (".document_parser", "extract_text_from_pdf"),
    "extract_texts": (".document_parser", "extract_texts"),
    "extract_texts_from_pdfs": (".document_parser", "extract_texts_from_pdfs"),
    "sanitize_zero_width_characters": (
        ".document_parser",
        "sanitize_zero_width_characters",
    ),
    "embed_chunks": (".embedding_model", "embed_chunks"),
    "embed_documents": (".embedding_model", "embed_documents"),
    "get_document_embedding": (".embedding_model", "get_document_embedding"),
    "FAISSIndex": (".faiss_index", "FAISSIndex"),
    "FaissIndexManager": (".faiss_index", "FaissIndexManager"),
    "ChunkRecord": (".faiss_index", "ChunkRecord"),
    "FaissChunkRecord": (".faiss_index", "FaissChunkRecord"),
    "build_index": (".faiss_index", "build_index"),
    "build_index_from_matrix": (".faiss_index", "build_index_from_matrix"),
    "find_plagiarised_chunks": (".faiss_index", "find_plagiarised_chunks"),
    "format_faiss_memory_badge": (".faiss_index", "format_faiss_memory_badge"),
    "get_faiss_index_memory_bytes": (".faiss_index", "get_faiss_index_memory_bytes"),
    "load_index": (".faiss_index", "load_index"),
    "rebuild_index_from_database": (".faiss_index", "rebuild_index_from_database"),
    "rebuild_index_from_db": (".faiss_index", "rebuild_index_from_db"),
    "save_index": (".faiss_index", "save_index"),
    "search_similar_chunks": (".faiss_index", "search_similar_chunks"),
    "PipelineChunkRecord": (".pipeline", "PipelineChunkRecord"),
    "run_extraction_pipeline": (".pipeline", "run_extraction_pipeline"),
    "run_pipeline": (".pipeline", "run_pipeline"),
    "PLAGIARISM_THRESHOLD": (".similarity", "PLAGIARISM_THRESHOLD"),
    "calculate_paragraph_similarity_breakdown": (
        ".similarity",
        "calculate_paragraph_similarity_breakdown",
    ),
    "chunk_similarity_matrix": (".similarity", "chunk_similarity_matrix"),
    "document_similarity_matrix": (".similarity", "document_similarity_matrix"),
    "find_most_similar_chunks": (".similarity", "find_most_similar_chunks"),
    "flag_plagiarism": (".similarity", "flag_plagiarism"),
    "manhattan_similarity": (".similarity", "manhattan_similarity"),
    "BaseSimilarityEngine": (".similarity_base", "BaseSimilarityEngine"),
    "HybridSimilarityEngine": (".similarity_engines", "HybridSimilarityEngine"),
    "LexicalSimilarityEngine": (".similarity_engines", "LexicalSimilarityEngine"),
    "SemanticSimilarityEngine": (".similarity_engines", "SemanticSimilarityEngine"),
    "SimilarityEngineFactory": (".similarity_engines", "SimilarityEngineFactory"),
    "TagManager": (".tag_manager", "TagManager"),
    "sanitize_tag_name": (".tag_manager", "sanitize_tag_name"),
    "Chunk": (".text_chunking", "Chunk"),
    "ChunkString": (".text_chunking", "ChunkString"),
    "chunk_by_sentences": (".text_chunking", "chunk_by_sentences"),
    "chunk_document": (".text_chunking", "chunk_document"),
    "chunk_document_by_tokens": (".text_chunking", "chunk_document_by_tokens"),
    "chunk_documents": (".text_chunking", "chunk_documents"),
    "translate_text": (".translator", "translate_text"),
    "EventDispatcher": (".webhook", "EventDispatcher"),
    "dispatch_plagiarism_alert": (".webhook", "dispatch_plagiarism_alert"),
    "send_plagiarism_alert": (".webhook", "send_plagiarism_alert"),
    "with_sqlite_retry": (".concurrency", "with_sqlite_retry"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
