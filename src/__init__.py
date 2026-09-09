"""Public compatibility exports, loaded on demand without startup side effects."""

from importlib import import_module

_EXPORTS = {
    "PLAGIARISM_THRESHOLD": (".core", "PLAGIARISM_THRESHOLD"),
    "BrandingConfig": (".core", "BrandingConfig"),
    "ChunkRecord": (".core", "ChunkRecord"),
    "FaissChunkRecord": (".core", "FaissChunkRecord"),
    "PipelineChunkRecord": (".core", "PipelineChunkRecord"),
    "TagManager": (".core", "TagManager"),
    "build_index": (".core", "build_index"),
    "build_index_from_matrix": (".core", "build_index_from_matrix"),
    "calculate_paragraph_similarity_breakdown": (
        ".core",
        "calculate_paragraph_similarity_breakdown",
    ),
    "check_ocr_dependencies": (".core", "check_ocr_dependencies"),
    "chunk_by_sentences": (".core", "chunk_by_sentences"),
    "chunk_document": (".core", "chunk_document"),
    "chunk_documents": (".core", "chunk_documents"),
    "chunk_similarity_matrix": (".core", "chunk_similarity_matrix"),
    "document_similarity_matrix": (".core", "document_similarity_matrix"),
    "embed_chunks": (".core", "embed_chunks"),
    "embed_documents": (".core", "embed_documents"),
    "extract_text": (".core", "extract_text"),
    "extract_text_from_pdf": (".core", "extract_text_from_pdf"),
    "extract_texts": (".core", "extract_texts"),
    "extract_texts_from_pdfs": (".core", "extract_texts_from_pdfs"),
    "find_most_similar_chunks": (".core", "find_most_similar_chunks"),
    "find_plagiarised_chunks": (".core", "find_plagiarised_chunks"),
    "flag_plagiarism": (".core", "flag_plagiarism"),
    "get_branding_config": (".core", "get_branding_config"),
    "get_document_embedding": (".core", "get_document_embedding"),
    "load_branding_config": (".core", "load_branding_config"),
    "load_index": (".core", "load_index"),
    "rebuild_index_from_database": (".core", "rebuild_index_from_database"),
    "rebuild_index_from_db": (".core", "rebuild_index_from_db"),
    "reload_branding_config": (".core", "reload_branding_config"),
    "sanitize_tag_name": (".core", "sanitize_tag_name"),
    "sanitize_zero_width_characters": (".core", "sanitize_zero_width_characters"),
    "save_index": (".core", "save_index"),
    "send_plagiarism_alert": (".core", "send_plagiarism_alert"),
    "translate_text": (".core", "translate_text"),
    "CorpusRepository": (".db", "CorpusRepository"),
    "add_chunks": (".db", "add_chunks"),
    "add_document": (".db", "add_document"),
    "add_user": (".db", "add_user"),
    "clear_all_data": (".db", "clear_all_data"),
    "delete_document": (".db", "delete_document"),
    "delete_user": (".db", "delete_user"),
    "disable_2fa": (".db", "disable_2fa"),
    "enable_2fa": (".db", "enable_2fa"),
    "get_2fa_status": (".db", "get_2fa_status"),
    "get_all_documents": (".db", "get_all_documents"),
    "get_all_embeddings": (".db", "get_all_embeddings"),
    "get_all_users": (".db", "get_all_users"),
    "get_chunk_registry": (".db", "get_chunk_registry"),
    "get_deleted_documents_count": (".db", "get_deleted_documents_count"),
    "get_document_by_hash": (".db", "get_document_by_hash"),
    "get_document_chunks_count": (".db", "get_document_chunks_count"),
    "get_documents_by_class": (".db", "get_documents_by_class"),
    "get_incidents_by_assignment": (".db", "get_incidents_by_assignment"),
    "get_unique_class_sections": (".db", "get_unique_class_sections"),
    "get_user_active_status": (".db", "get_user_active_status"),
    "get_user_role": (".db", "get_user_role"),
    "init_corpus_db": (".db", "init_corpus_db"),
    "init_db": (".db", "init_db"),
    "is_user_active": (".db", "is_user_active"),
    "restore_document": (".db", "restore_document"),
    "set_user_active_status": (".db", "set_user_active_status"),
    "soft_delete_document": (".db", "soft_delete_document"),
    "revoke_all_user_refresh_tokens": (".db", "revoke_all_user_refresh_tokens"),
    "update_password": (".db", "update_password"),
    "update_user_profile": (".db", "update_user_profile"),
    "verify_user": (".db", "verify_user"),
    "build_network_data": (".visualization", "build_network_data"),
    "plot_chunk_similarity_comparison": (
        ".visualization",
        "plot_chunk_similarity_comparison",
    ),
    "plot_document_similarity_heatmap": (
        ".visualization",
        "plot_document_similarity_heatmap",
    ),
    "plot_similarity_heatmap": (".visualization", "plot_similarity_heatmap"),
    "plot_similarity_heatmap_plotly": (
        ".visualization",
        "plot_similarity_heatmap_plotly",
    ),
    "plot_similarity_network": (".visualization", "plot_similarity_network"),
    "render_network_plotly": (".visualization", "render_network_plotly"),
    "filter_heatmap_by_class_tag": (
        ".visualization.heatmap",
        "filter_heatmap_by_class_tag",
    ),
    "export_graph_to_csv": (".visualization.network_graph", "export_graph_to_csv"),
    "export_network_to_csv_bytes": (
        ".visualization.network_graph",
        "export_network_to_csv_bytes",
    ),
    "dispatch_plagiarism_alert": (".core.webhook", "dispatch_plagiarism_alert"),
    "search_similar_chunks": (".core.faiss_index", "search_similar_chunks"),
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
