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
src/__init__.py
---------------
Package initialization for the Semantic Plagiarism Detector core utilities, database handlers, and visualization routines.
"""

from .core import (
    PLAGIARISM_THRESHOLD,
    BrandingConfig,
    ChunkRecord,
    FaissChunkRecord,
    PipelineChunkRecord,
    TagManager,
    build_index,
    build_index_from_matrix,
    calculate_paragraph_similarity_breakdown,
    check_ocr_dependencies,
    chunk_by_sentences,
    chunk_document,
    chunk_documents,
    chunk_similarity_matrix,
    document_similarity_matrix,
    embed_chunks,
    embed_documents,
    extract_text,
    extract_text_from_pdf,
    extract_texts,
    extract_texts_from_pdfs,
    find_most_similar_chunks,
    find_plagiarised_chunks,
    flag_plagiarism,
    get_branding_config,
    get_document_embedding,
    load_branding_config,
    load_index,
    reload_branding_config,
    sanitize_tag_name,
    sanitize_zero_width_characters,
    save_index,
    send_plagiarism_alert,
    translate_text,
)
from .db import (
    CorpusRepository,
    add_chunks,
    add_document,
    add_user,
    clear_all_data,
    delete_document,
    delete_user,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_all_documents,
    get_all_embeddings,
    get_all_users,
    get_chunk_registry,
    get_deleted_documents_count,
    get_document_by_hash,
    get_document_chunks_count,
    get_documents_by_class,
    get_incidents_by_assignment,
    get_unique_class_sections,
    get_user_active_status,
    get_user_role,
    init_corpus_db,
    init_db,
    is_user_active,
    restore_document,
    set_user_active_status,
    soft_delete_document,
    update_password,
    update_user_profile,
    verify_user,
)

try:
    from .visualization import (
        build_network_data,
        plot_chunk_similarity_comparison,
        plot_document_similarity_heatmap,
        plot_similarity_heatmap,
        plot_similarity_heatmap_plotly,
        plot_similarity_network,
        render_network_plotly,
    )
except ImportError:
    build_network_data = None
    plot_chunk_similarity_comparison = None
    plot_similarity_heatmap = None
    plot_similarity_heatmap_plotly = None
    plot_similarity_network = None
    render_network_plotly = None
    plot_document_similarity_heatmap = None


__all__ = [
    "extract_text_from_pdf",
    "extract_texts_from_pdfs",
    "extract_text",
    "extract_texts",
    "chunk_document",
    "chunk_documents",
    "chunk_by_sentences",
    "embed_chunks",
    "embed_documents",
    "get_document_embedding",
    "document_similarity_matrix",
    "chunk_similarity_matrix",
    "flag_plagiarism",
    "find_most_similar_chunks",
    "calculate_paragraph_similarity_breakdown",
    "PLAGIARISM_THRESHOLD",
    "plot_similarity_heatmap",
    "plot_similarity_heatmap_plotly",
    "plot_document_similarity_heatmap",
    "filter_heatmap_by_class_tag",
    "plot_chunk_similarity_comparison",
    "build_network_data",
    "export_graph_to_csv",
    "export_network_to_csv_bytes",
    "render_network_plotly",
    "plot_similarity_network",
    "translate_text",
    "send_plagiarism_alert",
    "dispatch_plagiarism_alert",
    "build_index",
    "search_similar_chunks",
    "find_plagiarised_chunks",
    "save_index",
    "load_index",
    "ChunkRecord",
    "FaissChunkRecord",
    "PipelineChunkRecord",
    "build_index_from_matrix",
    "init_db",
    "verify_user",
    "get_user_role",
    "get_all_users",
    "add_user",
    "delete_user",
    "update_password",
    "get_2fa_status",
    "enable_2fa",
    "disable_2fa",
    "get_user_active_status",
    "set_user_active_status",
    "is_user_active",
    "update_user_profile",
    "get_deleted_documents_count",
    "get_incidents_by_assignment",
    "init_corpus_db",
    "add_document",
    "get_document_by_hash",
    "get_all_documents",
    "soft_delete_document",
    "restore_document",
    "add_chunks",
    "get_chunk_registry",
    "get_all_embeddings",
    "delete_document",
    "clear_all_data",
    "get_document_chunks_count",
    "get_unique_class_sections",
    "get_documents_by_class",
    "BrandingConfig",
    "get_branding_config",
    "reload_branding_config",
    "load_branding_config",
    "sanitize_zero_width_characters",
    "TagManager",
    "sanitize_tag_name",
    "CorpusRepository",
]
