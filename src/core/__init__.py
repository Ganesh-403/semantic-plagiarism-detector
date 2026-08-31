from typing import TYPE_CHECKING

from .config import (
    BrandingConfig,
    get_branding_config,
    load_branding_config,
    reload_branding_config,
)
from .document_parser import (
    check_ocr_dependencies,
    extract_text,
    extract_text_from_pdf,
    extract_texts,
    extract_texts_from_pdfs,
    sanitize_zero_width_characters,
)
from .embedding_model import embed_chunks, embed_documents, get_document_embedding
from .faiss_index import (
    FAISSIndex,
    FaissIndexManager,
    ChunkRecord,
    FaissChunkRecord,
    build_index,
    build_index_from_matrix,
    find_plagiarised_chunks,
    format_faiss_memory_badge,
    get_faiss_index_memory_bytes,
    load_index,
    rebuild_index_from_database,
    rebuild_index_from_db,
    save_index,
    search_similar_chunks,
)
from .pipeline import (
    PipelineChunkRecord,
    run_extraction_pipeline,
    run_pipeline,
)
from .similarity import (
    PLAGIARISM_THRESHOLD,
    calculate_paragraph_similarity_breakdown,
    chunk_similarity_matrix,
    document_similarity_matrix,
    find_most_similar_chunks,
    flag_plagiarism,
    manhattan_similarity,
)
from .similarity_base import BaseSimilarityEngine
from .similarity_engines import (
    HybridSimilarityEngine,
    LexicalSimilarityEngine,
    SemanticSimilarityEngine,
    SimilarityEngineFactory,
)
from .tag_manager import TagManager, sanitize_tag_name
from .text_chunking import Chunk, ChunkString, chunk_by_sentences, chunk_document, chunk_documents
from .translator import translate_text
from .webhook import (
    EventDispatcher,
    dispatch_plagiarism_alert,
    send_plagiarism_alert,
)

# TYPE_CHECKING block for lazy imports (Issue #2363)
# This satisfies static analysis tools (mypy, pylance) that would otherwise
# complain that src.core has no attribute 'with_sqlite_retry', even though
# it's dynamically resolved via __getattr__ and listed in __all__.
if TYPE_CHECKING:
    from .concurrency import with_sqlite_retry

__all__ = [
    "BaseSimilarityEngine",
    "BrandingConfig",
    "Chunk",
    "ChunkRecord",
    "ChunkString",
    "EventDispatcher",
    "FAISSIndex",
    "FaissChunkRecord",
    "FaissIndexManager",
    "HybridSimilarityEngine",
    "LexicalSimilarityEngine",
    "PLAGIARISM_THRESHOLD",
    "PipelineChunkRecord",
    "SemanticSimilarityEngine",
    "SimilarityEngineFactory",
    "TagManager",
    "build_index",
    "build_index_from_matrix",
    "calculate_paragraph_similarity_breakdown",
    "chunk_by_sentences",
    "chunk_document",
    "chunk_documents",
    "chunk_similarity_matrix",
    "dispatch_plagiarism_alert",
    "document_similarity_matrix",
    "embed_chunks",
    "embed_documents",
    "extract_text",
    "extract_text_from_pdf",
    "extract_texts",
    "extract_texts_from_pdfs",
    "find_most_similar_chunks",
    "find_plagiarised_chunks",
    "flag_plagiarism",
    "format_faiss_memory_badge",
    "get_branding_config",
    "get_document_embedding",
    "get_faiss_index_memory_bytes",
    "load_branding_config",
    "load_index",
    "manhattan_similarity",
    "rebuild_index_from_database",
    "rebuild_index_from_db",
    "reload_branding_config",
    "run_extraction_pipeline",
    "run_pipeline",
    "sanitize_tag_name",
    "sanitize_zero_width_characters",
    "save_index",
    "search_similar_chunks",
    "send_plagiarism_alert",
    "translate_text",
    "with_sqlite_retry",
]


# with_sqlite_retry is re-exported from src.core.concurrency (which lazily
# re-exports it from src.db.common). A lazy lookup avoids the circular import
# chain (src.db -> src.core -> src.db.common).
def __getattr__(name):
    if name == "with_sqlite_retry":
        from .concurrency import with_sqlite_retry

        return with_sqlite_retry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
