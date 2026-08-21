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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.core.ai_detector import detect_documents_ai_probability
from src.core.config import PLAGIARISM_THRESHOLD, severity_from_score
from src.core.document_parser import extract_text
from src.core.embedding_model import embed_documents
from src.core.faiss_index import ChunkRecord, build_index
from src.core.similarity import document_similarity_matrix, flag_plagiarism
from src.core.text_chunking import chunk_documents
from src.utils.tracing import get_tracer

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    """Named outputs from ``run_full_pipeline`` (still unpackable as a tuple)."""

    raw_texts: Dict[str, str]
    chunked_docs: Dict[str, List[str]]
    embeddings: Dict[str, np.ndarray]
    sim_df: pd.DataFrame
    chunk_sim_df: pd.DataFrame
    faiss_index: Any
    registry: List[ChunkRecord]
    ai_probabilities: Dict[str, Dict[str, Any]]
    flags: List[Dict[str, Any]]


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
        A ``PipelineResult`` with all pipeline outputs including the final flags list.
    """
    tracer = get_tracer()

    with tracer.start_as_current_span("run_full_pipeline") as root_span:
        root_span.set_attribute("file.count", len(file_bytes_dict))

        try:
            import psutil
        except ImportError:
            psutil = None
            logger.debug("psutil is not installed; skipping system memory usage check.")

        raw_texts: Dict[str, str] = {}
        failed_files: List[str] = []
        failure_details: List[str] = []

        with tracer.start_as_current_span("pipeline.parse") as parse_span:
            for name, data in file_bytes_dict.items():
                if not data:
                    continue
                try:
                    raw_texts[name] = extract_text(
                        io.BytesIO(data),
                        name,
                        ocr_language=ocr_language,
                        ocr_dpi=ocr_dpi,
                    )
                except Exception as exc:
                    failed_files.append(name)
                    failure_details.append(f"{name}: {exc}")

            if url_text and url_filename:
                raw_texts[url_filename] = url_text

            if failed_files:
                from src.exceptions import OCRFileBatchError

                raise OCRFileBatchError(failed_files, failure_details)
            if ignore_phrases and ignore_phrases.strip():
                from src.core.document_parser import remove_ignore_phrases

                raw_texts = {
                    name: remove_ignore_phrases(text, ignore_phrases)
                    for name, text in raw_texts.items()
                }
            parse_span.set_attribute("parsed.count", len(raw_texts))

        with tracer.start_as_current_span("pipeline.chunk") as chunk_span:
            chunked_docs = chunk_documents(
                raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            total_chunks = sum(len(chunks) for chunks in chunked_docs.values())
            chunk_span.set_attribute("chunk.count", total_chunks)

        with tracer.start_as_current_span("pipeline.embed") as embed_span:
            embeddings = embed_documents(chunked_docs)
            first_emb = next(iter(embeddings.values()), None)
            embed_span.set_attribute(
                "embedding.dims",
                first_emb.shape[1] if first_emb is not None and first_emb.size else 0,
            )

        with tracer.start_as_current_span("pipeline.similarity_scoring") as sim_span:
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

        if psutil is not None:
            try:
                memory = psutil.virtual_memory()
                if memory.percent >= 85:
                    logger.warning(
                        "High memory usage detected (%d%%). Large FAISS indexes may cause "
                        "instability or OOM crashes.",
                        memory.percent,
                    )
            except Exception as exc:
                logger.debug("System memory usage check failed: %s", exc)

        with tracer.start_as_current_span("pipeline.faiss_search") as index_span:
            faiss_index, registry = build_index(embeddings, chunked_docs)
            index_span.set_attribute("faiss.index_size", len(registry))

        ai_probabilities = detect_documents_ai_probability(chunked_docs)

        flags = flag_plagiarism(
            sim_df,
            threshold=threshold,
            chunked_docs=chunked_docs,
            embeddings=embeddings,
        )

        with tracer.start_as_current_span("pipeline.incident_sync"):
            pass

        return PipelineResult(
            raw_texts=raw_texts,
            chunked_docs=chunked_docs,
            embeddings=embeddings,
            sim_df=sim_df,
            chunk_sim_df=chunk_sim_df,
            faiss_index=faiss_index,
            registry=registry,
            ai_probabilities=ai_probabilities,
            flags=flags,
        )


# ── Scheduled / continuous rescan pipeline ──────────────────────────────────
#
# Requirements addressed here (see issue: "Build a scheduled, event-driven
# plagiarism re-scan and notification pipeline"):
#   * Re-check newly-added documents against the *entire* corpus so that
#     cross-submission cheating that only becomes visible once a second,
#     later document is uploaded is still caught.
#   * Never create a duplicate incident for a pair that's already known.
#   * Only fire webhook alerts for genuinely NEW incidents.
#   * Never race a concurrent manual scan while touching the FAISS index.

RESCAN_JOB_NAME = "rescan_recent_documents"


class RescanResult(NamedTuple):
    """Summary of a single scheduled rescan pass."""

    documents_scanned: int
    candidate_pairs_checked: int
    new_incidents: List[Dict[str, Any]]
    total_flags: int


def _aggregate_chunk_matches_to_flags(
    matches: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Collapse chunk-level FAISS matches into one flag per document pair.

    ``find_plagiarised_chunks`` returns one row per matching chunk pair;
    a single pair of documents may share several similar paragraphs. This
    keeps only the strongest (max-similarity) match per document pair,
    mirroring the semantics of ``flag_plagiarism``.
    """
    best_per_pair: Dict[tuple, Dict[str, Any]] = {}

    for match in matches:
        doc_a = str(match.get("source_doc", "")).strip()
        doc_b = str(match.get("match_doc", "")).strip()
        if not doc_a or not doc_b or doc_a == doc_b:
            continue

        score = float(match.get("similarity", 0.0))
        if score < threshold:
            continue

        pair_key = tuple(sorted((doc_a, doc_b)))
        existing = best_per_pair.get(pair_key)
        if existing is None or score > existing["similarity"]:
            first, second = pair_key
            best_per_pair[pair_key] = {
                "doc_a": first,
                "doc_b": second,
                "similarity": round(score, 4),
                "threshold_at_time_of_flag": float(threshold),
                "severity": severity_from_score(score),
            }

    flags = list(best_per_pair.values())
    flags.sort(key=lambda item: item["similarity"], reverse=True)
    return flags


def rescan_recent_documents(
    grace_period: Optional[int] = None,
    threshold: float = PLAGIARISM_THRESHOLD,
    *,
    top_k: int = 10,
    now: Optional[datetime] = None,
    db_path: Optional[str] = None,
    dispatch_alerts: bool = True,
) -> RescanResult:
    """Re-check recently-added documents against the full corpus.

    This is the reusable entrypoint used by both the background scheduler
    (``src.core.scheduler``) and any manual/API-triggered rescan. It:

    1. Finds documents uploaded within the last ``grace_period`` minutes.
    2. Loads (or rebuilds) the full FAISS index and searches each recent
       document's existing chunk embeddings against it — catching matches
       against documents that were already in the corpus *before* the
       recent document arrived, which the original upload-time scan may
       have missed if the older document didn't exist yet, or vice versa.
    3. Deduplicates against known incidents (``build_incident_id``
       semantics) so a rerun after an incident already exists creates zero
       new rows.
    4. Persists any new incidents and dispatches a webhook alert only for
       the ones that are genuinely new.
    5. Records the run in the ``scheduler_runs`` table so the scheduler is
       restart-safe.

    The whole read-index/search/write-incidents sequence is guarded by the
    same ``FAISSLock`` used elsewhere, so a scheduled rescan and a manual
    scan can never corrupt the FAISS index by writing to it concurrently.

    Args:
        grace_period: How many minutes back to look for "recently added"
            documents. Defaults to ``get_rescan_grace_period_minutes()``.
        threshold: Similarity threshold above which a chunk match is
            considered plagiarism.
        top_k: Number of nearest-neighbour chunks to retrieve per query
            chunk when searching the FAISS index.
        now: Injectable clock for testing; defaults to the current UTC time.
        db_path: Optional override of the incidents database path (tests).
        dispatch_alerts: When False, incidents are still synced but no
            webhook alert is sent (used by callers that handle
            notification themselves).

    Returns:
        A ``RescanResult`` summarizing what the pass found.
    """
    from src.core.app_config import FAISS_INDEX_PATH, get_rescan_grace_period_minutes
    from src.core.concurrency import faiss_write_lock
    from src.core.faiss_index import find_plagiarised_chunks, load_or_rebuild_index
    from src.core.synchronization import run_background
    from src.core.webhook import dispatch_plagiarism_alert
    from src.db.corpus_db import get_chunks_for_documents, get_documents_since
    from src.db.incidents import (
        build_incident_id,
        get_existing_incident_pairs,
        record_scheduler_run,
        sync_flagged_incidents,
    )

    if grace_period is None:
        grace_period = get_rescan_grace_period_minutes()

    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(minutes=grace_period)
    cutoff_iso = cutoff.replace(microsecond=0).isoformat()

    recent_filenames = get_documents_since(cutoff_iso)

    if not recent_filenames:
        logger.info(
            "[rescan] No documents uploaded since %s; nothing to rescan.",
            cutoff_iso,
        )
        record_scheduler_run(
            RESCAN_JOB_NAME,
            db_path,
            documents_scanned=0,
            new_incidents=0,
        )
        return RescanResult(
            documents_scanned=0,
            candidate_pairs_checked=0,
            new_incidents=[],
            total_flags=0,
        )

    recent_chunks = get_chunks_for_documents(recent_filenames)
    # Documents that have no stored chunks yet (e.g. still mid-upload) are
    # skipped for this pass; they'll be picked up on a later tick once
    # their chunks/embeddings are durably persisted.
    recent_chunked_docs = {name: texts for name, (texts, _emb) in recent_chunks.items()}
    recent_embeddings = {name: emb for name, (_texts, emb) in recent_chunks.items()}

    new_incidents: List[Dict[str, Any]] = []
    all_flags: List[Dict[str, Any]] = []

    with faiss_write_lock(lock_path=f"{FAISS_INDEX_PATH}.lock"):
        if not recent_embeddings:
            matches: List[Dict[str, Any]] = []
        else:
            index, registry, _recovered = load_or_rebuild_index(str(FAISS_INDEX_PATH))
            matches = find_plagiarised_chunks(
                recent_embeddings,
                recent_chunked_docs,
                index,
                registry,
                threshold=threshold,
                top_k=top_k,
            )

        all_flags = _aggregate_chunk_matches_to_flags(matches, threshold)

        if all_flags:
            existing_pairs = get_existing_incident_pairs(db_path)
            for flag in all_flags:
                incident_id = build_incident_id(flag["doc_a"], flag["doc_b"])
                if incident_id not in existing_pairs:
                    new_incidents.append(flag)

            # sync_flagged_incidents upserts via ON CONFLICT, so re-running
            # this against already-known pairs is always a no-op for rows
            # that already exist (only similarity/last_seen are refreshed).
            sync_flagged_incidents(all_flags, db_path, threshold=threshold)

    if dispatch_alerts:
        for flag in new_incidents:
            run_background(
                dispatch_plagiarism_alert,
                flag["doc_a"],
                flag["doc_b"],
                flag["similarity"],
            )

    record_scheduler_run(
        RESCAN_JOB_NAME,
        db_path,
        now=current_time.replace(microsecond=0).isoformat(),
        documents_scanned=len(recent_filenames),
        new_incidents=len(new_incidents),
    )

    logger.info(
        "[rescan] Scanned %d recent document(s), found %d flag(s), %d new incident(s).",
        len(recent_filenames),
        len(all_flags),
        len(new_incidents),
    )

    return RescanResult(
        documents_scanned=len(recent_filenames),
        candidate_pairs_checked=len(all_flags),
        new_incidents=new_incidents,
        total_flags=len(all_flags),
    )
