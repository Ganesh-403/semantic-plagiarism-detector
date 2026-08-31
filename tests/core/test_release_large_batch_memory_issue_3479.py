"""
tests/core/test_release_large_batch_memory_issue_3479.py
--------------------------------------------------------
Tests for explicit garbage collection after large batch scans (Issue #3479).

Verifies that release_large_batch_memory():
* does nothing for batches at or below the 20-item threshold,
* calls gc.collect() for larger batches,
* releases the CUDA cache via torch.cuda.empty_cache() only when CUDA is
  available,
* is invoked by _process_scan_job once a large scan job finishes.
"""

from unittest.mock import patch

import numpy as np

from src.core.embedding_model import (
    LARGE_BATCH_GC_THRESHOLD,
    release_large_batch_memory,
)


def _patch_cuda(available: bool):
    """Patch torch.cuda availability checks inside embedding_model."""
    return (
        patch("src.core.embedding_model.torch.cuda.is_available", return_value=available),
        patch("src.core.embedding_model.torch.cuda.empty_cache"),
    )


def test_threshold_is_twenty():
    """The acceptance criteria for Issue #3479 specify a batch size of 20."""
    assert LARGE_BATCH_GC_THRESHOLD == 20


def test_noop_at_or_below_threshold():
    """No cleanup should run for batches at or below the threshold."""
    is_available, empty_cache = _patch_cuda(available=True)
    with patch(
        "src.core.embedding_model.gc.collect"
    ) as gc_collect, is_available as _, empty_cache as empty_cache_mock:
        release_large_batch_memory(LARGE_BATCH_GC_THRESHOLD)
        release_large_batch_memory(LARGE_BATCH_GC_THRESHOLD - 5)
        release_large_batch_memory(0)
        gc_collect.assert_not_called()
        empty_cache_mock.assert_not_called()


def test_gc_collect_runs_above_threshold_without_cuda():
    """gc.collect() runs for large batches; CUDA cache is untouched without CUDA."""
    is_available, empty_cache = _patch_cuda(available=False)
    with patch(
        "src.core.embedding_model.gc.collect"
    ) as gc_collect, is_available as _, empty_cache as empty_cache_mock:
        release_large_batch_memory(LARGE_BATCH_GC_THRESHOLD + 1)
        gc_collect.assert_called_once()
        empty_cache_mock.assert_not_called()


def test_cuda_cache_released_when_available():
    """Both gc.collect() and torch.cuda.empty_cache() run when CUDA exists."""
    is_available, empty_cache = _patch_cuda(available=True)
    with patch(
        "src.core.embedding_model.gc.collect"
    ) as gc_collect, is_available as _, empty_cache as empty_cache_mock:
        release_large_batch_memory(150)
        gc_collect.assert_called_once()
        empty_cache_mock.assert_called_once()


def setup_function():
    from src.api.routers.analysis import scan_jobs

    scan_jobs.clear()


def teardown_function():
    from src.api.routers.analysis import scan_jobs

    scan_jobs.clear()


def test_process_scan_job_releases_memory_for_large_batches():
    """_process_scan_job must call release_large_batch_memory(len(chunks))."""
    from src.api.routers.analysis import _process_scan_job, scan_jobs

    job_id = "job_gc_large_batch_1"
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_percent": 0,
        "stage": "",
        "filename": "large.txt",
        "created_at": "2026-08-23T20:00:00Z",
        "completed_at": None,
        "result": None,
        "error": None,
    }

    chunks = [f"chunk number {i} with some text" for i in range(25)]

    with patch(
        "src.api.routers.analysis.extract_text", return_value="sample document text"
    ), patch(
        "src.api.routers.analysis.chunk_document", return_value=chunks
    ), patch(
        "src.api.routers.analysis.embed_chunks",
        return_value=np.ones((25, 384), dtype=np.float32),
    ), patch(
        "src.api.routers.analysis.get_corpus_documents_with_embeddings",
        return_value={},
    ), patch(
        "src.api.routers.analysis.release_large_batch_memory"
    ) as release_mock:
        _process_scan_job(
            job_id=job_id,
            file_input="fake_path",
            filename="large.txt",
            threshold=0.59,
            top_k=3,
        )

    release_mock.assert_called_once_with(25)
    assert scan_jobs[job_id]["status"] == "completed"


def test_process_scan_job_skips_release_for_small_batches():
    """Small scans stay under the threshold, so cleanup must not trigger."""
    from src.api.routers.analysis import _process_scan_job, scan_jobs

    job_id = "job_gc_small_batch_1"
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_percent": 0,
        "stage": "",
        "filename": "small.txt",
        "created_at": "2026-08-23T20:00:00Z",
        "completed_at": None,
        "result": None,
        "error": None,
    }

    chunks = [f"chunk {i}" for i in range(5)]

    with patch(
        "src.api.routers.analysis.extract_text", return_value="sample document text"
    ), patch(
        "src.api.routers.analysis.chunk_document", return_value=chunks
    ), patch(
        "src.api.routers.analysis.embed_chunks",
        return_value=np.ones((5, 384), dtype=np.float32),
    ), patch(
        "src.api.routers.analysis.get_corpus_documents_with_embeddings",
        return_value={},
    ), patch(
        "src.api.routers.analysis.release_large_batch_memory"
    ) as release_mock:
        _process_scan_job(
            job_id=job_id,
            file_input="fake_path",
            filename="small.txt",
            threshold=0.59,
            top_k=3,
        )

    release_mock.assert_called_once_with(5)
    assert scan_jobs[job_id]["status"] == "completed"
