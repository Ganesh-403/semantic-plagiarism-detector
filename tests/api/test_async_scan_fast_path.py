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

import os
import uuid
from datetime import datetime, timezone
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from src.api.routers.analysis import _process_scan_job, scan_jobs
from src.core.embedding_model import embed_chunks


def test_exact_hash_match_fast_path(tmp_path):
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    filename = "test_exact.txt"
    file_path = tmp_path / filename
    file_path.write_text("Hello world exact match")

    # Pre-populate scan_jobs as if enqueued
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_percent": 0,
        "stage": "",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    with mock.patch("src.api.routers.analysis.calculate_file_sha256") as mock_calc:
        with mock.patch(
            "src.api.routers.analysis.get_document_by_hash"
        ) as mock_get_hash:
            with mock.patch("src.api.routers.analysis.embed_chunks") as mock_embed:
                with mock.patch(
                    "src.api.routers.analysis.get_document_embedding"
                ) as mock_doc_embed:
                    mock_calc.return_value = "fake_sha256"
                    mock_get_hash.return_value = "corpus_doc_match.txt"

                    _process_scan_job(
                        job_id=job_id,
                        file_input=str(file_path),
                        filename=filename,
                        threshold=0.59,
                        top_k=3,
                    )

                    # Assertions
                    mock_calc.assert_called_once_with(str(file_path))
                    mock_get_hash.assert_called_once_with("fake_sha256")

                    # Ensure expensive steps are completely bypassed
                    mock_embed.assert_not_called()
                    mock_doc_embed.assert_not_called()

                    job = scan_jobs[job_id]
                    assert job["status"] == "completed"
                    assert job["progress_percent"] == 100

                    res = job["result"]
                    assert res["filename"] == filename
                    assert res["plagiarism_flagged"] is True
                    assert res["plagiarism_density"] == 100
                    assert res["overall_document_similarity"] == 1.0
                    assert res["max_chunk_similarity"] == 1.0
                    assert res["matched_documents_count"] == 1

                    match = res["matched_documents"][0]
                    assert match["filename"] == "corpus_doc_match.txt"
                    assert match["document_similarity_score"] == 1.0
                    assert match["max_chunk_similarity_score"] == 1.0
                    assert match["severity"] == "🔴 High"


def test_no_hash_match_fallback(tmp_path):
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    filename = "test_no_match.txt"
    file_path = tmp_path / filename
    file_path.write_text("Hello world no match")

    # Pre-populate scan_jobs as if enqueued
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_percent": 0,
        "stage": "",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    with mock.patch("src.api.routers.analysis.calculate_file_sha256") as mock_calc:
        with mock.patch(
            "src.api.routers.analysis.get_document_by_hash"
        ) as mock_get_hash:
            with mock.patch("src.api.routers.analysis.embed_chunks") as mock_embed:
                with mock.patch(
                    "src.api.routers.analysis.get_document_embedding"
                ) as mock_doc_embed:
                    with mock.patch(
                        "src.api.routers.analysis.get_corpus_documents_with_embeddings"
                    ) as mock_corpus:
                        mock_calc.return_value = "fake_sha256"
                        mock_get_hash.return_value = None  # No match

                        mock_embed.return_value = mock.MagicMock()
                        mock_doc_embed.return_value = mock.MagicMock()
                        mock_corpus.return_value = {}

                        _process_scan_job(
                            job_id=job_id,
                            file_input=str(file_path),
                            filename=filename,
                            threshold=0.59,
                            top_k=3,
                        )

                        mock_calc.assert_called_once_with(str(file_path))
                        mock_get_hash.assert_called_once_with("fake_sha256")

                        # Existing pipeline should be invoked
                        assert mock_embed.called
                        assert mock_doc_embed.called
                        assert mock_corpus.called
