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

import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token

client = TestClient(app)


@patch("src.api.app.get_document_by_hash")
@patch("src.api.app.calculate_file_sha256")
@patch("src.api.app.get_corpus_documents_with_embeddings")
@patch("src.api.app.embed_chunks")
def test_scan_duplicate_rejected(mock_embed, mock_corpus, mock_hash, mock_get_doc):
    """Verify that a duplicate upload returns 409 Conflict when reprocess=False."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    expected_token = get_expected_bearer_token()
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 409
    assert response.json()["duplicate"] is True
    assert "already been uploaded" in response.json()["message"]


@patch("src.api.app.get_document_by_hash")
@patch("src.api.app.calculate_file_sha256")
@patch("src.api.app.get_corpus_documents_with_embeddings")
@patch("src.api.app.embed_chunks")
def test_scan_duplicate_reprocess(mock_embed, mock_corpus, mock_hash, mock_get_doc):
    """Verify that a duplicate upload with reprocess=True succeeds."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    import numpy as np

    mock_embed.return_value = np.ones((1, 384), dtype=np.float32)
    mock_corpus.return_value = {}

    expected_token = get_expected_bearer_token()
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan?reprocess=true",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 200
    assert "plagiarism_flagged" in response.json()


@patch("src.api.app.get_document_by_hash")
@patch("src.api.app.calculate_file_sha256")
def test_scan_async_duplicate_rejected(mock_hash, mock_get_doc):
    """Verify that async duplicate upload returns 409 Conflict when reprocess=False."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    expected_token = get_expected_bearer_token()
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan/async",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 409
    assert response.json()["duplicate"] is True
    assert "already been uploaded" in response.json()["message"]


@patch("src.api.app.get_document_by_hash")
@patch("src.api.app.calculate_file_sha256")
def test_scan_async_duplicate_reprocess(mock_hash, mock_get_doc):
    """Verify that async duplicate upload with reprocess=True succeeds."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    expected_token = get_expected_bearer_token()
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan/async?reprocess=true",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
