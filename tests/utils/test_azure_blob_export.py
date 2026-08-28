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
tests/utils/test_azure_blob_export.py
-------------------------------------
Tests for Azure Blob Storage report export utilities (#3465).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.azure_blob_export import (
    AZURE_STORAGE_CONNECTION_STRING_ENV,
    _get_connection_string,
    upload_to_azure_blob,
)

try:
    from azure.core.exceptions import ResourceExistsError
except ImportError:

    class ResourceExistsError(Exception):
        """Fallback stand-in used when azure-storage-blob is not installed."""


CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=acc;AccountKey=a2V5;"
    "EndpointSuffix=core.windows.net"
)


@pytest.fixture
def fake_service_client(monkeypatch):
    """
    Patch the module-level ``BlobServiceClient`` with a mock, provide the
    connection string via the environment, and yield the mock SDK class.
    Tests can reach the service client through ``from_connection_string``.
    ``ResourceExistsError`` is patched alongside the client so the module
    behaves identically whether or not azure-storage-blob is installed.
    """
    monkeypatch.setenv(AZURE_STORAGE_CONNECTION_STRING_ENV, CONNECTION_STRING)

    sdk_mock = MagicMock(name="BlobServiceClient")
    blob_client = sdk_mock.from_connection_string.return_value.get_container_client.return_value.get_blob_client.return_value
    blob_client.url = "https://acc.blob.core.windows.net/reports/report-42.pdf"

    with (
        patch("src.utils.azure_blob_export.BlobServiceClient", sdk_mock),
        patch("src.utils.azure_blob_export.ResourceExistsError", ResourceExistsError),
    ):
        yield sdk_mock


@pytest.fixture
def service_client(fake_service_client):
    """The mock client returned by ``BlobServiceClient.from_connection_string``."""
    return fake_service_client.from_connection_string.return_value


def test_get_connection_string_reads_environment(monkeypatch):
    monkeypatch.setenv(AZURE_STORAGE_CONNECTION_STRING_ENV, CONNECTION_STRING)
    assert _get_connection_string() == CONNECTION_STRING


def test_get_connection_string_override_wins(monkeypatch):
    monkeypatch.setenv(AZURE_STORAGE_CONNECTION_STRING_ENV, "env-value")
    assert _get_connection_string("override-value") == "override-value"


def test_get_connection_string_missing_raises(monkeypatch):
    monkeypatch.delenv(AZURE_STORAGE_CONNECTION_STRING_ENV, raising=False)
    with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
        _get_connection_string()


def test_upload_returns_blob_url(service_client):
    url = upload_to_azure_blob(b"report-bytes", "reports", "report-42.pdf")
    assert url == "https://acc.blob.core.windows.net/reports/report-42.pdf"
    blob_client = (
        service_client.get_container_client.return_value.get_blob_client.return_value
    )
    blob_client.upload_blob.assert_called_once_with(b"report-bytes", overwrite=True)


def test_upload_passes_connection_string_to_sdk(fake_service_client, service_client):
    fake_service_client.from_connection_string.reset_mock()
    upload_to_azure_blob(b"data", "container", "blob.txt")
    fake_service_client.from_connection_string.assert_called_once_with(
        CONNECTION_STRING
    )
    container_client = service_client.get_container_client
    container_client.assert_called_once_with("container")
    container_client.return_value.get_blob_client.assert_called_once_with("blob.txt")


def test_upload_creates_container_by_default(service_client):
    upload_to_azure_blob(b"report", "reports", "report.pdf")
    container_client = service_client.get_container_client.return_value
    container_client.create_container.assert_called_once_with()


def test_upload_can_skip_container_creation(service_client):
    upload_to_azure_blob(b"report", "reports", "report.pdf", create_container=False)
    container_client = service_client.get_container_client.return_value
    container_client.create_container.assert_not_called()


def test_upload_existing_container_is_tolerated(service_client):
    container_client = service_client.get_container_client.return_value
    container_client.create_container.side_effect = ResourceExistsError("exists")
    url = upload_to_azure_blob(b"report", "reports", "report.pdf")
    assert url  # Upload still succeeds.


def test_upload_without_overwrite(service_client):
    upload_to_azure_blob(b"report", "reports", "report.pdf", overwrite=False)
    blob_client = (
        service_client.get_container_client.return_value.get_blob_client.return_value
    )
    blob_client.upload_blob.assert_called_once_with(b"report", overwrite=False)


def test_upload_missing_sdk_raises_import_error(monkeypatch):
    monkeypatch.setenv(AZURE_STORAGE_CONNECTION_STRING_ENV, CONNECTION_STRING)
    with patch("src.utils.azure_blob_export.BlobServiceClient", None):
        with pytest.raises(ImportError, match="azure-storage-blob"):
            upload_to_azure_blob(b"report", "reports", "report.pdf")


def test_upload_missing_environment_variable(monkeypatch):
    monkeypatch.delenv(AZURE_STORAGE_CONNECTION_STRING_ENV, raising=False)
    with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
        upload_to_azure_blob(b"report", "reports", "report.pdf")


@pytest.mark.parametrize(
    "file_bytes",
    [b"bytes", bytearray(b"bytearray"), memoryview(b"memoryview")],
)
def test_upload_accepts_bytes_like_input(file_bytes, service_client):
    url = upload_to_azure_blob(file_bytes, "reports", "report.pdf")
    assert url


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"file_bytes": "text"}, TypeError),
        ({"file_bytes": b""}, ValueError),
        ({"container_name": ""}, ValueError),
        ({"container_name": "   "}, ValueError),
        ({"container_name": None}, TypeError),
        ({"blob_name": ""}, ValueError),
        ({"blob_name": "  "}, ValueError),
        ({"blob_name": 123}, TypeError),
    ],
)
def test_upload_rejects_invalid_inputs(kwargs, expected, service_client):
    call_kwargs = {
        "file_bytes": b"data",
        "container_name": "reports",
        "blob_name": "report.pdf",
    }
    call_kwargs.update(kwargs)
    with pytest.raises(expected):
        upload_to_azure_blob(**call_kwargs)
