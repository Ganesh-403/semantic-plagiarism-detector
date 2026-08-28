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
src/utils/azure_blob_export.py
------------------------------
Utilities for exporting plagiarism reports to Azure Blob Storage.

Institutions hosted on Microsoft Azure can persist generated report files
(PDF, HTML, JSON, ...) in a storage container instead of relying on local
disk. This module uploads raw report bytes to a container using the
``azure-storage-blob`` SDK and authenticates through the standard
``AZURE_STORAGE_CONNECTION_STRING`` environment variable.
"""

import logging
import os
from typing import Optional

try:
    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob import BlobServiceClient
except ImportError:  # pragma: no cover - exercised only without the SDK installed
    BlobServiceClient = None
    ResourceExistsError = None

logger = logging.getLogger(__name__)

#: Environment variable holding the Azure Storage connection string.
AZURE_STORAGE_CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"


def _get_connection_string(override: Optional[str] = None) -> str:
    """
    Resolve the Azure Storage connection string.

    Args:
        override: Explicit connection string supplied by the caller. When
            provided it takes precedence over the environment variable.

    Returns:
        The resolved connection string.

    Raises:
        ValueError: If no connection string was supplied and the
            ``AZURE_STORAGE_CONNECTION_STRING`` environment variable is not set.
    """
    if override:
        return override

    connection_string = os.getenv(AZURE_STORAGE_CONNECTION_STRING_ENV)
    if not connection_string:
        raise ValueError(
            f"Azure Storage connection string is missing. Set the "
            f"{AZURE_STORAGE_CONNECTION_STRING_ENV} environment variable or "
            f"pass an explicit connection string."
        )
    return connection_string


def _validate_upload_inputs(
    file_bytes: bytes, container_name: str, blob_name: str
) -> None:
    """
    Validate the arguments passed to :func:`upload_to_azure_blob`.

    Args:
        file_bytes: Raw file content to upload.
        container_name: Target Azure Blob Storage container name.
        blob_name: Name (key) assigned to the uploaded blob.

    Raises:
        TypeError: If ``file_bytes`` is not a bytes-like object or the names
            are not strings.
        ValueError: If any argument is empty after validation.
    """
    if isinstance(file_bytes, (bytearray, memoryview)):
        file_bytes = bytes(file_bytes)
    if not isinstance(file_bytes, bytes):
        raise TypeError("file_bytes must be a bytes-like object.")
    if not file_bytes:
        raise ValueError("file_bytes must not be empty.")
    if not isinstance(container_name, str):
        raise TypeError("container_name must be a string.")
    if not container_name.strip():
        raise ValueError("container_name must not be empty.")
    if not isinstance(blob_name, str):
        raise TypeError("blob_name must be a string.")
    if not blob_name.strip():
        raise ValueError("blob_name must not be empty.")


def _get_blob_service_client(connection_string: str):
    """
    Build an Azure ``BlobServiceClient`` from a connection string.

    Args:
        connection_string: Azure Storage account connection string.

    Returns:
        A configured ``BlobServiceClient`` instance.

    Raises:
        ImportError: If the ``azure-storage-blob`` package is not installed.
    """
    if BlobServiceClient is None:
        raise ImportError(
            "The 'azure-storage-blob' package is required for Azure Blob "
            "export. Install it with: pip install azure-storage-blob"
        )
    return BlobServiceClient.from_connection_string(connection_string)


def upload_to_azure_blob(
    file_bytes: bytes,
    container_name: str,
    blob_name: str,
    connection_string: Optional[str] = None,
    create_container: bool = True,
    overwrite: bool = True,
) -> str:
    """
    Upload raw report bytes to an Azure Blob Storage container.

    Reads the storage account credentials from the
    ``AZURE_STORAGE_CONNECTION_STRING`` environment variable (unless an
    explicit ``connection_string`` is provided), ensures the target container
    exists, and uploads ``file_bytes`` as ``blob_name``.

    Args:
        file_bytes: Raw content of the report file to upload.
        container_name: Name of the target blob container. Created when it
            does not exist and ``create_container`` is True.
        blob_name: Name (key) assigned to the uploaded blob inside the
            container, e.g. ``"reports/2026/report-42.pdf"``.
        connection_string: Optional explicit connection string that overrides
            the ``AZURE_STORAGE_CONNECTION_STRING`` environment variable.
        create_container: When True (default), create the container if it is
            missing instead of failing.
        overwrite: When True (default), replace an existing blob of the same
            name instead of raising.

    Returns:
        The public URL of the uploaded blob.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If arguments are empty or the connection string cannot be
            resolved from the environment.
        ImportError: If the ``azure-storage-blob`` package is not installed.
        azure.core.exceptions.ResourceExistsError: If the container already
            exists and ``create_container`` handling did not suppress it.
    """
    _validate_upload_inputs(file_bytes, container_name, blob_name)

    resolved_connection_string = _get_connection_string(connection_string)
    service_client = _get_blob_service_client(resolved_connection_string)

    logger.info(
        "Uploading report to Azure Blob Storage (container=%s, blob=%s)",
        container_name,
        blob_name,
    )

    container_client = service_client.get_container_client(container_name)
    if create_container:
        try:
            container_client.create_container()
        except ResourceExistsError:
            # Container already exists - nothing to do.
            logger.debug("Container '%s' already exists", container_name)

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(file_bytes, overwrite=overwrite)

    logger.info(
        "Successfully uploaded blob '%s' to container '%s'", blob_name, container_name
    )
    return blob_client.url
