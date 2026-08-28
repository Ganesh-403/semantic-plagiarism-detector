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
src/utils/file_streaming.py
---------------------------
Streaming multipart chunk reader utility for handling large file uploads.
Streams incoming payload chunks directly to a temporary file on disk using UploadFile,
enforcing maximum upload file size bounds during streaming.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import HTTPException, UploadFile, status

DEFAULT_MAX_UPLOAD_BYTES = 52_428_800  # Default 50 MB
CHUNK_SIZE = 64 * 1024  # 64 KB chunk size


async def stream_upload_file_to_disk(
    upload_file: UploadFile,
    max_bytes: Optional[int] = None,
    chunk_size: int = CHUNK_SIZE,
) -> str:
    """
    Stream incoming upload payload chunks directly to a temporary file on disk.

    Args:
        upload_file: FastAPI UploadFile instance containing stream handle.
        max_bytes: Maximum allowed size in bytes (defaults to DEFAULT_MAX_UPLOAD_BYTES).
        chunk_size: Streaming chunk buffer size in bytes (default 64KB).

    Returns:
        Absolute string path to the created temporary file on disk.

    Raises:
        HTTPException 400: If uploaded file is empty (0 bytes).
        HTTPException 413: If total size exceeds max_bytes during chunk streaming.
    """
    effective_max_bytes = (
        max_bytes
        if max_bytes is not None
        else int(os.getenv("MAX_UPLOAD_SIZE_BYTES", DEFAULT_MAX_UPLOAD_BYTES))
    )

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = temp_file.name
    total_bytes = 0

    try:
        # Seek to beginning of upload stream if supported
        if hasattr(upload_file, "seek"):
            try:
                await upload_file.seek(0)
            except Exception:
                pass

        while True:
            # Read chunk in 64KB increments directly from upload_file
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break

            total_bytes += len(chunk)
            if effective_max_bytes is not None and total_bytes > effective_max_bytes:
                temp_file.close()
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Uploaded file size ({total_bytes} bytes) exceeds maximum allowed limit of {effective_max_bytes} bytes.",
                )

            temp_file.write(chunk)

        temp_file.close()

        if total_bytes == 0:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)",
            )

        return temp_path

    except HTTPException:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        raise
    except Exception as exc:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream uploaded file: {str(exc)}",
        ) from exc
