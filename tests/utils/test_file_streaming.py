"""
test_file_streaming.py
---------------------
Unit tests for stream_upload_file_to_disk in src/utils/file_streaming.py.
"""

import io
import os
import pytest
from fastapi import HTTPException, UploadFile
from src.utils.file_streaming import stream_upload_file_to_disk


@pytest.mark.asyncio
async def test_stream_upload_file_to_disk_success():
    content = b"Hello, this is a test payload for streaming upload." * 10
    upload_file = UploadFile(filename="test.txt", file=io.BytesIO(content))

    temp_path = await stream_upload_file_to_disk(upload_file, max_bytes=100000)

    try:
        assert os.path.exists(temp_path)
        with open(temp_path, "rb") as f:
            saved_content = f.read()
        assert saved_content == content
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_stream_upload_file_to_disk_empty_file_returns_400():
    upload_file = UploadFile(filename="empty.txt", file=io.BytesIO(b""))

    with pytest.raises(HTTPException) as exc_info:
        await stream_upload_file_to_disk(upload_file)

    assert exc_info.value.status_code == 400
    assert "empty" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_stream_upload_file_to_disk_exceeds_max_bytes_returns_413():
    content = b"X" * 1000  # 1000 bytes
    upload_file = UploadFile(filename="large.txt", file=io.BytesIO(content))

    with pytest.raises(HTTPException) as exc_info:
        await stream_upload_file_to_disk(upload_file, max_bytes=500, chunk_size=128)

    assert exc_info.value.status_code == 413
    assert "exceeds maximum" in str(exc_info.value.detail).lower()
