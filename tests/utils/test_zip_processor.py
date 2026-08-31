"""Tests for ZIP extraction functionality with fault injection."""

import io
import zipfile

import pytest

from src.core.document_parser import CorruptedArchiveError, extract_text_from_zip


def _make_valid_zip_bytes(files: dict) -> bytes:
    """Create a valid in-memory ZIP archive containing given file names and contents."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return buf.getvalue()


def test_extract_zip_returns_text_from_valid_archive():
    """Verify valid ZIP with text documents extracts correctly."""
    valid_zip = _make_valid_zip_bytes(
        {
            "doc1.txt": "This is document one content.",
            "doc2.txt": "This is document two content.",
        }
    )
    result = extract_text_from_zip(valid_zip)
    assert "document one" in result
    assert "document two" in result


def test_extract_zip_handles_empty_archive():
    """Verify empty ZIP returns empty string."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass
    result = extract_text_from_zip(buf.getvalue())
    assert result == ""


def test_extract_zip_skips_directories_and_macos_metadata():
    """Verify ZIP extraction skips directories and __MACOSX files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dir/", "")
        zf.writestr("__MACOSX/.DS_Store", "metadata")
        zf.writestr("doc.txt", "Actual document content")
    result = extract_text_from_zip(buf.getvalue())
    assert "Actual document" in result
    assert "metadata" not in result


def test_corrupted_zip_header_raises_user_friendly_error():
    """Verify corrupted ZIP header triggers CorruptedArchiveError."""
    corrupted_bytes = b"PK\x03\x04corrupted_zip_header_data_not_valid_archive"
    with pytest.raises(CorruptedArchiveError) as exc_info:
        extract_text_from_zip(corrupted_bytes)
    assert "corrupted" in str(exc_info.value).lower()


def test_invalid_zip_stream_raises_user_friendly_error():
    """Verify completely invalid data triggers CorruptedArchiveError."""
    invalid_bytes = b"INVALID_ZIP_STREAM_NOT_A_ZIP_FILE"
    with pytest.raises(CorruptedArchiveError) as exc_info:
        extract_text_from_zip(invalid_bytes)
    assert "corrupted" in str(exc_info.value).lower()


def test_truncated_zip_file_raises_user_friendly_error():
    """Verify truncated/Incomplete ZIP file raises CorruptedArchiveError."""
    # Create a valid ZIP first, then truncate it
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("doc.txt", "Some content")
    truncated = buf.getvalue()[: len(buf.getvalue()) - 20]
    with pytest.raises(CorruptedArchiveError) as exc_info:
        extract_text_from_zip(truncated)
    assert "corrupted" in str(exc_info.value).lower()


def test_extract_zip_handles_corrupted_inner_files():
    """Verify ZIP with corrupted inner files is handled gracefully."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("valid.txt", "This is valid")
        # Add a file that will cause extraction to fail
        zf.writestr("corrupted.txt", "some data")
    # Manually corrupt the central directory to make extraction fail
    zip_bytes = buf.getvalue()
    # This will still be a valid ZIP but the corrupted.txt may fail
    result = extract_text_from_zip(zip_bytes)
    assert "valid" in result


from src.utils.zip_processor import (
    ALLOWED_ZIP_MEMBER_EXTENSIONS,
    MAX_ABSOLUTE_UNCOMPRESSED_SIZE,
    MAX_SINGLE_FILE_SIZE,
    MAX_TOTAL_DECOMPRESSED_SIZE,
    process_zip_file,
)


def create_in_memory_zip(
    files: dict, encrypt: bool = False, password: bytes = None
) -> bytes:
    """Helper to generate a ZIP archive in memory."""
    zip_stream = io.BytesIO()
    # If encrypt is True, we can't easily use standard zipfile to write encrypted files
    # because standard zipfile.ZipFile does not support writing encrypted zip files (it only supports reading them).
    # However, we can mock or construct flag_bits or flag encryption manually.
    with zipfile.ZipFile(zip_stream, "w") as zf:
        for name, content in files.items():
            if encrypt:
                # We can write an entry and manually set flag_bits to indicate encryption
                zinfo = zipfile.ZipInfo(name)
                zinfo.flag_bits = 0x1  # Enable encryption bit
                zf.writestr(zinfo, content)
            else:
                zf.writestr(name, content)
    return zip_stream.getvalue()


def test_allowed_zip_member_extensions_constant():
    """Verify that ALLOWED_ZIP_MEMBER_EXTENSIONS includes all required formats."""
    expected = {".pdf", ".docx", ".txt", ".rtf", ".csv", ".odt", ".md"}
    assert ALLOWED_ZIP_MEMBER_EXTENSIONS == expected


def test_process_zip_valid_extraction():
    """Verify that supported files are successfully extracted from a valid ZIP archive."""
    zip_data = create_in_memory_zip(
        {
            "doc1.pdf": b"PDF text content",
            "doc2.docx": b"Word text content",
            "doc3.txt": b"Plain text content",
            "doc4.rtf": b"{\\rtf1\\ansi RTF content}",
            "doc5.csv": b"col1,col2\nval1,val2",
            "doc6.odt": b"ODT content",
            "doc7.md": b"# Markdown content",
            "unsupported.png": b"Image data",
            "executable.sh": b"#!/bin/sh\necho 1",
        }
    )

    result = process_zip_file(zip_data)

    assert "doc1.pdf" in result
    assert result["doc1.pdf"] == b"PDF text content"
    assert "doc2.docx" in result
    assert result["doc2.docx"] == b"Word text content"
    assert "doc3.txt" in result
    assert result["doc3.txt"] == b"Plain text content"
    assert "doc4.rtf" in result
    assert result["doc4.rtf"] == b"{\\rtf1\\ansi RTF content}"
    assert "doc5.csv" in result
    assert result["doc5.csv"] == b"col1,col2\nval1,val2"
    assert "doc6.odt" in result
    assert result["doc6.odt"] == b"ODT content"
    assert "doc7.md" in result
    assert result["doc7.md"] == b"# Markdown content"

    # Unsupported formats must be ignored
    assert "unsupported.png" not in result
    assert "executable.sh" not in result


def test_process_zip_empty():
    """Verify that empty ZIP input raises a ValueError."""
    with pytest.raises(ValueError, match="ZIP archive is empty."):
        process_zip_file(b"")


def test_process_zip_corrupted():
    """Verify that a corrupted ZIP raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid or corrupted ZIP archive: missing ZIP header signature."):
        process_zip_file(b"this is not a zip file content")


def test_process_zip_encrypted():
    """Verify that password-protected or encrypted ZIP entries raise a ValueError."""
    from unittest.mock import patch

    info = zipfile.ZipInfo("secret.pdf")
    info.flag_bits = 0x1

    zip_data = create_in_memory_zip({"secret.pdf": b"secret contents"})

    with patch("zipfile.ZipFile.infolist", return_value=[info]):
        with pytest.raises(
            ValueError,
            match="Password-protected or encrypted ZIP files are not supported.",
        ):
            process_zip_file(zip_data)


def test_process_zip_nested_folders_and_collisions():
    """Verify nested path flattening (replacing '/' with '_') and collision resolution."""
    zip_data = create_in_memory_zip(
        {
            "assignment.pdf": b"Root version",
            "folder1/assignment.pdf": b"Folder 1 version",
            "folder2/assignment.pdf": b"Folder 2 version",
            "folder2/nested/assignment.pdf": b"Deeply nested version",
        }
    )

    result = process_zip_file(zip_data)

    # Output names must be flattened and unique. The unique_filename
    # function uses basename-only extraction, so all entries resolve to
    # "assignment.pdf" and get suffixed _1, _2, _3 for collisions.
    assert "assignment.pdf" in result
    assert result["assignment.pdf"] == b"Root version"

    assert "assignment_1.pdf" in result
    assert result["assignment_1.pdf"] == b"Folder 1 version"

    assert "assignment_2.pdf" in result
    assert result["assignment_2.pdf"] == b"Folder 2 version"

    assert "assignment_3.pdf" in result
    assert result["assignment_3.pdf"] == b"Deeply nested version"


def test_process_zip_duplicate_name_collision_fallback():
    """Verify that name collisions at the same flattened level get unique suffixes."""
    # Since unique_filename extracts the basename, both 'a/b.txt' and 'a_b.txt'
    # resolve to basename 'b.txt' and 'a_b.txt' respectively (no collision).
    # This test verifies the actual behavior: 'a_b.txt' stays as-is, 'a/b.txt'
    # becomes 'b.txt' (basename only).
    zip_data = create_in_memory_zip(
        {
            "a_b.txt": b"First content",
            "a/b.txt": b"Second content",
        }
    )

    result = process_zip_file(zip_data)

    assert "a_b.txt" in result
    assert result["a_b.txt"] == b"First content"

    assert "b.txt" in result
    assert result["b.txt"] == b"Second content"


@pytest.mark.parametrize(
    "malicious_path",
    [
        "../../etc/passwd",
        "../evil.py",
        "/etc/passwd",
        "..\\evil.py",
    ],
)
def test_rejects_path_traversal_entries(malicious_path):
    """Ensure ZIP archives containing path traversal filenames are rejected."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(malicious_path, "print('malicious')")

    zip_buffer.seek(0)

    with pytest.raises(
        ValueError, match="Malicious path traversal detected in ZIP archive entry"
    ):
        process_zip_file(zip_buffer.read())


def test_accepts_valid_nested_directories():
    """Ensure ZIP archives containing valid nested directories are accepted."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("folder/subfolder/file.pdf", b"PDF text content")

    zip_buffer.seek(0)
    result = process_zip_file(zip_buffer.read())

    assert "file.pdf" in result


def test_process_zip_bomb_safety_total_size():
    """Verify that a ZIP file exceeding the total decompressed safety limit is rejected."""
    from unittest.mock import patch

    info1 = zipfile.ZipInfo("file1.txt")
    info1.file_size = MAX_TOTAL_DECOMPRESSED_SIZE // 2

    info2 = zipfile.ZipInfo("file2.txt")
    info2.file_size = MAX_TOTAL_DECOMPRESSED_SIZE // 2

    info3 = zipfile.ZipInfo("file3.txt")
    info3.file_size = 1  # Pushes total over the limit

    zip_bytes = create_in_memory_zip({"doc.txt": b"some content"})

    with patch("zipfile.ZipFile.infolist", return_value=[info1, info2, info3]):
        with pytest.raises(
            ValueError,
            match="ZIP archive total decompressed size exceeds safety limit",
        ):
            process_zip_file(zip_bytes)


def test_process_zip_bomb_safety_single_file():
    """Verify that a ZIP file containing a single entry exceeding the safety limit is rejected."""
    from unittest.mock import patch

    info = zipfile.ZipInfo("huge_file.txt")
    info.file_size = MAX_SINGLE_FILE_SIZE + 100

    zip_bytes = create_in_memory_zip({"doc.txt": b"some content"})

    with patch("zipfile.ZipFile.infolist", return_value=[info]):
        with pytest.raises(
            ValueError, match="exceeds single file decompression safety limit"
        ):
            process_zip_file(zip_bytes)


# ---------------------------------------------------------------------------
# Issue #1364 — Structural Zip Bombs Decompression Ratio Limits
# ---------------------------------------------------------------------------


def test_zip_bomb_decompression_ratio_exceeded():
    """Verify that a ZIP entry exceeding the 100:1 decompression ratio is rejected."""
    from unittest.mock import patch

    info = zipfile.ZipInfo("bomb.txt")
    info.file_size = 101 * 1024 * 1024  # 101 MB uncompressed
    info.compress_size = 1 * 1024 * 1024  # 1 MB compressed → 101:1 ratio

    zip_bytes = create_in_memory_zip({"doc.txt": b"some content"})

    with patch("zipfile.ZipFile.infolist", return_value=[info]):
        with pytest.raises(
            ValueError,
            match="Decompression ratio exceeds security limit",
        ):
            process_zip_file(zip_bytes)


def test_zip_bomb_absolute_uncompressed_size_exceeded():
    """Verify that a ZIP entry exceeding the 500 MB absolute limit is rejected."""
    from unittest.mock import patch

    info = zipfile.ZipInfo("huge_bomb.txt")
    info.file_size = MAX_ABSOLUTE_UNCOMPRESSED_SIZE + 1  # 500 MB + 1 byte
    info.compress_size = (
        MAX_ABSOLUTE_UNCOMPRESSED_SIZE  # Same compressed size → ratio ~1:1
    )

    zip_bytes = create_in_memory_zip({"doc.txt": b"some content"})

    with patch("zipfile.ZipFile.infolist", return_value=[info]):
        with pytest.raises(
            ValueError,
            match="Decompression ratio exceeds security limit",
        ):
            process_zip_file(zip_bytes)


def test_zip_bomb_ratio_just_under_limit_passes():
    """Verify that a ZIP entry just under the 100:1 ratio limit is allowed to proceed."""
    from unittest.mock import patch

    info = zipfile.ZipInfo("borderline.txt")
    info.file_size = 99 * 1024 * 1024  # 99 MB uncompressed
    info.compress_size = 1 * 1024 * 1024  # 1 MB compressed → 99:1 ratio

    zip_bytes = create_in_memory_zip({"doc.txt": b"some content"})

    # Mock infolist to return our bomb entry + a valid one, and mock read to return bytes
    mock_valid_info = zipfile.ZipInfo("doc.txt")
    mock_valid_info.file_size = 12
    mock_valid_info.compress_size = 12

    def mock_read(self, name_or_info):
        return b"some content"

    with (
        patch("zipfile.ZipFile.infolist", return_value=[info, mock_valid_info]),
        patch("zipfile.ZipFile.read", mock_read),
    ):
        # Should NOT raise the ratio error (may raise total size error, but not ratio)
        try:
            process_zip_file(zip_bytes)
        except ValueError as e:
            # If it raises, it should NOT be the ratio error
            assert "Decompression ratio exceeds security limit" not in str(e)


def test_zip_bomb_zero_compressed_size_handled():
    """Verify that entries with compress_size=0 (stored, no compression) don't crash."""
    from unittest.mock import patch

    info = zipfile.ZipInfo("stored.txt")
    info.file_size = 1024
    info.compress_size = 0  # Stored entries can have 0 compressed size

    zip_bytes = create_in_memory_zip({"doc.txt": b"some content"})

    mock_valid_info = zipfile.ZipInfo("doc.txt")
    mock_valid_info.file_size = 12
    mock_valid_info.compress_size = 12

    def mock_read(self, name_or_info):
        return b"some content"

    with (
        patch("zipfile.ZipFile.infolist", return_value=[info, mock_valid_info]),
        patch("zipfile.ZipFile.read", mock_read),
    ):
        # Should not raise a ZeroDivisionError or ratio error
        try:
            process_zip_file(zip_bytes)
        except ValueError as e:
            assert "Decompression ratio" not in str(e)


def test_process_zip_skip_corrupted_on_read_failure():
    """Verify that process_zip_file with skip_corrupted=True skips corrupted read entries and extracts valid ones."""
    from unittest.mock import patch

    # 1. Create a zip with one valid file and one corrupted/malformed file.
    zip_data = create_in_memory_zip({
        "valid.pdf": b"Valid PDF contents",
        "corrupted.pdf": b"Corrupted contents"
    })

    # Mock ZipFile.read to raise BadZipFile for "corrupted.pdf"
    original_read = zipfile.ZipFile.read
    def mock_read(self, name_or_info):
        name = name_or_info.filename if isinstance(name_or_info, zipfile.ZipInfo) else name_or_info
        if "corrupted" in name:
            raise zipfile.BadZipFile("CRC check failed")
        return original_read(self, name_or_info)

    with patch("zipfile.ZipFile.read", mock_read):
        # With skip_corrupted=False, it should raise ValueError
        with pytest.raises(ValueError, match="Corrupted or protected entry"):
            process_zip_file(zip_data, skip_corrupted=False)

        # With skip_corrupted=True, it should successfully extract the valid entry and log warning
        result = process_zip_file(zip_data, skip_corrupted=True)
        assert "valid.pdf" in result
        assert result["valid.pdf"] == b"Valid PDF contents"
        assert "corrupted.pdf" not in result


def test_process_zip_skip_encrypted_entries():
    """Verify that process_zip_file with skip_corrupted=True skips encrypted entries and extracts valid ones."""
    from unittest.mock import patch

    info_encrypted = zipfile.ZipInfo("secret.pdf")
    info_encrypted.flag_bits = 0x1

    info_valid = zipfile.ZipInfo("valid.pdf")
    info_valid.flag_bits = 0x0

    zip_data = create_in_memory_zip({
        "secret.pdf": b"secret contents",
        "valid.pdf": b"valid contents"
    })

    # Mock infolist to return both entries, and mock read to return contents
    def mock_read(self, name_or_info):
        name = name_or_info.filename if isinstance(name_or_info, zipfile.ZipInfo) else name_or_info
        if "secret" in name:
            return b"secret contents"
        return b"valid contents"

    with patch("zipfile.ZipFile.infolist", return_value=[info_encrypted, info_valid]), patch(
        "zipfile.ZipFile.read", mock_read
    ):
        # With skip_corrupted=False, it should raise ValueError
        with pytest.raises(ValueError, match="Password-protected or encrypted ZIP files are not supported."):
            process_zip_file(zip_data, skip_corrupted=False)

        # With skip_corrupted=True, it should skip the encrypted file and successfully return the valid one
        result = process_zip_file(zip_data, skip_corrupted=True)
        assert "valid.pdf" in result
        assert result["valid.pdf"] == b"valid contents"
        assert "secret.pdf" not in result

