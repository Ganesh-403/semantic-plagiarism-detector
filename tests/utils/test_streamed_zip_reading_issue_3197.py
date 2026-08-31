"""
test_streamed_zip_reading_issue_3197.py
----------------------------------------
Unit test suite for Issue #3197:
Validates that iter_zip_files provides a memory-optimized generator yielding (filename, file_data)
tuples sequentially to minimize peak RAM usage.
"""

import io
import types
import zipfile
import pytest

from src.utils.zip_processor import iter_zip_files, process_zip_file


def _create_sample_zip(files: dict) -> bytes:
    """Helper to create an in-memory ZIP archive from dict mapping filename to content."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in files.items():
            zf.writestr(fname, content)
    return buf.getvalue()


def test_iter_zip_files_is_generator_yielding_tuples():
    """Verify iter_zip_files returns a generator yielding (filename, file_data) tuples one by one."""
    sample_data = {
        "doc1.txt": b"Content of document 1",
        "doc2.md": b"# Header of document 2",
        "nested/doc3.pdf": b"%PDF-1.4 dummy pdf bytes",
    }
    zip_bytes = _create_sample_zip(sample_data)

    gen = iter_zip_files(zip_bytes)
    assert isinstance(gen, types.GeneratorType)

    extracted_entries = list(gen)
    assert len(extracted_entries) == 3

    filenames = [item[0] for item in extracted_entries]
    contents = [item[1] for item in extracted_entries]

    assert "doc1.txt" in filenames
    assert "doc2.md" in filenames
    assert "doc3.pdf" in filenames
    assert b"Content of document 1" in contents


def test_process_zip_file_matches_iter_zip_files_output():
    """Verify process_zip_file returns a dictionary consistent with iter_zip_files."""
    sample_data = {
        "a.txt": b"First document",
        "b.txt": b"Second document",
    }
    zip_bytes = _create_sample_zip(sample_data)

    gen_dict = dict(iter_zip_files(zip_bytes))
    proc_dict = process_zip_file(zip_bytes)

    assert gen_dict == proc_dict
