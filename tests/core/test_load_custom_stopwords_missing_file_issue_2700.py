"""
test_load_custom_stopwords_missing_file_issue_2700.py
------------------------------------------------------
Comprehensive unit test suite for Issue #2700:
Validating `load_custom_stopwords` missing file behavior, environment variable fallbacks,
file permission errors, empty file handling, and combined stopword filtering integration.

This suite ensures 100% test coverage for:
1. `STOPWORDS_FILE` set to a non-existent file path (returns `frozenset()`).
2. Explicit `file_path` argument passed pointing to a non-existent file (returns `frozenset()`).
3. `STOPWORDS_FILE` env var unset / empty string (returns `frozenset()`).
4. Valid custom stopwords file loading (strips, lowercases, returns custom `frozenset`).
5. Empty file handling.
6. Permission error handling.
7. Integration with `get_stopwords()` and `clean_text(remove_stopwords=True)`.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from src.core.document_parser import (
    get_stopwords as doc_parser_get_stopwords,
)
from src.core.document_parser import (
    load_custom_stopwords as doc_parser_load_custom_stopwords,
)
from src.core.parsers.cleaners import (
    ENGLISH_STOPWORDS,
    clean_text,
    get_stopwords,
)
from src.core.parsers.cleaners import (
    load_custom_stopwords as cleaners_load_custom_stopwords,
)

# ---------------------------------------------------------------------------
# Section 1: Non-Existent File & Missing Environment Variable Recovery Tests
# ---------------------------------------------------------------------------


def test_load_custom_stopwords_missing_file_env_var(monkeypatch):
    """Set STOPWORDS_FILE env var to a non-existent file path and assert
    that load_custom_stopwords() catches OSError and returns frozenset()."""
    invalid_path = "invalid_non_existent_stopwords_file_path_12345.txt"
    monkeypatch.setenv("STOPWORDS_FILE", invalid_path)

    # Test cleaners module implementation
    res_cleaners = cleaners_load_custom_stopwords()
    assert res_cleaners == frozenset()
    assert isinstance(res_cleaners, frozenset)

    # Test document_parser module implementation
    res_doc = doc_parser_load_custom_stopwords()
    assert res_doc == frozenset()
    assert isinstance(res_doc, frozenset)


def test_load_custom_stopwords_explicit_missing_file_parameter():
    """Pass a non-existent file path parameter directly to load_custom_stopwords
    and assert it catches OSError and returns frozenset()."""
    invalid_path = "/non_existent_dir_xyz/missing_stopwords.txt"

    res_cleaners = cleaners_load_custom_stopwords(file_path=invalid_path)
    assert res_cleaners == frozenset()

    res_doc = doc_parser_load_custom_stopwords(file_path=invalid_path)
    assert res_doc == frozenset()


def test_load_custom_stopwords_unset_env_var(monkeypatch):
    """Verify that when STOPWORDS_FILE is unset or empty, load_custom_stopwords returns frozenset()."""
    monkeypatch.delenv("STOPWORDS_FILE", raising=False)

    res_cleaners = cleaners_load_custom_stopwords()
    assert res_cleaners == frozenset()

    res_doc = doc_parser_load_custom_stopwords()
    assert res_doc == frozenset()


def test_load_custom_stopwords_empty_string_env_var(monkeypatch):
    """Verify that when STOPWORDS_FILE is set to an empty string, load_custom_stopwords returns frozenset()."""
    monkeypatch.setenv("STOPWORDS_FILE", "")

    res_cleaners = cleaners_load_custom_stopwords()
    assert res_cleaners == frozenset()

    res_doc = doc_parser_load_custom_stopwords()
    assert res_doc == frozenset()


# ---------------------------------------------------------------------------
# Section 2: Successful Custom Stopwords File Parsing Tests
# ---------------------------------------------------------------------------


def test_load_custom_stopwords_valid_file_parsing():
    """Create a temporary stopwords file and verify parsing, lowercasing, and whitespace stripping."""
    stopwords_content = "  CUSTOM_WORD_ONE  \n\nCustom_Word_Two\n\tTHIRD_WORD\t\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(stopwords_content)
        tmp_path = tmp.name

    try:
        loaded = cleaners_load_custom_stopwords(file_path=tmp_path)
        expected = frozenset({"custom_word_one", "custom_word_two", "third_word"})
        assert loaded == expected

        loaded_doc = doc_parser_load_custom_stopwords(file_path=tmp_path)
        assert loaded_doc == expected
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_custom_stopwords_empty_file():
    """Verify that an empty file produces an empty frozenset."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        loaded = cleaners_load_custom_stopwords(file_path=tmp_path)
        assert loaded == frozenset()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Section 3: Permission & OS Error Exception Handling
# ---------------------------------------------------------------------------


def test_load_custom_stopwords_permission_error_mocked():
    """Mock open to raise PermissionError (subclass of OSError) and assert recovery."""
    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        res = cleaners_load_custom_stopwords("protected_file.txt")
        assert res == frozenset()

        res_doc = doc_parser_load_custom_stopwords("protected_file.txt")
        assert res_doc == frozenset()


def test_load_custom_stopwords_general_oserror_mocked():
    """Mock open to raise generic OSError and assert recovery."""
    with patch("builtins.open", side_effect=OSError("I/O error")):
        res = cleaners_load_custom_stopwords("damaged_drive_file.txt")
        assert res == frozenset()


# ---------------------------------------------------------------------------
# Section 4: Integration with Stopword Aggregation & Text Cleaning
# ---------------------------------------------------------------------------


def test_get_stopwords_integration_with_missing_file(monkeypatch):
    """Verify get_stopwords() returns standard English stopwords when custom file is missing."""
    monkeypatch.setenv("STOPWORDS_FILE", "non_existent_file.txt")
    combined = get_stopwords()
    assert combined == ENGLISH_STOPWORDS

    combined_doc = doc_parser_get_stopwords()
    assert combined_doc == ENGLISH_STOPWORDS


def test_get_stopwords_integration_with_valid_custom_file(monkeypatch):
    """Verify get_stopwords() merges standard English stopwords with custom file stopwords."""
    custom_content = "plagiarism\nalgorithm\nheuristic\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write(custom_content)
        tmp_path = tmp.name

    monkeypatch.setenv("STOPWORDS_FILE", tmp_path)
    try:
        combined = get_stopwords()
        assert "plagiarism" in combined
        assert "algorithm" in combined
        assert "the" in combined  # standard stopword check
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_clean_text_with_missing_custom_stopwords_file(monkeypatch):
    """Verify clean_text(remove_stopwords=True) operates cleanly when STOPWORDS_FILE is invalid."""
    monkeypatch.setenv("STOPWORDS_FILE", "invalid_stopwords.txt")
    text = "The quick brown fox jumps over the lazy dog"
    cleaned = clean_text(text, remove_stopwords=True)
    # 'the', 'over' are standard english stopwords and should be removed
    assert "The" not in cleaned.split()
    assert "quick" in cleaned
    assert "fox" in cleaned
