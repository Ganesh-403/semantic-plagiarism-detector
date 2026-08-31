from src.utils.file_parser import truncate_filename


def test_truncate_filename_short():
    """Verify that filenames shorter than max_len are returned unchanged."""
    assert truncate_filename("short.pdf", max_len=15) == "short.pdf"
    assert truncate_filename("essay.txt", max_len=35) == "essay.txt"


def test_truncate_filename_exact():
    """Verify that filenames exactly equal to max_len are returned unchanged."""
    assert truncate_filename("exact_length.pdf", max_len=16) == "exact_length.pdf"


def test_truncate_filename_long():
    """Verify that long filenames are truncated with ellipsis, keeping length <= max_len."""
    name = "this_is_a_very_long_filename_that_exceeds_the_default_limit.pdf"
    truncated = truncate_filename(name, max_len=35)
    assert len(truncated) == 35
    assert truncated.endswith("...")
    assert truncated == "this_is_a_very_long_filename_tha..."


def test_truncate_filename_custom_max_len():
    """Verify that custom max_len bounds are respected."""
    name = "my_document.docx"
    assert truncate_filename(name, max_len=10) == "my_docu..."
    assert len(truncate_filename(name, max_len=10)) == 10


# ── Edge Case Tests (Issue #2252) ────────────────────────────────────────────


def test_truncate_filename_empty_string():
    """Empty string should be returned unchanged."""
    assert truncate_filename("", max_len=35) == ""


def test_truncate_filename_single_character():
    """Single character is well below any reasonable max_len."""
    assert truncate_filename("a", max_len=35) == "a"


def test_truncate_filename_one_over_limit():
    """Name one character over limit should be truncated with ellipsis."""
    name = "abcdefgh"  # length 8
    result = truncate_filename(name, max_len=7)
    assert result == "abcd..."
    assert len(result) == 7


def test_truncate_filename_max_len_equals_three():
    """max_len=3 forces entire stem to be replaced by '...'."""
    long_name = "averylongfilename.pdf"
    result = truncate_filename(long_name, max_len=3)
    assert result == "..."
    assert len(result) == 3


def test_truncate_filename_default_max_len():
    """Default max_len=35 is used when not specified."""
    name = "x" * 36
    result = truncate_filename(name)
    assert len(result) == 35
    assert result.endswith("...")


def test_truncate_filename_unicode_characters():
    """Unicode characters are counted by code point, not byte length."""
    name = "日本語ファイル名前長いです_document_report.txt"
    result = truncate_filename(name, max_len=10)
    assert len(result) == 10
    assert result.endswith("...")


def test_truncate_filename_special_characters():
    """Filenames with special characters truncate correctly."""
    name = "report_@_2024!final_version_submitted.pdf"
    result = truncate_filename(name, max_len=20)
    assert len(result) == 20
    assert result.endswith("...")


def test_truncate_filename_large_max_len():
    """Very large max_len never truncates a normal filename."""
    name = "normal_file.pdf"
    assert truncate_filename(name, max_len=1000) == "normal_file.pdf"


def test_truncate_filename_no_extension():
    """Filenames without extensions truncate correctly."""
    name = "this_is_a_long_filename_without_any_extension"
    result = truncate_filename(name, max_len=15)
    assert len(result) == 15
    assert result.endswith("...")


def test_truncate_filename_repeated_chars():
    """Repeated character filenames truncate deterministically."""
    name = "a" * 100
    result = truncate_filename(name, max_len=10)
    assert result == "aaaaaaa..."
    assert len(result) == 10


def test_truncate_filename_whitespace_string():
    """Whitespace-only name within limit is returned unchanged."""
    assert truncate_filename("   ", max_len=35) == "   "


def test_truncate_filename_preserves_ellipsis_suffix():
    """Truncated result always ends with exactly three dots."""
    name = "some_very_long_document_filename_here.docx"
    result = truncate_filename(name, max_len=20)
    assert result[-3:] == "..."
    assert result.count("...") == 1
