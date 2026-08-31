"""
Comprehensive Unit Tests for sanitize_filename_list helper
Issue: #3728
Tests filename sanitation, edge cases, and security against path traversal.
"""

import os
import pytest
import re


# ==============================================================================
# SECTION 1: Defining the Helper Function (Under Test)
# ==============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a single filename by removing dangerous characters and path traversal attempts.
    """
    if not isinstance(filename, str):
        return ""
    
    # Remove path traversal attempts
    filename = filename.replace("../", "").replace("..\\", "")
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    
    # Remove leading/trailing whitespace and dots (prevent hidden files)
    filename = filename.strip().strip('.')
    
    return filename


def sanitize_filename_list(filenames: list) -> list:
    """
    Sanitizes a list of filenames.
    """
    if not isinstance(filenames, list):
        return []
    return [sanitize_filename(f) for f in filenames]


# ==============================================================================
# SECTION 2: Testing Basic Sanitization
# ==============================================================================

class TestSanitizeFilename:
    def test_clean_filename_preserved(self):
        """A clean filename should be preserved."""
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_filename_with_spaces(self):
        """Spaces should be preserved (not stripped from middle)."""
        assert sanitize_filename("my report.pdf") == "my report.pdf"

    def test_filename_with_hyphens(self):
        """Hyphens should be preserved."""
        assert sanitize_filename("my-report.pdf") == "my-report.pdf"

    def test_filename_with_underscores(self):
        """Underscores should be preserved."""
        assert sanitize_filename("my_report.pdf") == "my_report.pdf"

    def test_filename_with_numbers(self):
        """Numbers should be preserved."""
        assert sanitize_filename("file123.txt") == "file123.txt"

    def test_filename_with_uppercase(self):
        """Uppercase letters should be preserved."""
        assert sanitize_filename("REPORT.PDF") == "REPORT.PDF"


# ==============================================================================
# SECTION 3: Testing Path Traversal Prevention
# ==============================================================================

class TestPathTraversal:
    def test_removes_parent_directory(self):
        """Should remove ../ path traversal."""
        assert sanitize_filename("../malicious.txt") == "malicious.txt"

    def test_removes_windows_parent_directory(self):
        """Should remove ..\\ path traversal."""
        assert sanitize_filename("..\\malicious.txt") == "malicious.txt"

    def test_removes_nested_parent_directory(self):
        """Should remove nested ../ path traversal."""
        assert sanitize_filename("../../etc/passwd") == "etcpasswd"

    def test_removes_absolute_path(self):
        """Should remove absolute path starts."""
        assert sanitize_filename("/etc/passwd") == "etcpasswd"

    def test_removes_windows_absolute_path(self):
        """Should remove C:\\ drive paths."""
        assert sanitize_filename("C:\\Users\\Admin") == "CUsersAdmin"


# ==============================================================================
# SECTION 4: Testing Dangerous Character Removal
# ==============================================================================

class TestDangerousCharacters:
    def test_removes_angle_brackets(self):
        """Should remove < and >."""
        assert sanitize_filename("file<1>.txt") == "file1.txt"

    def test_removes_colon(self):
        """Should remove :."""
        assert sanitize_filename("file:1.txt") == "file1.txt"

    def test_removes_quotes(self):
        """Should remove double quotes."""
        assert sanitize_filename('file"1".txt') == "file1.txt"

    def test_removes_single_quotes(self):
        """Should remove single quotes."""
        assert sanitize_filename("file'1'.txt") == "file1.txt"

    def test_removes_pipes(self):
        """Should remove |."""
        assert sanitize_filename("file|1.txt") == "file1.txt"

    def test_removes_question_marks(self):
        """Should remove ?."""
        assert sanitize_filename("file?1.txt") == "file1.txt"

    def test_removes_asterisks(self):
        """Should remove *."""
        assert sanitize_filename("file*1.txt") == "file1.txt"

    def test_removes_control_characters(self):
        """Should remove control characters like newlines."""
        assert sanitize_filename("file\n1.txt") == "file1.txt"


# ==============================================================================
# SECTION 5: Testing Edge Cases and Inputs
# ==============================================================================

class TestEdgeCases:
    def test_empty_string(self):
        """Should return empty string for empty input."""
        assert sanitize_filename("") == ""

    def test_none_input(self):
        """Should return empty string for None input."""
        assert sanitize_filename(None) == ""

    def test_integer_input(self):
        """Should return empty string for integer input."""
        assert sanitize_filename(123) == ""

    def test_string_of_dots(self):
        """Should strip leading/trailing dots (hidden files)."""
        assert sanitize_filename("...file.txt...") == "file.txt"

    def test_string_of_spaces(self):
        """Should strip leading/trailing spaces."""
        assert sanitize_filename("   file.txt   ") == "file.txt"

    def test_whitespace_only_string(self):
        """Should return empty string for whitespace only."""
        assert sanitize_filename("   ") == ""


# ==============================================================================
# SECTION 6: Testing the List Helper Function
# ==============================================================================

class TestSanitizeFilenameList:
    def test_clean_list(self):
        """Should sanitize a list of filenames."""
        result = sanitize_filename_list(["file1.txt", "file2.txt"])
        assert result == ["file1.txt", "file2.txt"]

    def test_mixed_list(self):
        """Should sanitize mixed valid and invalid filenames."""
        result = sanitize_filename_list(["../evil.txt", "good.txt"])
        assert result == ["evil.txt", "good.txt"]

    def test_empty_list(self):
        """Should return empty list for empty list input."""
        assert sanitize_filename_list([]) == []

    def test_none_list(self):
        """Should return empty list for None input."""
        assert sanitize_filename_list(None) == []

    def test_list_with_none_values(self):
        """Should handle None values inside the list."""
        result = sanitize_filename_list([None, "good.txt"])
        assert result == ["", "good.txt"]

    def test_list_with_non_strings(self):
        """Should handle non-string values inside the list."""
        result = sanitize_filename_list([123, "good.txt"])
        assert result == ["", "good.txt"]

    def test_duplicate_filenames_preserved(self):
        """Should preserve duplicate filenames in the list."""
        result = sanitize_filename_list(["file.txt", "file.txt"])
        assert len(result) == 2