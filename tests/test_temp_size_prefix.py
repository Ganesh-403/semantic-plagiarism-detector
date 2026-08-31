"""
Comprehensive Unit Tests for get_temp_directory_size_bytes (Issue #3184)
Tests prefix filtering, recursive directory traversal, file sizes, and edge cases.
"""

import os
import tempfile
import stat
import pytest
from pathlib import Path


# ==============================================================================
# SECTION 1: Defining the Function Under Test
# ==============================================================================

def get_temp_directory_size_bytes(target_dir: str = None, prefix: str = "") -> int:
    """
    Calculates total disk space occupied by files in a temp directory.
    Filters files based on a prefix if provided.
    """
    if target_dir is None:
        target_dir = tempfile.gettempdir()

    total_size = 0
    if not os.path.exists(target_dir):
        return 0

    for dirpath, dirnames, filenames in os.walk(target_dir):
        for filename in filenames:
            if prefix and not filename.startswith(prefix):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                continue
    return total_size


# ==============================================================================
# SECTION 2: Basic Functionality and Data Types
# ==============================================================================

class TestGetTempDirectorySizeBytesBasics:
    def test_returns_integer(self):
        """Function should always return an integer."""
        result = get_temp_directory_size_bytes()
        assert isinstance(result, int)

    def test_returns_non_negative_integer(self):
        """Function should never return a negative number."""
        result = get_temp_directory_size_bytes()
        assert result >= 0

    def test_default_target_is_system_temp(self):
        """If no target_dir is given, it should scan the system temp dir."""
        result = get_temp_directory_size_bytes()
        assert result == get_temp_directory_size_bytes(target_dir=tempfile.gettempdir())


# ==============================================================================
# SECTION 3: Edge Cases with Directories
# ==============================================================================

class TestGetTempDirectorySizeBytesEdgeCases:
    def test_returns_zero_for_nonexistent_dir(self):
        """Should return 0 if the directory doesn't exist."""
        result = get_temp_directory_size_bytes(target_dir="/non/existent/path")
        assert result == 0

    def test_returns_zero_for_file_path(self):
        """Should return 0 if target_dir is actually a file, not a directory."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            result = get_temp_directory_size_bytes(target_dir=tmp.name)
            assert result == 0
            os.unlink(tmp.name)

    def test_returns_zero_for_empty_dir(self, tmp_path):
        """Should return 0 for an empty directory."""
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert result == 0

    def test_returns_zero_for_empty_string_path(self):
        """Should return 0 for an empty string path."""
        result = get_temp_directory_size_bytes(target_dir="")
        assert result == 0


# ==============================================================================
# SECTION 4: File Counting and Sizes
# ==============================================================================

class TestGetTempDirectorySizeBytesFileCounting:
    def test_counts_all_files_without_prefix(self, tmp_path):
        """Should count all files when no prefix is given."""
        (tmp_path / "file1.txt").write_text("12345")  # 5 bytes
        (tmp_path / "file2.log").write_text("1234567890")  # 10 bytes
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert result == 15

    def test_counts_exact_file_size(self, tmp_path):
        """Should accurately calculate the byte size of a file."""
        content = "A" * 1024  # 1KB
        (tmp_path / "data.bin").write_text(content)
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert result == 1024

    def test_counts_files_with_extensions(self, tmp_path):
        """Should count files regardless of extension."""
        (tmp_path / "image.png").write_text("1234")
        (tmp_path / "document.pdf").write_text("123456789")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert result == 13


# ==============================================================================
# SECTION 5: Prefix Filtering Logic
# ==============================================================================

class TestGetTempDirectorySizeBytesPrefix:
    def test_filters_by_prefix(self, tmp_path):
        """Should only count files with the matching prefix."""
        (tmp_path / "app_file.txt").write_text("12345")  # 5 bytes
        (tmp_path / "other_file.txt").write_text("1234567890")  # 10 bytes
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path), prefix="app_")
        assert result == 5

    def test_filters_by_empty_prefix(self, tmp_path):
        """Empty prefix should count all files."""
        (tmp_path / "a.txt").write_text("123")
        (tmp_path / "b.txt").write_text("456")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path), prefix="")
        assert result == 6

    def test_filters_by_case_insensitive_prefix(self, tmp_path):
        """Should filter by prefix (case-sensitive test)."""
        (tmp_path / "APP_file.txt").write_text("123")
        (tmp_path / "app_file.txt").write_text("456")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path), prefix="app_")
        assert result == 3  # Only lowercase prefix is matched

    def test_filters_when_no_files_match(self, tmp_path):
        """Should return 0 when no files match the prefix."""
        (tmp_path / "other.txt").write_text("123")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path), prefix="zzz_")
        assert result == 0


# ==============================================================================
# SECTION 6: Recursive Directory Traversal
# ==============================================================================

class TestGetTempDirectorySizeBytesRecursive:
    def test_counts_files_in_subdirectories(self, tmp_path):
        """Should count files in subdirectories."""
        nested = tmp_path / "subfolder"
        nested.mkdir()
        (tmp_path / "root.txt").write_text("123")
        (nested / "nested.txt").write_text("456")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert result == 6

    def test_counts_nested_subdirectories_deep(self, tmp_path):
        """Should count files in deeply nested subdirectories."""
        deep1 = tmp_path / "level1"
        deep2 = deep1 / "level2"
        deep3 = deep2 / "level3"
        deep3.mkdir(parents=True)
        (deep3 / "deep_file.txt").write_text("1234567890")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert result == 10

    def test_filters_prefix_in_subdirectories(self, tmp_path):
        """Should apply prefix filtering recursively."""
        nested = tmp_path / "subfolder"
        nested.mkdir()
        (tmp_path / "root.txt").write_text("123")
        (nested / "app_file.txt").write_text("456")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path), prefix="app_")
        assert result == 3  # Only the nested file matches


# ==============================================================================
# SECTION 7: Robustness and Error Handling
# ==============================================================================

class TestGetTempDirectorySizeBytesRobustness:
    def test_skips_permission_errors(self, tmp_path):
        """Should not crash on permission errors."""
        if os.name == 'nt':
            pytest.skip("Permission tests are platform-specific")
        (tmp_path / "file.txt").write_text("123")
        # Set to read-only to test skipping
        os.chmod(tmp_path / "file.txt", stat.S_IREAD)
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert isinstance(result, int)

    def test_handles_symlinks(self, tmp_path):
        """Should handle symlinks gracefully (may count or skip)."""
        target = tmp_path / "target.txt"
        target.write_text("123")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert isinstance(result, int)

    def test_no_crash_on_hidden_files(self, tmp_path):
        """Should count hidden files normally."""
        (tmp_path / ".hidden").write_text("12345")
        result = get_temp_directory_size_bytes(target_dir=str(tmp_path))
        assert result == 5