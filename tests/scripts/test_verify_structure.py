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
tests/scripts/test_verify_structure.py
--------------------------------------
Unit tests for the project structure verification script.

Validates structure checking, output formatting (text/json),
argument parsing, and exit codes per Issue #2021.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import verify_structure


class TestVerifyStructure:
    """Test suite for verify_structure() core logic."""

    def test_finds_existing_directories(self, tmp_path):
        """Verify existing directories are detected."""
        # Create required directories
        for dir_path in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)

        # Create required files
        for file_path in verify_structure.REQUIRED_FILES:
            (tmp_path / file_path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / file_path).touch()

        results = verify_structure.verify_structure(tmp_path)

        assert len(results["missing"]) == 0
        expected_count = len(verify_structure.REQUIRED_DIRECTORIES) + len(
            verify_structure.REQUIRED_FILES
        )
        assert len(results["found"]) == expected_count

    def test_detects_missing_directories(self, tmp_path):
        """Verify missing directories are detected."""
        # Don't create any directories
        results = verify_structure.verify_structure(tmp_path)

        assert len(results["missing"]) > 0
        assert any("dir:" in item for item in results["missing"])

    def test_detects_missing_files(self, tmp_path):
        """Verify missing files are detected."""
        # Create directories but not files
        for dir_path in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)

        results = verify_structure.verify_structure(tmp_path)

        assert len(results["missing"]) > 0
        assert any("file:" in item for item in results["missing"])

    def test_distinguishes_files_from_directories(self, tmp_path):
        """Verify files and directories are distinguished correctly."""
        # Create a file where a directory is expected
        (tmp_path / "src").touch()  # Should be a directory

        results = verify_structure.verify_structure(tmp_path)

        # "src" should be in missing (it's a file, not a directory)
        assert "dir:src" in results["missing"]

    def test_results_format(self, tmp_path):
        """Verify results dictionary has correct structure."""
        results = verify_structure.verify_structure(tmp_path)

        assert "found" in results
        assert "missing" in results
        assert isinstance(results["found"], list)
        assert isinstance(results["missing"], list)


class TestTextFormat:
    """Test suite for text output formatting."""

    def test_text_format_passed(self):
        """Verify text format for passing verification."""
        results = {
            "found": ["dir:src", "file:README.md"],
            "missing": [],
        }

        output = verify_structure.format_text_output(results)

        assert "PASSED" in output
        assert "2 required items found" in output
        assert "✓ dir:src" in output
        assert "✓ file:README.md" in output

    def test_text_format_failed(self):
        """Verify text format for failed verification."""
        results = {
            "found": ["dir:src"],
            "missing": ["dir:tests", "file:README.md"],
        }

        output = verify_structure.format_text_output(results)

        assert "FAILED" in output
        assert "2 items missing" in output
        assert "dir:tests" in output
        assert "file:README.md" in output

    def test_text_format_sorted_output(self):
        """Verify text format sorts items alphabetically."""
        results = {
            "found": ["file:z.txt", "file:a.txt", "dir:src"],
            "missing": [],
        }

        output = verify_structure.format_text_output(results)

        # Check that items appear in sorted order
        lines = output.split("\n")
        found_lines = [l for l in lines if "✓" in l]  # noqa: E741

        # Extract item names
        items = [l.strip().replace("✓ ", "") for l in found_lines]  # noqa: E741

        # Verify sorted
        assert items == sorted(items)


class TestJsonFormat:
    """Test suite for JSON output formatting (Issue #2021)."""

    def test_json_format_passed(self):
        """Verify JSON format for passing verification."""
        results = {
            "found": ["dir:src", "file:README.md"],
            "missing": [],
        }

        output = verify_structure.format_json_output(results)
        parsed = json.loads(output)

        assert parsed["passed"] is True
        assert parsed["missing"] == []
        assert len(parsed["found"]) == 2

    def test_json_format_failed(self):
        """Verify JSON format for failed verification."""
        results = {
            "found": ["dir:src"],
            "missing": ["dir:tests", "file:README.md"],
        }

        output = verify_structure.format_json_output(results)
        parsed = json.loads(output)

        assert parsed["passed"] is False
        assert len(parsed["missing"]) == 2
        assert "dir:tests" in parsed["missing"]
        assert "file:README.md" in parsed["missing"]

    def test_json_format_valid_json(self):
        """Verify output is valid JSON."""
        results = {
            "found": ["dir:src"],
            "missing": ["dir:tests"],
        }

        output = verify_structure.format_json_output(results)

        # Should not raise
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_format_structure(self):
        """Verify JSON output has correct structure."""
        results = {
            "found": ["dir:src", "file:README.md"],
            "missing": ["dir:tests"],
        }

        output = verify_structure.format_json_output(results)
        parsed = json.loads(output)

        # Verify required keys
        assert "passed" in parsed
        assert "missing" in parsed
        assert "found" in parsed

        # Verify types
        assert isinstance(parsed["passed"], bool)
        assert isinstance(parsed["missing"], list)
        assert isinstance(parsed["found"], list)


class TestArgumentParsing:
    """Test suite for command line argument parsing."""

    def test_parse_default_arguments(self):
        """Verify default argument values."""
        with patch("sys.argv", ["verify_structure.py"]):
            args = verify_structure.parse_arguments()

        assert args.root_dir == verify_structure.ROOT_DIR
        assert args.format == "text"

    def test_parse_json_format(self):
        """Verify --format json is parsed correctly."""
        with patch("sys.argv", ["verify_structure.py", "--format", "json"]):
            args = verify_structure.parse_arguments()

        assert args.format == "json"

    def test_parse_text_format(self):
        """Verify --format text is parsed correctly."""
        with patch("sys.argv", ["verify_structure.py", "--format", "text"]):
            args = verify_structure.parse_arguments()

        assert args.format == "text"

    def test_parse_custom_root_dir(self, tmp_path):
        """Verify --root-dir argument is parsed correctly."""
        with patch("sys.argv", ["verify_structure.py", "--root-dir", str(tmp_path)]):
            args = verify_structure.parse_arguments()

        assert args.root_dir == tmp_path

    def test_parse_invalid_format(self):
        """Verify invalid format raises error."""
        with patch("sys.argv", ["verify_structure.py", "--format", "xml"]):
            with pytest.raises(SystemExit):
                verify_structure.parse_arguments()


class TestMainFunction:
    """Test suite for main() entry point."""

    def test_main_returns_zero_on_pass(self, tmp_path):
        """Verify main() returns 0 when structure is valid."""
        # Create valid structure
        for dir_path in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
        for file_path in verify_structure.REQUIRED_FILES:
            (tmp_path / file_path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / file_path).touch()

        with patch("sys.argv", ["verify_structure.py", "--root-dir", str(tmp_path)]):
            exit_code = verify_structure.main()

        assert exit_code == 0

    def test_main_returns_one_on_fail(self, tmp_path):
        """Verify main() returns 1 when structure is invalid."""
        # Empty directory
        with patch("sys.argv", ["verify_structure.py", "--root-dir", str(tmp_path)]):
            exit_code = verify_structure.main()

        assert exit_code == 1

    def test_main_returns_one_for_nonexistent_root(self, tmp_path):
        """Verify main() returns 1 if root directory doesn't exist."""
        fake_path = tmp_path / "nonexistent"
        with patch("sys.argv", ["verify_structure.py", "--root-dir", str(fake_path)]):
            exit_code = verify_structure.main()

        assert exit_code == 1

    def test_main_json_output(self, tmp_path, capsys):
        """Verify main() outputs JSON when --format json is used."""
        # Create valid structure
        for dir_path in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
        for file_path in verify_structure.REQUIRED_FILES:
            (tmp_path / file_path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / file_path).touch()

        with patch(
            "sys.argv",
            ["verify_structure.py", "--root-dir", str(tmp_path), "--format", "json"],
        ):
            exit_code = verify_structure.main()

        assert exit_code == 0

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)

        assert parsed["passed"] is True
        assert "found" in parsed
        assert "missing" in parsed

    def test_main_text_output(self, tmp_path, capsys):
        """Verify main() outputs text when --format text is used."""
        # Create valid structure
        for dir_path in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
        for file_path in verify_structure.REQUIRED_FILES:
            (tmp_path / file_path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / file_path).touch()

        with patch(
            "sys.argv",
            ["verify_structure.py", "--root-dir", str(tmp_path), "--format", "text"],
        ):
            exit_code = verify_structure.main()

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out or "PASSED" in captured.out


class TestIntegration:
    """Integration tests for the complete verification workflow."""

    def test_full_verification_workflow_pass(self, tmp_path):
        """Test complete verification workflow with valid structure."""
        # Create complete valid structure
        for dir_path in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
        for file_path in verify_structure.REQUIRED_FILES:
            (tmp_path / file_path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / file_path).touch()

        # Verify structure
        results = verify_structure.verify_structure(tmp_path)

        # Format as JSON
        json_output = verify_structure.format_json_output(results)
        parsed = json.loads(json_output)

        # Verify results
        assert parsed["passed"] is True
        assert len(parsed["missing"]) == 0
        expected_count = len(verify_structure.REQUIRED_DIRECTORIES) + len(
            verify_structure.REQUIRED_FILES
        )
        assert len(parsed["found"]) == expected_count

    def test_full_verification_workflow_fail(self, tmp_path):
        """Test complete verification workflow with invalid structure."""
        # Create incomplete structure (missing some directories)
        (tmp_path / "src").mkdir()
        (tmp_path / "README.md").touch()

        # Verify structure
        results = verify_structure.verify_structure(tmp_path)

        # Format as JSON
        json_output = verify_structure.format_json_output(results)
        parsed = json.loads(json_output)

        # Verify results
        assert parsed["passed"] is False
        assert len(parsed["missing"]) > 0
