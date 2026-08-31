"""
test_generate_coverage_badge.py
---------------------------------
Unit tests for the coverage badge generator script (scripts/generate_coverage_badge.py).

Validates:
- Color threshold calculation (Green >=85%, Yellow >=70%, Red <70%)
- Parsing of coverage.json and .coverage data
- Coverage file discovery and resolution
- SVG badge generation and file writing
- CLI argument parsing and main entry point execution
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts and root directories to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_coverage_badge
from generate_coverage_badge import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    extract_coverage_percentage,
    generate_badge_file,
    get_coverage_color,
    main,
    parse_arguments,
    parse_coverage_db,
    parse_json_coverage,
    resolve_coverage_file,
)


# ─── Color Threshold Tests ───────────────────────────────────────────────────


class TestCoverageColorThresholds:
    """Validate acceptance criteria: Green >=85%, Yellow >=70%, Red <70%."""

    @pytest.mark.parametrize(
        "percentage",
        [85.0, 85.1, 90.0, 99.9, 100.0, 85.0001],
    )
    def test_green_threshold(self, percentage: float):
        """Percentages >= 85% must return Green color."""
        assert get_coverage_color(percentage) == COLOR_GREEN

    @pytest.mark.parametrize(
        "percentage",
        [70.0, 70.1, 75.0, 84.9, 84.999],
    )
    def test_yellow_threshold(self, percentage: float):
        """Percentages >= 70% and < 85% must return Yellow color."""
        assert get_coverage_color(percentage) == COLOR_YELLOW

    @pytest.mark.parametrize(
        "percentage",
        [0.0, 10.5, 50.0, 69.0, 69.9, 69.999],
    )
    def test_red_threshold(self, percentage: float):
        """Percentages < 70% must return Red color."""
        assert get_coverage_color(percentage) == COLOR_RED


# ─── JSON Coverage Parsing Tests ──────────────────────────────────────────────


class TestParseJsonCoverage:
    """Test parsing logic for coverage.json files."""

    def test_parse_standard_json(self, tmp_path):
        """Parse standard coverage.json with percent_covered in totals."""
        json_file = tmp_path / "coverage.json"
        data = {
            "totals": {
                "covered_lines": 85,
                "num_statements": 100,
                "percent_covered": 87.5,
            }
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_json_coverage(json_file)
        assert result == 87.5

    def test_parse_json_with_display_string(self, tmp_path):
        """Parse coverage.json with percent_covered_display in totals."""
        json_file = tmp_path / "coverage.json"
        data = {"totals": {"percent_covered_display": "92.4"}}
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_json_coverage(json_file)
        assert result == 92.4

    def test_parse_json_calculate_from_statements(self, tmp_path):
        """Calculate percentage from lines and branches when percent is missing."""
        json_file = tmp_path / "coverage.json"
        data = {
            "totals": {
                "covered_lines": 80,
                "num_statements": 100,
                "covered_branches": 10,
                "num_branches": 20,
            }
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_json_coverage(json_file)
        # (80 + 10) / (100 + 20) = 90 / 120 = 75.0%
        assert result == 75.0

    def test_parse_json_fallback_flat_structure(self, tmp_path):
        """Parse flat dictionary containing coverage/percent_covered."""
        json_file = tmp_path / "coverage.json"
        data = {"percent_covered": 95.0}
        json_file.write_text(json.dumps(data), encoding="utf-8")

        result = parse_json_coverage(json_file)
        assert result == 95.0

    def test_parse_json_missing_file_raises_error(self, tmp_path):
        """Raise FileNotFoundError if file does not exist."""
        missing = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            parse_json_coverage(missing)

    def test_parse_json_invalid_content_raises_error(self, tmp_path):
        """Raise ValueError if JSON content is corrupted."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json}", encoding="utf-8")

        with pytest.raises(ValueError):
            parse_json_coverage(bad_json)

    def test_parse_json_no_totals_raises_error(self, tmp_path):
        """Raise ValueError if JSON has no recognizable coverage data."""
        empty_json = tmp_path / "empty.json"
        empty_json.write_text(json.dumps({"some_key": 123}), encoding="utf-8")

        with pytest.raises(ValueError):
            parse_json_coverage(empty_json)


# ─── Coverage DB / File Parsing Tests ─────────────────────────────────────────


class TestParseCoverageDb:
    """Test parsing logic for .coverage files."""

    def test_parse_db_missing_file_raises_error(self, tmp_path):
        """Raise FileNotFoundError when .coverage does not exist."""
        missing = tmp_path / ".coverage"
        with pytest.raises(FileNotFoundError):
            parse_coverage_db(missing)

    @patch("coverage.Coverage")
    def test_parse_db_success(self, mock_coverage_class, tmp_path):
        """Verify .coverage file is loaded and reported via coverage library."""
        dummy_file = tmp_path / ".coverage"
        dummy_file.write_text("sqlite-data")

        mock_cov_instance = MagicMock()
        mock_cov_instance.report.return_value = 88.5
        mock_coverage_class.return_value = mock_cov_instance

        percentage = parse_coverage_db(dummy_file)

        assert percentage == 88.5
        mock_coverage_class.assert_called_once_with(data_file=str(dummy_file.resolve()))
        mock_cov_instance.load.assert_called_once()
        mock_cov_instance.report.assert_called_once()


# ─── Coverage File Resolution Tests ───────────────────────────────────────────


class TestResolveCoverageFile:
    """Test locating .coverage or coverage.json."""

    def test_resolve_custom_existing_file(self, tmp_path):
        """Return custom file if it exists."""
        custom = tmp_path / "my_coverage.json"
        custom.write_text("{}")
        resolved = resolve_coverage_file(custom)
        assert resolved == custom.resolve()

    def test_resolve_custom_missing_file_raises_error(self, tmp_path):
        """Raise FileNotFoundError if custom file does not exist."""
        custom = tmp_path / "not_found.json"
        with pytest.raises(FileNotFoundError):
            resolve_coverage_file(custom)

    def test_resolve_auto_detect_json(self, tmp_path, monkeypatch):
        """Auto-detect coverage.json in current directory."""
        monkeypatch.chdir(tmp_path)
        json_file = tmp_path / "coverage.json"
        json_file.write_text("{}")

        resolved = resolve_coverage_file()
        assert resolved == json_file.resolve()

    def test_resolve_auto_detect_db(self, tmp_path, monkeypatch):
        """Auto-detect .coverage in current directory."""
        monkeypatch.chdir(tmp_path)
        db_file = tmp_path / ".coverage"
        db_file.write_text("")

        resolved = resolve_coverage_file()
        assert resolved == db_file.resolve()

    def test_resolve_not_found_raises_error(self, tmp_path, monkeypatch):
        """Raise FileNotFoundError when neither file exists."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)

        # Also patch ROOT_DIR candidates to ensure non-existence
        with patch.object(generate_coverage_badge, "ROOT_DIR", empty_dir):
            with pytest.raises(FileNotFoundError):
                resolve_coverage_file()


# ─── Badge Generation Tests ───────────────────────────────────────────────────


class TestGenerateBadgeFile:
    """Test SVG badge file generation."""

    def test_generate_badge_green(self, tmp_path):
        """Verify green badge output content and file creation."""
        out_svg = tmp_path / "coverage.svg"
        generated_path = generate_badge_file(percentage=90.0, output_path=out_svg)

        assert generated_path.is_file()
        content = generated_path.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "coverage" in content
        assert "90%" in content
        assert COLOR_GREEN in content

    def test_generate_badge_yellow(self, tmp_path):
        """Verify yellow badge output content and file creation."""
        out_svg = tmp_path / "coverage.svg"
        generated_path = generate_badge_file(percentage=75.5, output_path=out_svg)

        assert generated_path.is_file()
        content = generated_path.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "76%" in content
        assert COLOR_YELLOW in content

    def test_generate_badge_red(self, tmp_path):
        """Verify red badge output content and file creation."""
        out_svg = tmp_path / "coverage.svg"
        generated_path = generate_badge_file(percentage=65.0, output_path=out_svg)

        assert generated_path.is_file()
        content = generated_path.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "65%" in content
        assert COLOR_RED in content

    def test_generate_badge_with_precision(self, tmp_path):
        """Verify custom precision formatting."""
        out_svg = tmp_path / "coverage.svg"
        generated_path = generate_badge_file(
            percentage=88.45, output_path=out_svg, precision=1
        )

        content = generated_path.read_text(encoding="utf-8")
        assert "88.5%" in content

    def test_generate_badge_creates_parent_dirs(self, tmp_path):
        """Verify parent directories are created automatically."""
        out_svg = tmp_path / "nested" / "dir" / "coverage.svg"
        generated_path = generate_badge_file(percentage=100.0, output_path=out_svg)

        assert generated_path.is_file()


# ─── Argument Parsing & Main CLI Tests ─────────────────────────────────────────


class TestCLIExecution:
    """Test CLI argument handling and execution."""

    def test_parse_arguments_defaults(self):
        """Verify default CLI arguments."""
        args = parse_arguments([])
        assert args.input is None
        assert args.output == "coverage.svg"
        assert args.percentage is None
        assert args.label == "coverage"
        assert args.precision == 0
        assert args.quiet is False

    def test_parse_arguments_custom(self):
        """Verify custom CLI arguments."""
        args = parse_arguments(
            [
                "--input",
                "custom.json",
                "--output",
                "docs/badge.svg",
                "--percentage",
                "85.5",
                "--label",
                "test-cov",
                "--precision",
                "2",
                "--quiet",
            ]
        )
        assert args.input == "custom.json"
        assert args.output == "docs/badge.svg"
        assert args.percentage == 85.5
        assert args.label == "test-cov"
        assert args.precision == 2
        assert args.quiet is True

    def test_main_with_percentage_argument(self, tmp_path):
        """Verify main succeeds when --percentage is provided directly."""
        out_svg = tmp_path / "coverage.svg"
        ret = main(["--percentage", "92.0", "--output", str(out_svg), "--quiet"])

        assert ret == 0
        assert out_svg.is_file()
        content = out_svg.read_text(encoding="utf-8")
        assert "92%" in content
        assert COLOR_GREEN in content

    def test_main_with_json_file(self, tmp_path):
        """Verify main succeeds with --input json file."""
        json_file = tmp_path / "coverage.json"
        json_file.write_text(
            json.dumps({"totals": {"percent_covered": 78.0}}),
            encoding="utf-8",
        )
        out_svg = tmp_path / "coverage.svg"

        ret = main(
            ["--input", str(json_file), "--output", str(out_svg), "--quiet"]
        )

        assert ret == 0
        assert out_svg.is_file()
        content = out_svg.read_text(encoding="utf-8")
        assert "78%" in content
        assert COLOR_YELLOW in content

    def test_main_handles_error_gracefully(self, tmp_path):
        """Verify main returns 1 on error and doesn't crash."""
        ret = main(["--input", str(tmp_path / "nonexistent.json"), "--quiet"])
        assert ret == 1
