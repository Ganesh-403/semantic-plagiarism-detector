from __future__ import annotations

"""
test_coverage_report.py
-----------------------
Unit tests for the coverage report runner script (scripts/coverage_report.py).

Validates:
- Coverage test command construction
- Report path resolution
- Browser opening behavior
- Argument parsing logic
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import coverage_report

# ─── Coverage Test Runner Tests ───────────────────────────────────────────────


@patch("coverage_report.subprocess.run")
def test_run_coverage_tests_command(mock_run):
    """Verify pytest runs with coverage flags and returns exit code."""
    mock_run.return_value = MagicMock(returncode=0)

    returncode = coverage_report.run_coverage_tests()

    assert returncode == 0
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "pytest"
    assert "--cov=src" in cmd
    assert "--cov-report=html" in cmd
    env = mock_run.call_args[1]["env"]
    assert env["TESTING_MODE"] == "1"


@patch("coverage_report.subprocess.run")
def test_run_coverage_tests_with_extra_args(mock_run):
    """Verify extra pytest arguments are appended to the command."""
    mock_run.return_value = MagicMock(returncode=2)

    returncode = coverage_report.run_coverage_tests(
        extra_args=["-v", "--cov-report=term-missing"]
    )

    assert returncode == 2
    cmd = mock_run.call_args[0][0]
    assert cmd[-2:] == ["-v", "--cov-report=term-missing"]


@patch("coverage_report.subprocess.run", side_effect=OSError("pytest not found"))
def test_run_coverage_tests_handles_exception(mock_run):
    """Verify a subprocess failure returns a non-zero exit code."""
    assert coverage_report.run_coverage_tests() == 1


# ─── Report Path Resolution Tests ─────────────────────────────────────────────


def test_resolve_report_path_existing(tmp_path):
    """Verify an existing report directory resolves to its index.html."""
    (tmp_path / "index.html").write_text("<html></html>")
    report_path = coverage_report.resolve_report_path(tmp_path)
    assert report_path == tmp_path / "index.html"
    assert report_path.is_absolute()


def test_resolve_report_path_missing_dir(tmp_path):
    """Verify a missing report directory returns an index.html path without error."""
    missing = tmp_path / "does_not_exist"
    report_path = coverage_report.resolve_report_path(missing)
    assert report_path == missing / "index.html"


# ─── Browser Opening Tests ────────────────────────────────────────────────────


@patch("coverage_report.webbrowser.open", return_value=True)
def test_open_report_in_browser_existing(mock_open, tmp_path):
    """Verify an existing report is opened in the browser."""
    report = tmp_path / "index.html"
    report.write_text("<html></html>")
    assert coverage_report.open_report_in_browser(report) is True
    mock_open.assert_called_once_with(report.as_uri())


@patch("coverage_report.webbrowser.open")
def test_open_report_in_browser_missing(mock_open, tmp_path):
    """Verify a missing report is not opened in the browser."""
    missing = tmp_path / "index.html"
    assert coverage_report.open_report_in_browser(missing) is False
    mock_open.assert_not_called()


# ─── Argument Parsing Tests ───────────────────────────────────────────────────


def test_parse_arguments_defaults():
    """Verify default argument values."""
    with patch("sys.argv", ["coverage_report.py"]):
        args = coverage_report.parse_arguments()

    assert args.no_open is False
    assert args.report_dir == "htmlcov"
    assert args.cov_report_term is False
    assert args.verbose is False


def test_parse_arguments_custom_values():
    """Verify custom argument values are parsed correctly."""
    test_args = [
        "coverage_report.py",
        "--no-open",
        "--report-dir",
        "custom_htmlcov",
        "--cov-report-term",
        "-v",
    ]

    with patch("sys.argv", test_args):
        args = coverage_report.parse_arguments()

    assert args.no_open is True
    assert args.report_dir == "custom_htmlcov"
    assert args.cov_report_term is True
    assert args.verbose is True


# ─── Main Entry Point Tests ───────────────────────────────────────────────────


@patch("coverage_report.open_report_in_browser", return_value=True)
@patch("coverage_report.resolve_report_path")
@patch("coverage_report.run_coverage_tests", return_value=0)
@patch("coverage_report.check_dependencies")
def test_main_success_opens_browser(mock_deps, mock_run, mock_resolve, mock_open):
    """Verify main opens the report in the browser on success."""
    mock_resolve.return_value = Path("/abs/htmlcov/index.html")
    args = argparse.Namespace(
        no_open=False,
        report_dir="htmlcov",
        cov_report_term=False,
        verbose=False,
    )

    with patch("coverage_report.parse_arguments", return_value=args):
        assert coverage_report.main() == 0

    mock_run.assert_called_once_with(extra_args=[])
    mock_open.assert_called_once()


@patch("coverage_report.open_report_in_browser")
@patch("coverage_report.resolve_report_path")
@patch("coverage_report.run_coverage_tests", return_value=0)
@patch("coverage_report.check_dependencies")
def test_main_no_open_skips_browser(mock_deps, mock_run, mock_resolve, mock_open):
    """Verify main skips the browser when --no-open is used."""
    mock_resolve.return_value = Path("/abs/htmlcov/index.html")
    args = argparse.Namespace(
        no_open=True,
        report_dir="htmlcov",
        cov_report_term=False,
        verbose=False,
    )

    with patch("coverage_report.parse_arguments", return_value=args):
        assert coverage_report.main() == 0

    mock_open.assert_not_called()


@patch("coverage_report.open_report_in_browser")
@patch("coverage_report.resolve_report_path")
@patch("coverage_report.run_coverage_tests", return_value=3)
@patch("coverage_report.check_dependencies")
def test_main_test_failure_returns_code(mock_deps, mock_run, mock_resolve, mock_open):
    """Verify main propagates a non-zero pytest exit code."""
    mock_resolve.return_value = Path("/abs/htmlcov/index.html")
    args = argparse.Namespace(
        no_open=False,
        report_dir="htmlcov",
        cov_report_term=True,
        verbose=True,
    )

    with patch("coverage_report.parse_arguments", return_value=args):
        assert coverage_report.main() == 3

    mock_run.assert_called_once_with(extra_args=["--cov-report=term-missing", "-vv"])
