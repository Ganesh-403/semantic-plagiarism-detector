#!/usr/bin/env python3
"""
coverage_report.py
------------------
Runs the test suite with HTML coverage reporting and opens the generated
report (htmlcov/index.html) automatically in the default web browser.

Usage:
    python scripts/coverage_report.py
    python scripts/coverage_report.py --cov-report-term
    python scripts/coverage_report.py --no-open

Acceptance Criteria (Issue #1518):
- Execute `pytest --cov=src --cov-report=html`.
- Launch the default web browser pointing to `htmlcov/index.html`.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def check_dependencies() -> None:
    """Ensure pytest and coverage are installed."""
    if not (
        importlib.util.find_spec("coverage") and importlib.util.find_spec("pytest")
    ):
        print("Error: Missing dependency.")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)


def run_coverage_tests(extra_args: list[str] | None = None) -> int:
    """
    Execute pytest with HTML coverage reporting.

    Args:
        extra_args: Optional list of extra pytest arguments (e.g., "-v").

    Returns:
        int: The pytest subprocess return code (0 on success).
    """
    cmd = ["pytest", "--cov=src", "--cov-report=html"]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Executing Coverage Test Runner: {' '.join(cmd)}")

    # Environment isolation
    env = os.environ.copy()
    env["TESTING_MODE"] = "1"

    try:
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode
    except Exception as e:
        print(f"Failed to execute pytest: {e}")
        return 1


def resolve_report_path(report_dir: str | Path = "htmlcov") -> Path:
    """
    Resolve the absolute path to the HTML coverage report index.

    Args:
        report_dir: Directory where the HTML report was generated.

    Returns:
        Path: Absolute path to htmlcov/index.html.
    """
    report_path = Path(report_dir).resolve()
    if not report_path.is_dir():
        print(f"Error: Coverage report directory not found: {report_path}")
        return Path(report_dir) / "index.html"
    return report_path / "index.html"


def open_report_in_browser(report_path: Path) -> bool:
    """
    Open the HTML coverage report in the default web browser.

    Args:
        report_path: Path to the HTML report file to open.

    Returns:
        bool: True if the browser was launched successfully.
    """
    if not report_path.is_file():
        print(f"Error: Coverage report not found: {report_path}")
        return False
    return webbrowser.open(report_path.as_uri())


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the coverage report runner."""
    parser = argparse.ArgumentParser(
        description="Run pytest with HTML coverage reporting and open the report in the browser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the HTML coverage report in the browser.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="htmlcov",
        help="Directory where the HTML coverage report is written.",
    )
    parser.add_argument(
        "--cov-report-term",
        action="store_true",
        help="Also print a terminal coverage summary.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase pytest output verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the coverage report runner."""
    args = parse_arguments()

    check_dependencies()

    extra_args = []
    if args.cov_report_term:
        extra_args.append("--cov-report=term-missing")
    if args.verbose:
        extra_args.append("-vv")

    returncode = run_coverage_tests(extra_args=extra_args)

    report_path = resolve_report_path(args.report_dir)
    if args.no_open:
        print(f"HTML coverage report generated at: {report_path}")
        return returncode

    if returncode == 0:
        if open_report_in_browser(report_path):
            print(f"Opened coverage report in browser: {report_path}")
        else:
            print(f"HTML coverage report generated at: {report_path}")
    else:
        print(f"Tests failed with exit code {returncode}.")
        print(f"HTML coverage report generated at: {report_path}")

    return returncode


if __name__ == "__main__":
    sys.exit(main())
