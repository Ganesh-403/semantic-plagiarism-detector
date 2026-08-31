#!/usr/bin/env python3
"""
scripts/verify_structure.py
---------------------------
Maintenance script to verify that all required project directories and
initialization files exist before build or deployment.

Ensures the repository structure is intact and no critical folders
were accidentally deleted or excluded from version control.

Usage:
    # Verify structure and print human-readable output (default)
    python scripts/verify_structure.py

    # Output results as JSON for CI/CD consumption (Issue #2021)
    python scripts/verify_structure.py --format json

    # Specify custom root directory
    python scripts/verify_structure.py --root-dir /path/to/project

Exit Codes:
    0 - All required directories and files exist.
    1 - One or more required paths are missing.

Acceptance Criteria (Issue #2021):
- Added --format flag accepting 'text' (default) or 'json'
- JSON output: {"passed": bool, "missing": [], "found": [...]}
- Exit code remains 0 for pass, 1 for fail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Required Project Structure ─────────────────────────────────────────────────

# List of required directories relative to the repository root
REQUIRED_DIRECTORIES = [
    "src",
    "src/core",
    "src/db",
    "src/db/migrations",
    "src/api",
    "src/utils",
    "src/visualization",
    "src/security",
    "app",
    "tests",
    "tests/core",
    "tests/db",
    "tests/api",
    "tests/utils",
    "tests/visualization",
    "scripts",
    "config",
    "data",
]

# List of required __init__.py files to ensure Python packages are properly recognized
REQUIRED_INIT_FILES = [
    "src/__init__.py",
    "src/core/__init__.py",
    "src/db/__init__.py",
    "src/api/__init__.py",
    "src/utils/__init__.py",
    "src/visualization/__init__.py",
    "src/security/__init__.py",
    "tests/__init__.py",
]

# Critical configuration and entry-point files
REQUIRED_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "CONTRIBUTING.md",
]


# ── Verification Logic ─────────────────────────────────────────────────────────


def verify_project_structure(
    root_dir: Path,
    check_dirs: list[str],
    check_inits: list[str],
    check_files: list[str],
) -> tuple[bool, list[str], list[str]]:
    """
    Verify that all required directories, __init__.py files, and critical
    files exist in the project structure.

    Args:
        root_dir: The root directory of the repository.
        check_dirs: List of required directory paths (relative to root).
        check_inits: List of required __init__.py file paths.
        check_files: List of required critical file paths.

    Returns:
        A tuple of (is_valid, missing_paths, found_paths) where:
        - is_valid: True if all paths exist, False otherwise.
        - missing_paths: List of string paths that are missing.
        - found_paths: List of string paths that were found.
    """
    missing_paths = []
    found_paths = []

    # Check directories
    for dir_path in check_dirs:
        full_path = root_dir / dir_path
        if not full_path.exists():
            missing_paths.append(f"DIR:  {dir_path}")
            logger.error("Missing required directory: %s", full_path)
        elif not full_path.is_dir():
            missing_paths.append(f"DIR (is file): {dir_path}")
            logger.error("Path exists but is not a directory: %s", full_path)
        else:
            found_paths.append(f"DIR:  {dir_path}")

    # Check __init__.py files
    for init_path in check_inits:
        full_path = root_dir / init_path
        if not full_path.exists():
            missing_paths.append(f"FILE: {init_path}")
            logger.error("Missing required __init__.py: %s", full_path)
        elif not full_path.is_file():
            missing_paths.append(f"FILE (is dir): {init_path}")
            logger.error("Path exists but is not a file: %s", full_path)
        else:
            found_paths.append(f"FILE: {init_path}")

    # Check critical files
    for file_path in check_files:
        full_path = root_dir / file_path
        if not full_path.exists():
            missing_paths.append(f"FILE: {file_path}")
            logger.error("Missing required file: %s", full_path)
        else:
            found_paths.append(f"FILE: {file_path}")

    is_valid = len(missing_paths) == 0
    return is_valid, missing_paths, found_paths


# ── Reporting ──────────────────────────────────────────────────────────────────


def format_text_report(
    is_valid: bool, missing_paths: list[str], found_paths: list[str]
) -> str:
    """Format verification results as human-readable text."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Project Structure Verification Report")
    lines.append("=" * 70)

    if is_valid:
        lines.append("✅ SUCCESS: All required directories and files are present.")
        lines.append(
            "   The project structure is intact and ready for build/deployment."
        )
        lines.append(f"\nFound {len(found_paths)} items:")
        for path in sorted(found_paths):
            lines.append(f"  ✓ {path}")
    else:
        lines.append(
            f"❌ FAILURE: {len(missing_paths)} required path(s) are missing or invalid."
        )
        lines.append("\nMissing Paths:")
        lines.append("-" * 70)
        for path in missing_paths:
            lines.append(f"  - {path}")
        lines.append("-" * 70)
        lines.append(f"\nFound {len(found_paths)} items:")
        for path in sorted(found_paths):
            lines.append(f"  ✓ {path}")
        lines.append(
            "\nPlease restore the missing directories or files before proceeding."
        )

    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)


def format_json_report(
    is_valid: bool, missing_paths: list[str], found_paths: list[str]
) -> str:
    """Format verification results as JSON for CI/CD consumption (Issue #2021).

    Returns JSON with structure: {"passed": bool, "missing": [], "found": []}
    """
    output = {
        "passed": is_valid,
        "missing": missing_paths,
        "found": found_paths,
    }
    return json.dumps(output, indent=2)


# ── CLI Argument Parsing ───────────────────────────────────────────────────────


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the verification script."""
    parser = argparse.ArgumentParser(
        description="Semantic Plagiarism Detection System - Structure Verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Path to the repository root directory. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if ANY warnings are encountered (not just missing paths).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help=(
            "Output format: 'text' for human-readable output (default), "
            "'json' for machine-readable JSON for CI pipelines (Issue #2021)"
        ),
    )

    return parser.parse_args()


# ── Main Execution ─────────────────────────────────────────────────────────────


def main() -> int:
    """Main entry point for the structure verification script.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    args = parse_arguments()

    # Determine root directory
    if args.root_dir:
        root_dir = Path(args.root_dir).resolve()
    else:
        root_dir = ROOT_DIR

    logger.info("=" * 70)
    logger.info("Project Structure Verification")
    logger.info("=" * 70)
    logger.info("Repository root: %s", root_dir)

    if not root_dir.exists():
        logger.error("Specified root directory does not exist: %s", root_dir)
        return 1

    is_valid, missing_paths, found_paths = verify_project_structure(
        root_dir=root_dir,
        check_dirs=REQUIRED_DIRECTORIES,
        check_inits=REQUIRED_INIT_FILES,
        check_files=REQUIRED_FILES,
    )

    # Output results based on format selection
    if args.format == "json":
        # JSON output for CI/CD (Issue #2021)
        print(format_json_report(is_valid, missing_paths, found_paths))
    else:
        # Human-readable text output (default)
        report = format_text_report(is_valid, missing_paths, found_paths)
        print(report)

    # Log summary
    if is_valid:
        logger.info("Verification PASSED.")
        return 0
    else:
        logger.error("Verification FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
