#!/usr/bin/env python3
"""
generate_coverage_badge.py
---------------------------
Parses coverage data from .coverage or coverage.json and outputs an SVG badge
for repository documentation.

Acceptance Criteria (Issue #3775):
- Create scripts/generate_coverage_badge.py.
- Calculate percentage and write coverage.svg with appropriate color
  (Green >=85%, Yellow >=70%, Red <70%).

Usage:
    python scripts/generate_coverage_badge.py
    python scripts/generate_coverage_badge.py --input coverage.json --output coverage.svg
    python scripts/generate_coverage_badge.py --percentage 87.5
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent

# Color Constants
COLOR_GREEN = "#44cc11"   # >= 85%
COLOR_YELLOW = "#dfb317"  # >= 70% and < 85%
COLOR_RED = "#e05d44"     # < 70%


def generate_svg_badge(
    label: str,
    message: str,
    label_color: str = "#555555",
    message_color: str = "#007ec6",
) -> str:
    """Generate a shields.io-style SVG badge."""
    label_width = max(30, len(label) * 7 + 10)
    message_width = max(30, len(message) * 7 + 10)
    total_width = label_width + message_width
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="{label_color}" d="M0 0h{label_width}v20H0z"/>
    <path fill="{message_color}" d="M{label_width} 0h{message_width}v20H{label_width}z"/>
    <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width/2}" y="14">{label}</text>
    <text x="{label_width + message_width/2}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
    <text x="{label_width + message_width/2}" y="14">{message}</text>
  </g>
</svg>"""


def get_coverage_color(percentage: float) -> str:
    """
    Determine the badge color based on the coverage percentage.

    Thresholds:
        - Green (>= 85%)
        - Yellow (>= 70% and < 85%)
        - Red (< 70%)

    Args:
        percentage: The code coverage percentage (0.0 - 100.0).

    Returns:
        Hex color code for the badge background.
    """
    if percentage >= 85.0:
        return COLOR_GREEN
    elif percentage >= 70.0:
        return COLOR_YELLOW
    else:
        return COLOR_RED


def parse_json_coverage(json_path: Path | str) -> float:
    """
    Parse coverage percentage from a coverage.json file.

    Args:
        json_path: Path to the coverage.json file.

    Returns:
        float: Coverage percentage (0.0 - 100.0).

    Raises:
        FileNotFoundError: If the json file does not exist.
        ValueError: If JSON content cannot be parsed or lacks coverage data.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Coverage JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse coverage JSON from {path}: {e}") from e

    totals: dict[str, Any] = data.get("totals", {})
    if "percent_covered" in totals:
        return float(totals["percent_covered"])
    elif "percent_covered_display" in totals:
        try:
            return float(totals["percent_covered_display"])
        except ValueError:
            pass

    num_statements = totals.get("num_statements", 0)
    covered_lines = totals.get("covered_lines", 0)
    num_branches = totals.get("num_branches", 0)
    covered_branches = totals.get("covered_branches", 0)

    total_items = num_statements + num_branches
    if total_items > 0:
        return ((covered_lines + covered_branches) / total_items) * 100.0

    if "percent_covered" in data:
        return float(data["percent_covered"])
    if "coverage" in data:
        return float(data["coverage"])

    raise ValueError(f"Could not find coverage totals in JSON file: {path}")


def parse_coverage_db(db_path: Path | str) -> float:
    """
    Parse coverage percentage from a .coverage SQLite/data file using coverage package.

    Args:
        db_path: Path to the .coverage file.

    Returns:
        float: Coverage percentage (0.0 - 100.0).

    Raises:
        FileNotFoundError: If the .coverage file does not exist.
        RuntimeError: If the coverage library is unavailable or parsing fails.
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Coverage data file not found: {path}")

    try:
        import coverage
    except ImportError as e:
        raise RuntimeError(
            "The 'coverage' package is required to parse .coverage files. "
            "Install it via 'pip install coverage' or export to coverage.json."
        ) from e

    try:
        cov = coverage.Coverage(data_file=str(path.resolve()))
        cov.load()
        stream = io.StringIO()
        total_percent = cov.report(file=stream)
        return float(total_percent)
    except Exception as e:
        raise RuntimeError(f"Failed to read coverage from {path}: {e}") from e


def resolve_coverage_file(custom_input: Optional[Path | str] = None) -> Path:
    """
    Locate the coverage file (.coverage or coverage.json).

    Args:
        custom_input: Optional user-provided file path.

    Returns:
        Path: Resolved existing coverage file path.

    Raises:
        FileNotFoundError: If no valid coverage file is found.
    """
    if custom_input is not None:
        p = Path(custom_input).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified coverage file not found: {custom_input}")

    candidates = [
        Path("coverage.json").resolve(),
        ROOT_DIR / "coverage.json",
        Path(".coverage").resolve(),
        ROOT_DIR / ".coverage",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No coverage file found (checked coverage.json and .coverage). "
        "Run 'pytest --cov=src --cov-report=json' or 'pytest --cov=src' first."
    )


def extract_coverage_percentage(input_path: Optional[Path | str] = None) -> float:
    """
    Extract coverage percentage from coverage.json or .coverage file.

    Args:
        input_path: Optional path to coverage file.

    Returns:
        float: Coverage percentage (0.0 - 100.0).
    """
    resolved_path = resolve_coverage_file(input_path)

    if resolved_path.suffix.lower() == ".json" or "json" in resolved_path.name.lower():
        return parse_json_coverage(resolved_path)
    else:
        try:
            return parse_coverage_db(resolved_path)
        except Exception:
            try:
                return parse_json_coverage(resolved_path)
            except Exception:
                raise RuntimeError(
                    f"Unable to parse coverage from {resolved_path} as .coverage or JSON."
                )


def generate_badge_file(
    percentage: float,
    output_path: Path | str = "coverage.svg",
    label: str = "coverage",
    precision: int = 0,
) -> Path:
    """
    Generate an SVG coverage badge and write it to disk.

    Args:
        percentage: Code coverage percentage.
        output_path: Target SVG file path (default 'coverage.svg').
        label: Badge label text (default 'coverage').
        precision: Decimal places for percentage text (default 0).

    Returns:
        Path: Path to the generated SVG file.
    """
    color = get_coverage_color(percentage)

    if precision > 0:
        message = f"{percentage:.{precision}f}%"
    else:
        message = f"{int(round(percentage))}%"

    svg_content = generate_svg_badge(
        label=label,
        message=message,
        label_color="#555555",
        message_color=color,
    )

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return out


def parse_arguments(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for coverage badge generator."""
    parser = argparse.ArgumentParser(
        description="Generate an SVG code coverage badge from .coverage or coverage.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to .coverage or coverage.json file (auto-detected if omitted).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="coverage.svg",
        help="Target output file path for the SVG badge.",
    )
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        default=None,
        help="Manually specify coverage percentage (skips reading coverage files).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="coverage",
        help="Left-side label text on the badge.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=0,
        help="Number of decimal places in percentage display.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress stdout messages.",
    )
    return parser.parse_args(args)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for coverage badge generation script."""
    args = parse_arguments(argv)

    try:
        if args.percentage is not None:
            percentage = float(args.percentage)
        else:
            percentage = extract_coverage_percentage(args.input)

        out_path = generate_badge_file(
            percentage=percentage,
            output_path=args.output,
            label=args.label,
            precision=args.precision,
        )

        if not args.quiet:
            color = get_coverage_color(percentage)
            print(f"Coverage: {percentage:.1f}% -> Badge generated at: {out_path} (Color: {color})")
        return 0

    except Exception as err:
        if not args.quiet:
            print(f"Error generating coverage badge: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
