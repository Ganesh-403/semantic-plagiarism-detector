#!/usr/bin/env python3
"""
calibrate_thresholds.py
-----------------------
Automated threshold calibration and backtest harness (Issue #2267).

Sweeps candidate plagiarism thresholds over a labelled dataset (document
pairs + known-plagiarized flag), reports precision / recall / F1 / confusion
matrices, and writes a recommended threshold config that the
``src.core.config`` loader can consume.

The labelled dataset is a CSV with at least three columns:

    doc_a, doc_b, label

plus either one or more precomputed score columns or text columns:

    score, similarity, hybrid, ai, cross_lingual, ...
    text_a, text_b                     (used with --compute-scores)

``label`` accepts 0/1, True/False, "plagiarized"/"original", "yes"/"no", etc.

Usage:
    python scripts/calibrate_thresholds.py --csv labeled_pairs.csv
    python scripts/calibrate_thresholds.py --csv labeled_pairs.csv \
        --score-column hybrid --output config/thresholds.recommended.json
    python scripts/calibrate_thresholds.py --csv labeled_pairs.csv \
        --compute-scores --report calibration_sweep.csv

Acceptance Criteria (Issue #2267):
- Prints an F1 table for the sweep and writes a recommended config.
- Default behavior is unchanged when no calibration config is supplied:
  the loader falls back to DEFAULT_THRESHOLDS.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is importable regardless of the CWD.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.calibration import (  # noqa: E402
    as_label,
    best_evaluation,
    build_recommendation,
    evaluate_thresholds,
    write_recommended_config,
)
from src.core.config import DEFAULT_THRESHOLDS  # noqa: E402

# Score columns the harness knows about, in preference order.  Any of these
# may be present in the labeled CSV.
SCORE_COLUMN_PRIORITY = [
    "score",
    "similarity",
    "similarity_score",
    "semantic",
    "semantic_score",
    "hybrid",
    "hybrid_score",
    "hybrid_similarity",
    "ai",
    "ai_score",
    "ai_probability",
    "ai_probability_score",
    "cross_lingual",
    "cross_lingual_score",
]

REQUIRED_COLUMNS = ("doc_a", "doc_b", "label")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the threshold calibrator."""
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate plagiarism thresholds against a labeled dataset "
            "(document pairs + known-plagiarized flag)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "Path to the labeled dataset CSV. Requires doc_a, doc_b, label "
            "columns plus precomputed score column(s) or text_a/text_b."
        ),
    )
    parser.add_argument(
        "--score-column",
        default=None,
        help=(
            "Explicit precomputed score column to sweep over. If omitted, an "
            "existing recognized score column is auto-detected."
        ),
    )
    parser.add_argument(
        "--compute-scores",
        action="store_true",
        help=(
            "Compute semantic similarity scores from the text_a/text_b columns "
            "using the existing embedding pipeline (requires model availability)."
        ),
    )
    parser.add_argument(
        "--grid-start",
        type=float,
        default=0.30,
        help="Start of the threshold sweep (inclusive).",
    )
    parser.add_argument(
        "--grid-stop",
        type=float,
        default=0.96,
        help="End of the threshold sweep (exclusive).",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.01,
        help="Step size for the threshold sweep.",
    )
    parser.add_argument(
        "--medium",
        type=float,
        default=None,
        help=(
            "Medium severity boundary to preserve in the recommendation "
            f"(default: {DEFAULT_THRESHOLDS.medium})."
        ),
    )
    parser.add_argument(
        "--high",
        type=float,
        default=None,
        help=(
            "High severity boundary to preserve in the recommendation "
            f"(default: {DEFAULT_THRESHOLDS.high})."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "config" / "thresholds.recommended.json"),
        help="Path where the recommended threshold config is written.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "yaml"],
        default="json",
        help="Output format for the recommended config (YAML needs PyYAML).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional path for the full F1 sweep CSV report.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.grid_start < 1.0):
        parser.error("--grid-start must be in [0.0, 1.0).")
    if not (0.0 < args.grid_stop <= 1.0):
        parser.error("--grid-stop must be in (0.0, 1.0].")
    if args.grid_step <= 0.0:
        parser.error("--grid-step must be greater than 0.")
    if args.grid_start >= args.grid_stop:
        parser.error("--grid-start must be less than --grid-stop.")

    return args


def build_grid(start: float, stop: float, step: float) -> list[float]:
    """Generate the threshold sweep grid (start inclusive, stop exclusive)."""
    return [round(float(value), 6) for value in np.arange(start, stop, step)]


def load_labeled_dataset(csv_path: str) -> pd.DataFrame:
    """Load the labeled CSV and normalize the label column to binary ints."""
    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"Error: CSV file '{csv_path}' does not exist.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - report cleanly to the user
        raise SystemExit(f"Error: failed to read CSV '{csv_path}': {exc}") from exc

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(
            "Error: CSV is missing required column(s): "
            + ", ".join(missing)
            + ". Expected at least doc_a, doc_b, label."
        )

    df = df.dropna(subset=["doc_a", "doc_b", "label"])
    df["label_int"] = df["label"].apply(as_label)
    if df.empty:
        raise SystemExit("Error: CSV contains no usable labeled pairs.")

    n_pos = int(df["label_int"].sum())
    n_neg = int(len(df) - n_pos)
    logger.info(
        "Loaded %d labeled pairs from %s (%d plagiarized, %d original).",
        len(df),
        csv_path,
        n_pos,
        n_neg,
    )
    return df


def detect_score_columns(df: pd.DataFrame) -> list[str]:
    """Return recognized precomputed score columns present in the CSV."""
    return [col for col in SCORE_COLUMN_PRIORITY if col in df.columns]


def compute_semantic_scores(df: pd.DataFrame) -> list[float]:
    """Compute row-wise semantic cosine similarities for text_a/text_b pairs."""
    if "text_a" not in df.columns or "text_b" not in df.columns:
        raise SystemExit(
            "Error: --compute-scores requires text_a and text_b columns "
            "in the labeled CSV."
        )

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        from src.core.embedding_model import embed_chunks
    except ImportError as exc:
        raise SystemExit(
            "Error: could not import the embedding pipeline required by "
            "--compute-scores."
        ) from exc

    texts_a = df["text_a"].astype(str).tolist()
    texts_b = df["text_b"].astype(str).tolist()

    try:
        logger.info("Computing semantic embeddings for %d pairs...", len(df))
        emb_a = np.asarray(embed_chunks(texts_a), dtype=float)
        emb_b = np.asarray(embed_chunks(texts_b), dtype=float)
    except Exception as exc:  # noqa: BLE001 - model download/offline failures
        raise SystemExit(
            "Error: semantic scoring failed. Provide precomputed score "
            f"columns in the CSV instead. Details: {exc}"
        ) from exc

    scores = [
        float(cosine_similarity(emb_a[i : i + 1], emb_b[i : i + 1])[0, 0])
        for i in range(len(df))
    ]
    return scores


def print_f1_table(
    evaluations: list[dict],
    score_column: str,
    current_threshold: float,
) -> dict:
    """Print the F1 sweep table and highlight the best threshold row."""
    best = best_evaluation(evaluations)
    best_t = float(best["threshold"])

    print()
    print("=" * 78)
    print(f"  THRESHOLD SWEEP — score column: {score_column!r}")
    print("=" * 78)
    print(
        f"  {'Threshold':>9} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'Accuracy':>9} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    )
    print("  " + "-" * 74)
    for row in evaluations:
        marker = " <-- best" if abs(float(row["threshold"]) - best_t) < 1e-9 else ""
        print(
            f"  {row['threshold']:>9.4f} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['accuracy']:>9.4f} "
            f"{row['tp']:>4} {row['fp']:>4} {row['fn']:>4} {row['tn']:>4}{marker}"
        )
    print("  " + "-" * 74)
    print(
        f"  Current configured threshold : {current_threshold:.4f} "
        f"({current_threshold * 100:.0f}%)"
    )
    print(
        f"  Recommended threshold       : {best_t:.4f} "
        f"({best_t * 100:.0f}%)  F1 = {float(best['f1']):.4f}"
    )
    cm = (
        best["confusion_matrix"]
        if "confusion_matrix" in best
        else {
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
            "tn": best["tn"],
        }
    )
    print(f"  Confusion matrix @ {best_t:.4f}: {cm}")
    print("=" * 78)
    return best


def run_calibration(
    df: pd.DataFrame,
    score_column: str,
    grid: list[float],
) -> list[dict]:
    """Run the threshold sweep over a single score column."""
    scores = df[score_column].astype(float).tolist()
    labels = df["label_int"].tolist()
    return evaluate_thresholds(scores, labels, grid)


def main() -> None:
    """Main entry point for the threshold calibrator."""
    args = parse_arguments()
    grid = build_grid(args.grid_start, args.grid_stop, args.grid_step)

    if len(grid) < 2:
        raise SystemExit(
            "Error: threshold grid too small. Widen --grid-start/--grid-stop "
            "or shrink --grid-step."
        )

    df = load_labeled_dataset(args.csv)

    # ── Select score columns ──────────────────────────────────────────────────
    score_columns: list[str] = []
    if args.score_column is not None:
        if args.score_column not in df.columns:
            raise SystemExit(
                f"Error: score column {args.score_column!r} not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        score_columns = [args.score_column]
    else:
        score_columns = detect_score_columns(df)

    if not score_columns and args.compute_scores:
        logger.info("No precomputed score columns found; computing semantic scores...")
        df = df.copy()
        df["score"] = compute_semantic_scores(df)
        score_columns = ["score"]

    if not score_columns:
        raise SystemExit(
            "Error: no recognized score column found in CSV. Add a precomputed "
            f"score column (one of: {', '.join(SCORE_COLUMN_PRIORITY)}) or pass "
            "--compute-scores with text_a/text_b columns."
        )

    # ── Sweep each score column and pick the overall best ─────────────────────
    print()
    print("=" * 78)
    print("  THRESHOLD CALIBRATION & BACKTEST HARNESS (Issue #2267)")
    print("=" * 78)
    print(f"  Dataset : {args.csv}")
    print(
        f"  Grid    : {args.grid_start:.2f} -> {args.grid_stop:.2f} "
        f"step {args.grid_step:.2f} ({len(grid)} points)"
    )
    print(f"  Columns : {', '.join(score_columns)}")

    best_overall: Optional[dict] = None
    best_column: Optional[str] = None
    best_sweep: list[dict] = []
    best_scores: list[float] = []

    for score_column in score_columns:
        evaluations = run_calibration(df, score_column, grid)
        best = print_f1_table(evaluations, score_column, DEFAULT_THRESHOLDS.plagiarism)

        if best_overall is None or float(best["f1"]) > float(best_overall["f1"]):
            best_overall = best
            best_column = score_column
            best_sweep = evaluations

    assert best_overall is not None and best_column is not None

    # ── Build and write the recommendation ────────────────────────────────────
    medium = args.medium if args.medium is not None else DEFAULT_THRESHOLDS.medium
    high = args.high if args.high is not None else DEFAULT_THRESHOLDS.high

    recommendation = build_recommendation(
        best_overall,
        source="scripts/calibrate_thresholds.py",
        dataset=args.csv,
        score_column=best_column,
        samples=len(df),
        n_plagiarized=int(df["label_int"].sum()),
        sweep=best_sweep,
    )
    # Preserve explicit medium/high severity boundaries.
    recommendation["medium"] = round(float(medium), 6)
    recommendation["high"] = round(float(high), 6)

    written_path = write_recommended_config(
        recommendation,
        args.output,
        fmt=args.format,
    )

    # ── Optional full sweep CSV report ────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(best_sweep).to_csv(report_path, index=False)
        logger.info("Wrote F1 sweep report to %s", report_path)

    print()
    print("=" * 78)
    print("  RECOMMENDATION")
    print("=" * 78)
    print(f"  Best score column : {best_column}")
    print(f"  Recommended config: {written_path}")
    print(
        f"  plagiarism = {recommendation['plagiarism']:.4f}  "
        f"medium = {recommendation['medium']:.4f}  "
        f"high = {recommendation['high']:.4f}"
    )
    print(
        f"  Precision = {recommendation['calibration']['precision']:.4f}  "
        f"Recall = {recommendation['calibration']['recall']:.4f}  "
        f"F1 = {recommendation['calibration']['f1']:.4f}"
    )
    print(
        "  Load this config by setting THRESHOLD_CONFIG_PATH (or copying it "
        "to config/thresholds.json)."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
