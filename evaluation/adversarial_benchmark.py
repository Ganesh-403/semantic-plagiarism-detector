"""
adversarial_benchmark.py
-------------------------
Benchmark runner for adversarial paraphrase detection.

Evaluates the plagiarism detection pipeline against adversarial
transformation categories and produces machine-readable metrics
identifying which transformations cause the largest detection degradation.

Usage (from project root):
    python -m evaluation.adversarial_benchmark

Outputs (saved to evaluation/results/):
    - adversarial_metrics.json      Per-category and overall metrics
    - adversarial_threshold.csv     Threshold sweep per category
    - adversarial_summary.txt       Human-readable summary report
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ── Ensure project root is importable ──────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.adversarial_generator import (  # noqa: E402
    AdversarialBenchmarkGenerator,
    TRANSFORMATIONS,
)
from evaluation.evaluate import (  # noqa: E402
    compute_metrics_at_threshold,
    load_benchmark,
    sweep_thresholds,
)
from src.core.embedding_model import embed_chunks  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.75"))
REGRESSION_THRESHOLD = float(os.getenv("REGRESSION_THRESHOLD", "0.10"))

import os  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  Similarity computation
# ══════════════════════════════════════════════════════════════════════════════


def compute_pair_similarities(pairs: list[dict[str, Any]]) -> np.ndarray:
    """Compute semantic cosine similarity for each pair using Sentence Transformers.

    Args:
        pairs: List of pair dicts with 'text_a' and 'text_b' keys.

    Returns:
        1-D numpy array of similarity scores (0-1).
    """
    if not pairs:
        return np.array([])

    texts_a = [p["text_a"] for p in pairs]
    texts_b = [p["text_b"] for p in pairs]

    emb_a = embed_chunks(texts_a)
    emb_b = embed_chunks(texts_b)

    similarities = np.array([
        float(sklearn_cosine(emb_a[i : i + 1], emb_b[i : i + 1])[0, 0])
        for i in range(len(pairs))
    ])
    return similarities


# ══════════════════════════════════════════════════════════════════════════════
#  Category-level metrics
# ══════════════════════════════════════════════════════════════════════════════


def compute_category_metrics(
    similarities: np.ndarray,
    threshold: float,
    category: str,
    n_total: int,
) -> dict[str, Any]:
    """Compute detection metrics for a single transformation category.

    Args:
        similarities: Similarity scores for pairs in this category.
        threshold: Detection threshold.
        category: Category name.
        n_total: Total number of pairs across all categories.

    Returns:
        Dictionary of metrics for this category.
    """
    labels = np.ones(len(similarities), dtype=int)  # All should be detected
    metrics = compute_metrics_at_threshold(similarities, labels, threshold)

    detection_rate = float(metrics["tp"]) / len(similarities) if len(similarities) > 0 else 0.0
    mean_sim = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
    std_sim = float(np.std(similarities)) if len(similarities) > 0 else 0.0
    min_sim = float(np.min(similarities)) if len(similarities) > 0 else 0.0
    max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0

    return {
        "category": category,
        "n_pairs": len(similarities),
        "n_total_all_categories": n_total,
        "detection_rate": round(detection_rate, 4),
        "mean_similarity": round(mean_sim, 4),
        "std_similarity": round(std_sim, 4),
        "min_similarity": round(min_sim, 4),
        "max_similarity": round(max_sim, 4),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "threshold": threshold,
    }


def compute_overall_metrics(
    all_similarities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute overall metrics across all adversarial pairs.

    Args:
        all_similarities: All adversarial similarity scores.
        threshold: Detection threshold.

    Returns:
        Overall metrics dictionary.
    """
    labels = np.ones(len(all_similarities), dtype=int)
    metrics = compute_metrics_at_threshold(all_similarities, labels, threshold)

    return {
        "n_pairs": len(all_similarities),
        "detection_rate": round(float(metrics["tp"]) / len(all_similarities), 4) if len(all_similarities) > 0 else 0.0,
        "mean_similarity": round(float(np.mean(all_similarities)), 4) if len(all_similarities) > 0 else 0.0,
        "std_similarity": round(float(np.std(all_similarities)), 4) if len(all_similarities) > 0 else 0.0,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "threshold": threshold,
    }


def compute_negative_metrics(
    negative_sims: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute false-positive rate on unrelated document pairs.

    Args:
        negative_sims: Similarity scores for negative pairs.
        threshold: Detection threshold.

    Returns:
        False-positive metrics.
    """
    if len(negative_sims) == 0:
        return {
            "n_pairs": 0,
            "false_positive_rate": 0.0,
            "mean_similarity": 0.0,
            "std_similarity": 0.0,
            "threshold": threshold,
        }

    predictions = (negative_sims >= threshold).astype(int)
    n_false_positive = int(np.sum(predictions == 1))
    fpr = n_false_positive / len(negative_sims)

    return {
        "n_pairs": len(negative_sims),
        "false_positive_rate": round(fpr, 4),
        "n_false_positives": n_false_positive,
        "mean_similarity": round(float(np.mean(negative_sims)), 4),
        "std_similarity": round(float(np.std(negative_sims)), 4),
        "min_similarity": round(float(np.min(negative_sims)), 4),
        "max_similarity": round(float(np.max(negative_sims)), 4),
        "threshold": threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Degradation analysis
# ══════════════════════════════════════════════════════════════════════════════


def identify_worst_categories(
    category_metrics: dict[str, dict[str, Any]],
    baseline_detection_rate: float = 1.0,
) -> list[dict[str, Any]]:
    """Identify transformation categories causing the largest degradation.

    Args:
        category_metrics: Metrics per category.
        baseline_detection_rate: Expected detection rate (1.0 = perfect).

    Returns:
        List of categories sorted by degradation (worst first).
    """
    degradations = []
    for cat, metrics in category_metrics.items():
        degradation = baseline_detection_rate - metrics["detection_rate"]
        degradations.append({
            "category": cat,
            "detection_rate": metrics["detection_rate"],
            "degradation": round(degradation, 4),
            "mean_similarity": metrics["mean_similarity"],
            "n_pairs": metrics["n_pairs"],
        })

    degradations.sort(key=lambda x: x["degradation"], reverse=True)
    return degradations


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold sweep per category
# ══════════════════════════════════════════════════════════════════════════════


def sweep_category_thresholds(
    category_sims: dict[str, np.ndarray],
    start: float = 0.30,
    stop: float = 0.96,
    step: float = 0.01,
) -> pd.DataFrame:
    """Sweep thresholds per category and return a combined DataFrame.

    Args:
        category_sims: Mapping of category → similarity array.
        start: Start threshold.
        stop: End threshold (exclusive).
        step: Step size.

    Returns:
        DataFrame with columns: category, threshold, precision, recall, f1, detection_rate.
    """
    rows = []
    for cat, sims in category_sims.items():
        labels = np.ones(len(sims), dtype=int)
        for t in np.arange(start, stop, step):
            metrics = compute_metrics_at_threshold(sims, labels, t)
            detection_rate = float(metrics["tp"]) / len(sims) if len(sims) > 0 else 0.0
            rows.append({
                "category": cat,
                "threshold": round(float(t), 3),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "detection_rate": round(detection_rate, 4),
                "n_pairs": len(sims),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Main benchmark runner
# ══════════════════════════════════════════════════════════════════════════════


class AdversarialBenchmarkRunner:
    """Runs the adversarial benchmark and produces metrics reports."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        seed: int = 42,
        max_ratio: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.seed = seed
        self.max_ratio = max_ratio
        self.generator = AdversarialBenchmarkGenerator(seed=seed, max_ratio=max_ratio)
        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Execute the full benchmark pipeline.

        Steps:
            1. Generate adversarial pairs from the baseline dataset.
            2. Compute semantic similarities for all pairs.
            3. Compute per-category and overall metrics.
            4. Analyze degradation by transformation category.
            5. Compute false-positive rate on negative pairs.
            6. Save all results.

        Returns:
            Complete results dictionary.
        """
        start_time = time.time()

        # ── Step 1: Generate adversarial benchmark ──────────────────────────
        print("\n  [1/6] Generating adversarial benchmark...")
        self.generator.load_dataset()
        benchmark = self.generator.generate_full_benchmark()
        adversarial_pairs = benchmark["adversarial_pairs"]
        negative_pairs = benchmark["negative_pairs"]
        print(f"        Generated {len(adversarial_pairs)} adversarial + {len(negative_pairs)} negative pairs")

        # ── Step 2: Compute similarities ────────────────────────────────────
        print("  [2/6] Computing semantic similarities...")
        adv_sims = compute_pair_similarities(adversarial_pairs)
        neg_sims = compute_pair_similarities(negative_pairs) if negative_pairs else np.array([])
        print(f"        Adversarial similarities: mean={np.mean(adv_sims):.4f}, std={np.std(adv_sims):.4f}")
        if len(neg_sims) > 0:
            print(f"        Negative similarities:    mean={np.mean(neg_sims):.4f}, std={np.std(neg_sims):.4f}")

        # ── Step 3: Compute per-category metrics ────────────────────────────
        print("  [3/6] Computing per-category metrics...")
        category_sims: dict[str, np.ndarray] = defaultdict(list)
        for i, pair in enumerate(adversarial_pairs):
            category_sims[pair["transformation"]].append(adv_sims[i])
        category_sims = {k: np.array(v) for k, v in category_sims.items()}

        category_metrics: dict[str, dict[str, Any]] = {}
        for cat, sims in sorted(category_sims.items()):
            metrics = compute_category_metrics(sims, self.threshold, cat, len(adversarial_pairs))
            category_metrics[cat] = metrics
            print(f"        {cat:<28} det_rate={metrics['detection_rate']:.2%}  mean_sim={metrics['mean_similarity']:.4f}")

        # ── Step 4: Overall metrics ─────────────────────────────────────────
        print("  [4/6] Computing overall metrics...")
        overall = compute_overall_metrics(adv_sims, self.threshold)
        print(f"        Overall detection rate: {overall['detection_rate']:.2%}")
        print(f"        Overall mean similarity: {overall['mean_similarity']:.4f}")

        # ── Step 5: Degradation analysis ────────────────────────────────────
        print("  [5/6] Analyzing degradation by category...")
        worst = identify_worst_categories(category_metrics, baseline_detection_rate=1.0)
        for rank, entry in enumerate(worst, 1):
            print(f"        #{rank} {entry['category']:<28} degradation={entry['degradation']:.2%}")

        # ── Step 6: Negative pair analysis ──────────────────────────────────
        print("  [6/6] Computing false-positive rate...")
        negative_metrics = compute_negative_metrics(neg_sims, self.threshold)
        print(f"        False-positive rate: {negative_metrics['false_positive_rate']:.2%}")

        # ── Threshold sweep ─────────────────────────────────────────────────
        print("  Running threshold sweep per category...")
        sweep_df = sweep_category_thresholds(category_sims)

        # ── Compile results ─────────────────────────────────────────────────
        elapsed = time.time() - start_time
        self.results = {
            "metadata": {
                "benchmark_name": benchmark["name"],
                "benchmark_version": benchmark["version"],
                "seed": self.seed,
                "max_ratio": self.max_ratio,
                "threshold": self.threshold,
                "n_adversarial_pairs": len(adversarial_pairs),
                "n_negative_pairs": len(negative_pairs),
                "n_total_pairs": len(adversarial_pairs) + len(negative_pairs),
                "n_transformations": len(TRANSFORMATIONS),
                "elapsed_seconds": round(elapsed, 2),
            },
            "overall": overall,
            "per_category": category_metrics,
            "degradation_analysis": worst,
            "negative_pair_analysis": negative_metrics,
            "threshold_sweep": sweep_df.to_dict(orient="records"),
            "baseline_comparison": self._compare_to_baseline(),
        }

        return self.results

    def _compare_to_baseline(self) -> dict[str, Any]:
        """Compare adversarial results against the baseline evaluation."""
        baseline_metrics_path = RESULTS_DIR / "metrics.json"
        if not baseline_metrics_path.exists():
            return {"available": False, "reason": "Baseline metrics.json not found"}

        try:
            with open(baseline_metrics_path, "r") as f:
                baseline = json.load(f)

            baseline_f1 = baseline.get("semantic", {}).get("best_f1", 0.0)
            baseline_auc = baseline.get("semantic", {}).get("roc_auc", 0.0)

            return {
                "available": True,
                "baseline_f1": baseline_f1,
                "baseline_auc": baseline_auc,
                "adversarial_f1": self.results.get("overall", {}).get("f1", 0.0),
                "f1_delta": round(
                    self.results.get("overall", {}).get("f1", 0.0) - baseline_f1, 4
                ),
            }
        except Exception:
            return {"available": False, "reason": "Failed to read baseline metrics"}

    def save_results(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Save all benchmark results to disk.

        Args:
            output_dir: Directory for output files.

        Returns:
            Dictionary mapping output name → file path.
        """
        if not self.results:
            self.run()

        out = output_dir or RESULTS_DIR
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        # 1. JSON metrics
        metrics_path = out / "adversarial_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        saved["metrics_json"] = metrics_path

        # 2. Threshold sweep CSV
        if "threshold_sweep" in self.results:
            sweep_df = pd.DataFrame(self.results["threshold_sweep"])
            sweep_path = out / "adversarial_threshold.csv"
            sweep_df.to_csv(sweep_path, index=False)
            saved["threshold_csv"] = sweep_path

        # 3. Human-readable summary
        summary_path = out / "adversarial_summary.txt"
        summary_path.write_text(self._format_summary(), encoding="utf-8")
        saved["summary_txt"] = summary_path

        # 4. Adversarial benchmark dataset
        bench_path = self.generator.save_benchmark(out / "adversarial_benchmark.json")
        saved["benchmark_json"] = bench_path

        return saved

    def _format_summary(self) -> str:
        """Format a human-readable summary report."""
        meta = self.results.get("metadata", {})
        overall = self.results.get("overall", {})
        worst = self.results.get("degradation_analysis", [])
        neg = self.results.get("negative_pair_analysis", {})
        baseline_comp = self.results.get("baseline_comparison", {})

        lines = [
            "=" * 72,
            "  ADVERSARIAL PARAPHRASE DETECTION — BENCHMARK REPORT",
            "=" * 72,
            "",
            f"  Benchmark       : {meta.get('benchmark_name', 'N/A')}",
            f"  Version         : {meta.get('benchmark_version', 'N/A')}",
            f"  Seed            : {meta.get('seed', 'N/A')}",
            f"  Max ratio       : {meta.get('max_ratio', 'N/A')}",
            f"  Threshold       : {meta.get('threshold', 'N/A')}",
            f"  Transformations : {meta.get('n_transformations', 'N/A')}",
            f"  Adversarial pairs: {meta.get('n_adversarial_pairs', 'N/A')}",
            f"  Negative pairs  : {meta.get('n_negative_pairs', 'N/A')}",
            f"  Total pairs     : {meta.get('n_total_pairs', 'N/A')}",
            f"  Elapsed         : {meta.get('elapsed_seconds', 'N/A')}s",
            "",
            "-" * 72,
            "  OVERALL METRICS",
            "-" * 72,
            f"  Detection rate  : {overall.get('detection_rate', 0):.2%}",
            f"  Mean similarity : {overall.get('mean_similarity', 0):.4f}",
            f"  Precision       : {overall.get('precision', 0):.4f}",
            f"  Recall          : {overall.get('recall', 0):.4f}",
            f"  F1              : {overall.get('f1', 0):.4f}",
            "",
            "-" * 72,
            "  DEGRADATION ANALYSIS (worst → best)",
            "-" * 72,
        ]

        lines.append(f"  {'Rank':<6} {'Category':<30} {'Det. Rate':>10} {'Degradation':>12} {'Mean Sim':>10}")
        lines.append("  " + "-" * 68)
        for rank, entry in enumerate(worst, 1):
            lines.append(
                f"  {rank:<6} {entry['category']:<30} "
                f"{entry['detection_rate']:>9.2%} "
                f"{entry['degradation']:>11.2%} "
                f"{entry['mean_similarity']:>10.4f}"
            )

        lines.extend([
            "",
            "-" * 72,
            "  FALSE POSITIVE ANALYSIS",
            "-" * 72,
            f"  Negative pairs  : {neg.get('n_pairs', 0)}",
            f"  False-pos rate   : {neg.get('false_positive_rate', 0):.2%}",
            f"  Mean similarity  : {neg.get('mean_similarity', 0):.4f}",
            f"  Threshold        : {neg.get('threshold', 'N/A')}",
        ])

        if baseline_comp.get("available", False):
            lines.extend([
                "",
                "-" * 72,
                "  BASELINE COMPARISON",
                "-" * 72,
                f"  Baseline F1      : {baseline_comp.get('baseline_f1', 0):.4f}",
                f"  Adversarial F1   : {baseline_comp.get('adversarial_f1', 0):.4f}",
                f"  F1 Delta         : {baseline_comp.get('f1_delta', 0):+.4f}",
            ])

        # Per-category detail
        per_cat = self.results.get("per_category", {})
        if per_cat:
            lines.extend([
                "",
                "-" * 72,
                "  PER-CATEGORY DETAIL",
                "-" * 72,
            ])
            header = f"  {'Category':<30} {'N':>4} {'Det%':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'μSim':>7} {'σSim':>7}"
            lines.append(header)
            lines.append("  " + "-" * 68)
            for cat in sorted(per_cat.keys()):
                m = per_cat[cat]
                lines.append(
                    f"  {cat:<30} {m['n_pairs']:>4} "
                    f"{m['detection_rate']:>6.1%} "
                    f"{m['precision']:>7.4f} "
                    f"{m['recall']:>7.4f} "
                    f"{m['f1']:>7.4f} "
                    f"{m['mean_similarity']:>7.4f} "
                    f"{m['std_similarity']:>7.4f}"
                )

        # Regression check
        lines.extend([
            "",
            "-" * 72,
            "  REGRESSION CHECK",
            "-" * 72,
        ])
        regressions = [
            e for e in worst if e["degradation"] > REGRESSION_THRESHOLD
        ]
        if regressions:
            lines.append(f"  ⚠ {len(regressions)} categories show degradation > {REGRESSION_THRESHOLD:.0%}:")
            for r in regressions:
                lines.append(f"    - {r['category']}: degradation={r['degradation']:.2%}")
            lines.append("")
            lines.append("  ACTION REQUIRED: Investigate detection pipeline for these categories.")
        else:
            lines.append(f"  ✅ No categories exceed {REGRESSION_THRESHOLD:.0%} degradation threshold.")
            lines.append("  The detection pipeline is robust against all tested transformations.")

        lines.extend(["", "=" * 72, "  [DONE] Benchmark complete.", "=" * 72, ""])
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary report to stdout."""
        if not self.results:
            self.run()
        print(self._format_summary())


# ══════════════════════════════════════════════════════════════════════════════
#  CI integration
# ══════════════════════════════════════════════════════════════════════════════


def ci_regression_check(results: dict[str, Any]) -> int:
    """Check for regressions and return exit code (0 = pass, 1 = fail).

    Args:
        results: Benchmark results dictionary.

    Returns:
        Exit code: 0 if no regressions, 1 if regression detected.
    """
    worst = results.get("degradation_analysis", [])
    regressions = [
        e for e in worst if e["degradation"] > REGRESSION_THRESHOLD
    ]

    if regressions:
        logger.error(
            "REGRESSION DETECTED: %d categories exceed %.0f%% degradation",
            len(regressions), REGRESSION_THRESHOLD * 100,
        )
        for r in regressions:
            logger.error(
                "  %s: degradation=%.2f%%",
                r["category"], r["degradation"] * 100,
            )
        return 1

    logger.info(
        "No regressions detected (threshold=%.0f%%)",
        REGRESSION_THRESHOLD * 100,
    )
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the adversarial benchmark and save results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runner = AdversarialBenchmarkRunner()
    runner.run()
    saved = runner.save_results()
    runner.print_summary()

    print("  Output files:")
    for name, path in saved.items():
        print(f"    {name:<20} {path}")

    # CI regression check
    exit_code = ci_regression_check(runner.results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
