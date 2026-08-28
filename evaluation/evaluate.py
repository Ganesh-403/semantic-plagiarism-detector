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
evaluate.py
-----------
Evaluation framework for the Semantic Plagiarism Detection System.

Computes precision, recall, F1-score, and ROC-AUC on a labelled benchmark
dataset.  Compares the **semantic approach** (Sentence Transformers) against
a **TF-IDF lexical baseline** to demonstrate the value of semantic embeddings
for paraphrase detection.

Usage (from project root):
    python -m evaluation.evaluate

Outputs (saved to evaluation/results/):
    - metrics.json          Overall scores at the optimal threshold
    - threshold_sweep.csv   Precision / recall / F1 at every threshold
    - roc_curve.png         ROC curve (semantic vs TF-IDF)
    - pr_curve.png          Precision-Recall curve
    - similarity_distribution.png   Score histograms by label
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ── Ensure project root is importable ──────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

matplotlib.use("Agg")

from src.core.embedding_model import embed_chunks  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "benchmark_dataset.json"


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════


def load_benchmark() -> dict:
    """Load the benchmark dataset from JSON."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  Similarity computation
# ══════════════════════════════════════════════════════════════════════════════


def compute_semantic_similarities(pairs: list) -> np.ndarray:
    """Embed all texts with Sentence Transformers and compute pairwise cosine."""
    texts_a = [p["text_a"] for p in pairs]
    texts_b = [p["text_b"] for p in pairs]

    emb_a = embed_chunks(texts_a)  # (N, 384)
    emb_b = embed_chunks(texts_b)  # (N, 384)

    # Row-wise cosine similarity (not full N×N — just the diagonal pairs)
    similarities = np.array(
        [
            float(sklearn_cosine(emb_a[i : i + 1], emb_b[i : i + 1])[0, 0])
            for i in range(len(pairs))
        ]
    )
    return similarities


def compute_tfidf_similarities(pairs: list) -> np.ndarray:
    """Compute TF-IDF cosine similarity for each pair (lexical baseline)."""
    similarities = []
    for p in pairs:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([p["text_a"], p["text_b"]])
        sim = float(sklearn_cosine(tfidf[0:1], tfidf[1:2])[0, 0])
        similarities.append(sim)
    return np.array(similarities)


def compute_hybrid_similarities(pairs: list, w: float = 0.7) -> np.ndarray:
    """Compute hybrid similarity combining semantic and TF-IDF scores."""
    semantic_sims = compute_semantic_similarities(pairs)
    tfidf_sims = compute_tfidf_similarities(pairs)

    if not (0.0 <= w <= 1.0):
        raise ValueError(f"Weight w must be between 0.0 and 1.0, got {w}")

    hybrid_sims = w * semantic_sims + (1 - w) * tfidf_sims
    return hybrid_sims


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════


def compute_metrics_at_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Compute precision, recall, F1 at a given threshold."""
    predictions = (similarities >= threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "threshold": round(threshold, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def sweep_thresholds(
    similarities: np.ndarray,
    labels: np.ndarray,
    start: float = 0.30,
    stop: float = 0.96,
    step: float = 0.01,
) -> pd.DataFrame:
    """Compute metrics across a range of thresholds."""
    rows = []
    for t in np.arange(start, stop, step):
        rows.append(compute_metrics_at_threshold(similarities, labels, t))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════


def plot_roc_curves(
    labels: np.ndarray,
    semantic_sims: np.ndarray,
    tfidf_sims: np.ndarray,
    save_path: Path,
) -> None:
    """Plot ROC curves for both approaches on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for sims, name, color, ls in [
        (semantic_sims, "Sentence Transformers", "#e63946", "-"),
        (tfidf_sims, "TF-IDF Baseline", "#457b9d", "--"),
    ]:
        fpr, tpr, _ = roc_curve(labels, sims)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.2,
            linestyle=ls,
            label=f"{name}  (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#adb5bd", lw=1, linestyle=":")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curve — Semantic vs Lexical Plagiarism Detection",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(
    labels: np.ndarray,
    semantic_sims: np.ndarray,
    tfidf_sims: np.ndarray,
    save_path: Path,
) -> None:
    """Plot Precision-Recall curves for both approaches."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for sims, name, color, ls in [
        (semantic_sims, "Sentence Transformers", "#e63946", "-"),
        (tfidf_sims, "TF-IDF Baseline", "#457b9d", "--"),
    ]:
        precision, recall, _ = precision_recall_curve(labels, sims)
        pr_auc = auc(recall, precision)
        ax.plot(
            recall,
            precision,
            color=color,
            lw=2.2,
            linestyle=ls,
            label=f"{name}  (AUC = {pr_auc:.3f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curve — Semantic vs Lexical", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_similarity_distributions(
    labels: np.ndarray,
    semantic_sims: np.ndarray,
    tfidf_sims: np.ndarray,
    save_path: Path,
) -> None:
    """Plot histograms showing score separation between plagiarised and original."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, sims, title in [
        (axes[0], semantic_sims, "Sentence Transformers"),
        (axes[1], tfidf_sims, "TF-IDF Baseline"),
    ]:
        plag_scores = sims[labels == 1]
        noplag_scores = sims[labels == 0]

        ax.hist(
            noplag_scores,
            bins=15,
            alpha=0.65,
            color="#2a9d8f",
            label="Not Plagiarized",
            edgecolor="white",
        )
        ax.hist(
            plag_scores,
            bins=15,
            alpha=0.65,
            color="#e63946",
            label="Plagiarized",
            edgecolor="white",
        )
        ax.set_xlabel("Cosine Similarity", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Count", fontsize=11)
    fig.suptitle(
        "Similarity Score Distributions by Label",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ══════════════════════════════════════════════════════════════════════════════


def evaluate():
    """Run the full evaluation pipeline and save results."""

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_benchmark()
    pairs = dataset["pairs"]
    labels = np.array([1 if p["label"] == "plagiarized" else 0 for p in pairs])

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    print()
    print("=" * 72)
    print("  SEMANTIC PLAGIARISM DETECTOR -- EVALUATION REPORT")
    print("=" * 72)
    print(
        f"  Dataset : {len(pairs)} pairs  ({n_pos} plagiarized,  {n_neg} not plagiarized)"
    )
    print("  Model   : all-MiniLM-L6-v2  (384-dim, L2-normalised)")
    print("  Baseline: TF-IDF + cosine similarity")
    print("-" * 72)

    # ── Compute similarities ──────────────────────────────────────────────────
    print("\n  [1/6] Computing semantic similarities (Sentence Transformers)...")
    semantic_sims = compute_semantic_similarities(pairs)

    print("  [2/6] Computing TF-IDF similarities (lexical baseline)...")
    tfidf_sims = compute_tfidf_similarities(pairs)

    print("  [3/6] Computing hybrid similarities (w=0.7)...")
    hybrid_sims = compute_hybrid_similarities(pairs, w=0.7)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print("  [4/6] Sweeping thresholds (0.30 -> 0.95)...")
    sem_sweep = sweep_thresholds(semantic_sims, labels)
    tfidf_sweep = sweep_thresholds(tfidf_sims, labels)
    hybrid_sweep = sweep_thresholds(hybrid_sims, labels)

    # Find optimal threshold (max F1)
    sem_best = sem_sweep.loc[sem_sweep["f1"].idxmax()]
    tfidf_best = tfidf_sweep.loc[tfidf_sweep["f1"].idxmax()]
    hybrid_best = hybrid_sweep.loc[hybrid_sweep["f1"].idxmax()]

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    sem_fpr, sem_tpr, _ = roc_curve(labels, semantic_sims)
    sem_auc = auc(sem_fpr, sem_tpr)

    tfidf_fpr, tfidf_tpr, _ = roc_curve(labels, tfidf_sims)
    tfidf_auc = auc(tfidf_fpr, tfidf_tpr)

    hybrid_fpr, hybrid_tpr, _ = roc_curve(labels, hybrid_sims)
    hybrid_auc = auc(hybrid_fpr, hybrid_tpr)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    header = f"  {'Metric':<28} {'Semantic':>12} {'TF-IDF':>12} {'Hybrid':>12}"
    print(header)
    print("  " + "-" * 68)

    def row(name, sem_val, tfidf_val, hybrid_val, fmt=".4f", pct=False):
        s = f"{sem_val:{fmt}}"
        t = f"{tfidf_val:{fmt}}"
        h = f"{hybrid_val:{fmt}}"
        print(f"  {name:<28} {s:>12} {t:>12} {h:>12}")

    row("ROC-AUC", sem_auc, tfidf_auc, hybrid_auc)
    row(
        "Best F1",
        float(sem_best["f1"]),
        float(tfidf_best["f1"]),
        float(hybrid_best["f1"]),
    )
    row(
        "  @ Threshold",
        float(sem_best["threshold"]),
        float(tfidf_best["threshold"]),
        float(hybrid_best["threshold"]),
    )
    row(
        "  Precision",
        float(sem_best["precision"]),
        float(tfidf_best["precision"]),
        float(hybrid_best["precision"]),
    )
    row(
        "  Recall",
        float(sem_best["recall"]),
        float(tfidf_best["recall"]),
        float(hybrid_best["recall"]),
    )
    row(
        "  Accuracy",
        float(sem_best["accuracy"]),
        float(tfidf_best["accuracy"]),
        float(hybrid_best["accuracy"]),
    )

    print()
    print(f"  Confusion Matrix (Semantic @ threshold={sem_best['threshold']:.2f}):")
    print(
        f"    TP={int(sem_best['tp'])}  FP={int(sem_best['fp'])}  FN={int(sem_best['fn'])}  TN={int(sem_best['tn'])}"
    )
    print()
    print(f"  Confusion Matrix (TF-IDF @ threshold={tfidf_best['threshold']:.2f}):")
    print(
        f"    TP={int(tfidf_best['tp'])}  FP={int(tfidf_best['fp'])}  FN={int(tfidf_best['fn'])}  TN={int(tfidf_best['tn'])}"
    )
    print()
    print(f"  Confusion Matrix (Hybrid @ threshold={hybrid_best['threshold']:.2f}):")
    print(
        f"    TP={int(hybrid_best['tp'])}  FP={int(hybrid_best['fp'])}  FN={int(hybrid_best['fn'])}  TN={int(hybrid_best['tn'])}"
    )

    # ── Per-pair details ──────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  PER-PAIR SCORES")
    print("-" * 72)
    print(
        f"  {'ID':<8} {'Category':<22} {'Label':<16} {'Semantic':>9} {'TF-IDF':>9} {'Hybrid':>9}"
    )
    print("  " + "-" * 70)
    for i, p in enumerate(pairs):
        print(
            f"  {p['id']:<8} {p['category']:<22} {p['label']:<16} "
            f"{semantic_sims[i]:>8.4f}  {tfidf_sims[i]:>8.4f}  {hybrid_sims[i]:>8.4f}"
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n  [5/6] Saving plots to {RESULTS_DIR}/...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(labels, semantic_sims, tfidf_sims, RESULTS_DIR / "roc_curve.png")
    plot_pr_curves(labels, semantic_sims, tfidf_sims, RESULTS_DIR / "pr_curve.png")
    plot_similarity_distributions(
        labels, semantic_sims, tfidf_sims, RESULTS_DIR / "similarity_distribution.png"
    )

    print("  [6/6] Saving metrics...")

    # Threshold sweep CSV
    sem_sweep.to_csv(RESULTS_DIR / "threshold_sweep_semantic.csv", index=False)
    tfidf_sweep.to_csv(RESULTS_DIR / "threshold_sweep_tfidf.csv", index=False)
    hybrid_sweep.to_csv(RESULTS_DIR / "threshold_sweep_hybrid.csv", index=False)

    # Summary metrics JSON
    summary = {
        "dataset_size": len(pairs),
        "n_plagiarized": n_pos,
        "n_not_plagiarized": n_neg,
        "semantic": {
            "model": "all-MiniLM-L6-v2",
            "roc_auc": round(sem_auc, 4),
            "best_threshold": float(sem_best["threshold"]),
            "best_f1": float(sem_best["f1"]),
            "precision": float(sem_best["precision"]),
            "recall": float(sem_best["recall"]),
            "accuracy": float(sem_best["accuracy"]),
        },
        "tfidf_baseline": {
            "roc_auc": round(tfidf_auc, 4),
            "best_threshold": float(tfidf_best["threshold"]),
            "best_f1": float(tfidf_best["f1"]),
            "precision": float(tfidf_best["precision"]),
            "recall": float(tfidf_best["recall"]),
            "accuracy": float(tfidf_best["accuracy"]),
        },
        "hybrid": {
            "weight": 0.7,
            "roc_auc": round(hybrid_auc, 4),
            "best_threshold": float(hybrid_best["threshold"]),
            "best_f1": float(hybrid_best["f1"]),
            "precision": float(hybrid_best["precision"]),
            "recall": float(hybrid_best["recall"]),
            "accuracy": float(hybrid_best["accuracy"]),
        },
        "per_pair": [
            {
                "id": p["id"],
                "category": p["category"],
                "label": p["label"],
                "semantic_score": round(float(semantic_sims[i]), 4),
                "tfidf_score": round(float(tfidf_sims[i]), 4),
                "hybrid_score": round(float(hybrid_sims[i]), 4),
            }
            for i, p in enumerate(pairs)
        ],
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 72)
    print("  [OK] Evaluation complete.")
    print(f"  Results saved to: {RESULTS_DIR.resolve()}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    evaluate()
