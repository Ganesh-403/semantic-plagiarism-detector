"""
Document Fingerprinting & Deduplication Visualizations
=======================================================
Plotly and Matplotlib charts for the deduplication engine.
Provides cluster visualizations, match heatmaps, and summary dashboards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.security.document_fingerprint import (
    DedupReport,
    DuplicateCluster,
    DuplicateMatch,
    MatchType,
)

COLORS = {
    "exact": "#ef4444",
    "near_duplicate": "#f97316",
    "similar": "#eab308",
    "unique": "#22c55e",
    "primary": "#1e3a8a",
    "secondary": "#3b82f6",
    "bg": "#ffffff",
    "grid": "#e5e7eb",
    "text": "#1f2937",
    "text_light": "#6b7280",
}

MATCH_COLORS = {
    MatchType.EXACT: COLORS["exact"],
    MatchType.NEAR_DUPLICATE: COLORS["near_duplicate"],
    MatchType.SIMILAR: COLORS["similar"],
    MatchType.UNIQUE: COLORS["unique"],
}


def create_duplicate_summary_gauge(
    report: DedupReport,
    title: str = "Duplicate Detection Summary",
) -> Optional[Any]:
    """Create a gauge showing unique vs duplicate ratio."""
    if not HAS_PLOTLY:
        return None

    total = report.total_documents
    if total == 0:
        return _empty_figure("No documents to analyze")

    unique_pct = (report.unique_documents / total) * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=unique_pct,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": f"{title}<br><span style='font-size:12px;color:#6b7280'>"
                        f"Unique Documents</span>", "font": {"size": 14}},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=COLORS["unique"]),
            steps=[
                dict(range=[0, 50], color="#fee2e2"),
                dict(range=[50, 80], color="#fef3c7"),
                dict(range=[80, 100], color="#d1fae5"),
            ],
        ),
    ))
    fig.update_layout(height=220, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor=COLORS["bg"])
    return fig


def create_match_type_breakdown(
    report: DedupReport,
    title: str = "Match Type Distribution",
) -> Optional[Any]:
    """Create a donut chart showing the distribution of match types."""
    if not HAS_PLOTLY:
        return None

    labels = []
    values = []
    colors = []

    for mt, label in [
        (MatchType.EXACT, "Exact Duplicates"),
        (MatchType.NEAR_DUPLICATE, "Near Duplicates"),
        (MatchType.SIMILAR, "Similar"),
    ]:
        count = sum(1 for m in report.matches if m.match_type == mt)
        if count > 0:
            labels.append(label)
            values.append(count)
            colors.append(MATCH_COLORS[mt])

    if not values:
        labels = ["No Matches"]
        values = [1]
        colors = ["#d1d5db"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b>{len(report.matches)}</b><br>Total Matches",
        x=0.5, y=0.5, font=dict(size=14, color=COLORS["text"]), showarrow=False,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        height=340, margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True, paper_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


def create_cluster_size_chart(
    report: DedupReport,
    top_n: int = 15,
    title: str = "Duplicate Cluster Sizes",
) -> Optional[Any]:
    """Create a bar chart of cluster sizes."""
    if not HAS_PLOTLY:
        return None
    if not report.clusters:
        return _empty_figure("No duplicate clusters found")

    top = sorted(report.clusters, key=lambda c: c.cluster_size, reverse=True)[:top_n]
    labels = [f"Cluster {c.cluster_id}" for c in reversed(top)]
    sizes = [c.cluster_size for c in reversed(top)]
    types = [c.match_type.value for c in reversed(top)]
    bar_colors = [MATCH_COLORS.get(c.match_type, COLORS["text_light"]) for c in reversed(top)]

    fig = go.Figure(go.Bar(
        y=labels, x=sizes, orientation="h",
        marker_color=bar_colors,
        hovertemplate="<b>%{y}</b><br>Size: %{x}<br>Type: %{customdata}<extra></extra>",
        customdata=types,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis_title="Number of Documents", yaxis_title="",
        height=max(300, len(top) * 30 + 80),
        margin=dict(l=100, r=20, t=50, b=40),
        template="plotly_white", paper_bgcolor=COLORS["bg"],
    )
    return fig


def create_similarity_histogram(
    report: DedupReport,
    title: str = "Overall Similarity Score Distribution",
) -> Optional[Any]:
    """Create a histogram of overall match scores."""
    if not HAS_PLOTLY:
        return None
    if not report.matches:
        return _empty_figure("No matches to display")

    scores = [m.overall_score for m in report.matches]

    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=30,
        marker_color=COLORS["secondary"],
        hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(
        x=0.85, line_dash="dot", line_color=COLORS["exact"],
        annotation_text="Near-Dup Threshold",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        xaxis_title="Similarity Score", yaxis_title="Match Count",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white", paper_bgcolor=COLORS["bg"],
    )
    return fig


def create_scanning_metrics_table(report: DedupReport) -> Optional[Any]:
    """Create a Plotly table of key scanning metrics."""
    if not HAS_PLOTLY:
        return None

    headers = ["Metric", "Value"]
    cells = [
        ["Total Documents", "Unique Documents", "Exact Duplicates",
         "Near Duplicates", "Clusters Found", "Scan Duration"],
        [
            str(report.total_documents),
            str(report.unique_documents),
            str(report.exact_duplicate_count),
            str(report.near_duplicate_count),
            str(len(report.clusters)),
            f"{report.scan_duration_ms:.1f} ms",
        ],
    ]

    fig = go.Figure(go.Table(
        header=dict(values=headers, fill_color=COLORS["primary"],
                     font=dict(color="white", size=13), align="left"),
        cells=dict(values=cells, fill_color=[COLORS["bg"], COLORS["bg"]],
                    font=dict(size=12), align="left", height=28),
    ))

    fig.update_layout(
        title=dict(text="Scanning Summary", font=dict(size=16, color=COLORS["text"])),
        height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor=COLORS["bg"],
    )
    return fig


def create_match_heatmap(
    report: DedupReport,
    max_docs: int = 30,
    title: str = "Document Similarity Heatmap",
) -> Optional[Any]:
    """Create a heatmap of pairwise document similarities."""
    if not HAS_PLOTLY:
        return None
    if not report.matches:
        return _empty_figure("No matches for heatmap")

    # Collect unique document IDs from matches
    doc_ids: List[str] = []
    seen = set()
    for m in report.matches:
        if m.source_id not in seen:
            doc_ids.append(m.source_id)
            seen.add(m.source_id)
        if m.target_id not in seen:
            doc_ids.append(m.target_id)
            seen.add(m.target_id)
        if len(doc_ids) >= max_docs:
            break

    n = len(doc_ids)
    if n < 2:
        return _empty_figure("Not enough documents for heatmap")

    idx_map = {did: i for i, did in enumerate(doc_ids)}
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)

    for m in report.matches:
        if m.source_id in idx_map and m.target_id in idx_map:
            i, j = idx_map[m.source_id], idx_map[m.target_id]
            matrix[i, j] = m.overall_score
            matrix[j, i] = m.overall_score

    labels = [did[:25] for did in doc_ids]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale="RdYlBu_r", zmin=0, zmax=1,
        hovertemplate="Doc A: %{y}<br>Doc B: %{x}<br>Score: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"])),
        height=max(400, n * 22 + 120),
        margin=dict(l=150, r=20, t=50, b=100),
        paper_bgcolor=COLORS["bg"],
        xaxis=dict(tickangle=45),
    )
    return fig


def create_static_cluster_bar(
    report: DedupReport,
    output_path: str = "dedup_clusters.png",
    dpi: int = 150,
) -> Optional[str]:
    """Create a static bar chart of cluster sizes using matplotlib."""
    if not HAS_MATPLOTLIB or not report.clusters:
        return None

    top = sorted(report.clusters, key=lambda c: c.cluster_size, reverse=True)[:15]
    labels = [f"C{c.cluster_id}" for c in top]
    sizes = [c.cluster_size for c in top]
    bar_colors = [MATCH_COLORS.get(c.match_type, COLORS["text_light"]) for c in top]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    bars = ax.bar(labels, sizes, color=bar_colors, edgecolor="white")

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                str(size), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Document Count", fontsize=11)
    ax.set_title("Duplicate Cluster Sizes", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def create_static_similarity_histogram(
    report: DedupReport,
    output_path: str = "dedup_scores.png",
    dpi: int = 150,
) -> Optional[str]:
    """Create a static histogram of match similarity scores."""
    if not HAS_MATPLOTLIB or not report.matches:
        return None

    scores = [m.overall_score for m in report.matches]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=30, color=COLORS["secondary"], edgecolor="white", alpha=0.85)
    ax.axvline(x=0.85, color=COLORS["exact"], linestyle="--", linewidth=1.5,
               label="Near-Dup Threshold")
    ax.set_xlabel("Similarity Score", fontsize=11)
    ax.set_ylabel("Match Count", fontsize=11)
    ax.set_title("Similarity Score Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_dedup_summary_metrics(report: DedupReport) -> Dict[str, Any]:
    """Extract key metrics for Streamlit metric cards."""
    total = report.total_documents
    unique_pct = round((report.unique_documents / total) * 100, 1) if total > 0 else 0.0
    dup_pct = round(100.0 - unique_pct, 1)
    return {
        "total_documents": total,
        "unique_documents": report.unique_documents,
        "unique_pct": unique_pct,
        "duplicate_pct": dup_pct,
        "exact_duplicates": report.exact_duplicate_count,
        "near_duplicates": report.near_duplicate_count,
        "clusters": len(report.clusters),
        "scan_time_ms": report.scan_duration_ms,
        "avg_cluster_size": round(
            sum(c.cluster_size for c in report.clusters) / len(report.clusters), 1
        ) if report.clusters else 0,
    }


def _empty_figure(message: str) -> Optional[Any]:
    """Create a minimal Plotly figure with an info message."""
    if not HAS_PLOTLY:
        return None
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color=COLORS["text_light"]),
    )
    fig.update_layout(
        height=300, xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor=COLORS["bg"],
    )
    return fig
