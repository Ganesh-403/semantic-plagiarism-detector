"""
heatmap.py
----------
Generates similarity heatmaps for Semantic Plagiarism Detector.

This module provides high-quality, customizable heatmap visualizations for
document similarity matrices. It bridges the gap between backend scoring
and frontend rendering, offering both static (Matplotlib/Seaborn) and
interactive (Plotly) options.

Recent additions (Issue #628 & Issue #839):
- Added `log_scale` parameter to `plot_similarity_heatmap` and `render_heatmap_ui`.
- Implemented Matplotlib `LogNorm` for better visualization of highly skewed similarity distributions.
- Added shape guards in heatmap functions to handle single document (1x1) input gracefully without collapse (#839).
"""

import logging

logger = logging.getLogger(__name__)
import io
import re
from contextlib import contextmanager
from typing import Dict, Generator, Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

# Enforce non-interactive backend for standard plot generation to prevent thread-safety
# issues in web environments like Streamlit.
matplotlib.use("Agg")

try:
    from src.core.similarity import PLAGIARISM_THRESHOLD
except ImportError:
    # Fallback for standalone testing or isolated environments
    PLAGIARISM_THRESHOLD = 0.75


# ── Colormap Mappings & Constants ──────────────────────────────────────────────
try:
    from app.theme import (
        DEFAULT_UI_COLORMAP,
        MATPLOTLIB_CMAP_MAPPING,
        PLOTLY_CMAP_MAPPING,
        UI_COLORMAP_OPTIONS,
        apply_matplotlib_theme,
    )
except ImportError:
    UI_COLORMAP_OPTIONS = [
        "Viridis",
        "Cividis",
        "Plasma",
        "Coolwarm",
        "YlOrRd",
    ]

    MATPLOTLIB_CMAP_MAPPING = {
        "Viridis": "viridis",
        "Cividis": "cividis",
        "Plasma": "plasma",
        "Coolwarm": "coolwarm",
        "YlOrRd": "YlOrRd",
    }

    PLOTLY_CMAP_MAPPING = {
        "Viridis": "Viridis",
        "Cividis": "Cividis",
        "Plasma": "Plasma",
        "Coolwarm": "RdBu_r",
        "YlOrRd": "YlOrRd",
    }

    DEFAULT_UI_COLORMAP = "Viridis"

    def apply_matplotlib_theme(theme_colors=None):
        return None

    # Default Plotly font family
    DEFAULT_FONT_FAMILY: str = "Inter, sans-serif"


# ── Security & Sanitization ────────────────────────────────────────────────────
class MatplotlibInjectionError(ValueError):
    """Raised when a string contains forbidden formatting or injection tokens."""

    pass


class TitleSanitizer:
    """Sanitizes user-provided titles and labels to prevent injection exploits."""

    MATHTEXT_PATTERN = re.compile(r"[\$\_\^\{\}]")
    HTML_TAG_PATTERN = re.compile(r"<[^>]*?>")

    @classmethod
    def sanitize(cls, text: Optional[str], strict: bool = False) -> str:
        if not text:
            return ""
        clean_text = cls.HTML_TAG_PATTERN.sub("", str(text))
        if strict and cls.MATHTEXT_PATTERN.search(clean_text):
            logger.error("Potential Matplotlib text injection detected.")
            raise MatplotlibInjectionError(
                "Provided string contains unauthorized formatting characters."
            )
        clean_text = clean_text.replace("\n", " ").replace("\r", " ")
        return clean_text.strip()


# ── Data Validation Helpers ────────────────────────────────────────────────────
def validate_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Validates and cleans the input similarity matrix before visualization."""
    if df.empty:
        logger.warning("Empty DataFrame provided to heatmap generator.")
        return df

    rows, cols = df.shape
    if rows != cols:
        logger.error(f"Similarity matrix must be square. Received {rows}x{cols}.")
        raise ValueError("Similarity matrix is not square.")

    clean_df = df.copy()
    if clean_df.isnull().values.any():
        logger.info("NaN values detected in similarity matrix. Filling with 0.0.")
        clean_df = clean_df.fillna(0.0)

    clean_df = clean_df.clip(lower=0.0, upper=1.0)
    arr = clean_df.to_numpy(copy=True)
    np.fill_diagonal(arr, 1.0)
    clean_df = pd.DataFrame(arr, index=df.index, columns=df.columns)
    return clean_df


def export_heatmap_matrix_csv(df: pd.DataFrame) -> bytes:
    """Export a similarity matrix DataFrame as UTF-8 encoded CSV bytes."""
    buf = io.BytesIO()
    df.to_csv(buf, encoding="utf-8", index=True)
    return buf.getvalue()


def filter_heatmap_by_class_tag(
    similarity_df: pd.DataFrame,
    class_tag: Optional[str] = None,
    doc_class_map: Optional[dict] = None,
) -> pd.DataFrame:
    """Filter heatmap matrix rows and columns by matching document class section tags."""
    if similarity_df.empty or not class_tag or class_tag == "All Classes":
        return similarity_df

    if doc_class_map is None:
        try:
            from src.db.corpus_db import get_all_documents

            all_docs = get_all_documents(include_deleted=True)
            doc_class_map = {}
            for d in all_docs:
                fname = (
                    d.get("filename")
                    if isinstance(d, dict)
                    else getattr(d, "filename", None)
                )
                csec = (
                    d.get("class_section")
                    if isinstance(d, dict)
                    else getattr(d, "class_section", None)
                )
                if fname:
                    doc_class_map[fname] = csec
        except Exception as e:
            logger.warning(f"Could not load document class map from database: {e}")
            doc_class_map = {}

    matching_cols = [
        col for col in similarity_df.columns if doc_class_map.get(str(col)) == class_tag
    ]

    if not matching_cols:
        logger.info(f"No document cells match class tag '{class_tag}'.")
        return pd.DataFrame()

    return similarity_df.loc[matching_cols, matching_cols]


def _get_theme_color(theme_colors: Optional[dict], key: str, fallback: str) -> str:
    """Safely retrieves a color from a theme dictionary with a fallback."""
    if not theme_colors:
        return fallback
    return theme_colors.get(key, fallback)


# ── Static Visualization (Matplotlib/Seaborn) ──────────────────────────────────
@contextmanager
def matplotlib_figure(*args, **kwargs) -> Generator[tuple, None, None]:
    """Context manager that yields (fig, ax) and guarantees plt.close(fig)."""
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


def plot_similarity_heatmap(
    similarity_df: pd.DataFrame,
    title: str = "Semantic Similarity Matrix",
    threshold: float = PLAGIARISM_THRESHOLD,
    figsize: Optional[tuple] = None,
    show_annotations: bool = True,
    dpi: int = 150,
    theme_colors: Optional[dict[str, str]] = None,
    colormap_name: str = DEFAULT_UI_COLORMAP,
    mask_threshold: Optional[float] = None,
    log_scale: bool = False,
    class_tag: Optional[str] = None,
    doc_class_map: Optional[dict] = None,
    dim_diagonal: bool = False,
) -> Figure:
    """High-resolution Matplotlib heatmap optimized for static PNG export."""
    if class_tag and class_tag != "All Classes":
        similarity_df = filter_heatmap_by_class_tag(
            similarity_df, class_tag=class_tag, doc_class_map=doc_class_map
        )

    try:
        safe_title = TitleSanitizer.sanitize(title)
    except MatplotlibInjectionError:
        safe_title = "Semantic Similarity Matrix (Sanitized)"

    cmap = MATPLOTLIB_CMAP_MAPPING.get(colormap_name, "viridis")

    try:
        clean_df = validate_similarity_matrix(similarity_df)
    except ValueError as ve:
        logger.error(f"Validation failed: {ve}")
        clean_df = similarity_df

    n = len(clean_df)

    # Issue #839: Handle empty or single document (< 2) input cleanly
    if n < 2:
        with matplotlib_figure(figsize=figsize or (6, 4), dpi=dpi) as (fig, ax):
            ax.set_title(safe_title, fontsize=12, fontweight="bold", pad=12)
            ax.text(
                0.5,
                0.5,
                "At least 2 documents are required to build a pairwise heatmap",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#666666",
                bbox=dict(
                    boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc"
                ),
            )
            ax.axis("off")
            fig.tight_layout()
            return fig

    if dim_diagonal and n > 0:
        clean_df = clean_df.copy()
        vals = clean_df.to_numpy(copy=True)
        np.fill_diagonal(vals, np.nan)
        clean_df = pd.DataFrame(vals, index=clean_df.index, columns=clean_df.columns)

    if figsize is None:
        cell_size = max(1.2, 6 / n)
        width = max(6.0, n * cell_size + 2.0)
        height = max(5.0, n * cell_size + 1.5)
        figsize = (width, height)

    mask = None
    if mask_threshold is not None:
        mask = similarity_df < mask_threshold
    if dim_diagonal and n > 0:
        diag_mask = np.eye(n, dtype=bool)
        mask = diag_mask if mask is None else (mask | diag_mask)

    norm = None
    if log_scale:
        norm = mcolors.LogNorm(vmin=1e-3, vmax=1.0)
        logger.info("Applied logarithmic color scaling to heatmap.")
    apply_matplotlib_theme(theme_colors)

    with matplotlib_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        sns.heatmap(
            clean_df,
            ax=ax,
            annot=show_annotations,
            fmt=".2f" if show_annotations else "",
            cmap=cmap,
            vmin=0.0 if not log_scale else None,
            vmax=1.0,
            norm=norm,
            linewidths=0.6,
            linecolor="#cccccc",
            square=True,
            mask=mask,
            cbar_kws={"label": "Cosine Similarity", "shrink": 0.8, "pad": 0.02},
            annot_kws={"size": max(7, 14 - n), "weight": "bold"},
        )

        colorbar = ax.collections[0].colorbar
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

        if theme_colors:
            fig.patch.set_facecolor(theme_colors.get("background", "#FFFFFF"))
            ax.set_facecolor(theme_colors.get("surface", "#F8FAFC"))
            ax.tick_params(colors=theme_colors.get("ink", "#0F172A"))
            ax.xaxis.label.set_color(theme_colors.get("ink", "#0F172A"))
            ax.yaxis.label.set_color(theme_colors.get("ink", "#0F172A"))
            title_color = theme_colors.get("ink", "#0F172A")
        else:
            title_color = "black"

        if dim_diagonal:
            dim_color = (
                theme_colors.get("border", "#cccccc") if theme_colors else "#cccccc"
            )
            for i in range(n):
                ax.add_patch(
                    mpatches.Rectangle(
                        (i, i),
                        1,
                        1,
                        facecolor=dim_color,
                        alpha=0.4,
                        zorder=2,
                    )
                )

        data = clean_df.values
        for i in range(n):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (i, i),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    linewidth=2,
                    edgecolor="#777777" if dim_diagonal else "#555555",
                    facecolor="none",
                    zorder=3,
                )
            )

        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(data[i, j]) and data[i, j] >= threshold:
                    ax.add_patch(
                        mpatches.FancyBboxPatch(
                            (j, i),
                            1,
                            1,
                            boxstyle="square,pad=0",
                            linewidth=2.5,
                            edgecolor="#d62728",
                            facecolor="none",
                            zorder=4,
                        )
                    )

        ax.set_title(
            safe_title, fontsize=15, fontweight="bold", pad=16, color=title_color
        )
        ax.set_xlabel("Documents", fontsize=11, labelpad=10)
        ax.set_ylabel("Documents", fontsize=11, labelpad=10)

        safe_labels = [TitleSanitizer.sanitize(str(lbl)) for lbl in clean_df.columns]
        tick_fontsize = max(6, 12 - n // 10)
        ax.set_xticklabels(safe_labels, rotation=30, ha="right", fontsize=tick_fontsize)
        ax.set_yticklabels(safe_labels, rotation=0, fontsize=tick_fontsize)

        red_patch = mpatches.Patch(
            edgecolor="#d62728",
            facecolor="none",
            linewidth=2,
            label=f"Potential Plagiarism (≥ {threshold:.0%})",
        )
        ax.legend(
            handles=[red_patch],
            loc="upper left",
            bbox_to_anchor=(0.0, -0.18),
            frameon=True,
            fontsize=9,
        )

        if theme_colors:
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_color(theme_colors.get("ink", "#0F172A"))
                legend.get_frame().set_facecolor(
                    theme_colors.get("background", "#FFFFFF")
                )
                legend.get_frame().set_edgecolor(theme_colors.get("border", "#E2E8F0"))


# ── Interactive Visualization (Plotly) ─────────────────────────────────────────
def plot_similarity_heatmap_plotly(
    similarity_df: pd.DataFrame,
    title: str = "Semantic Similarity Matrix",
    threshold: float = PLAGIARISM_THRESHOLD,
    theme_colors: Optional[dict[str, str]] = None,
    colorscale: str = "Viridis",
    show_annotations: bool = True,
    mask_threshold: Optional[float] = None,
    log_scale: bool = False,
    class_tag: Optional[str] = None,
    doc_class_map: Optional[dict] = None,
    dim_diagonal: bool = False,
    zmin: float = 0.0,
    zmax: float = 1.0,
):
    """Interactive Plotly heatmap featuring dynamic hover values and custom threshold bounds."""
    import plotly.graph_objects as go

    if similarity_df.empty or len(similarity_df) == 0:
        fig = go.Figure()
        try:
            safe_title = TitleSanitizer.sanitize(title)
        except Exception:
            safe_title = "Semantic Similarity Matrix"
        fig.update_layout(
            title=safe_title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.add_annotation(
            text="No document data available for heatmap visualization",
            showarrow=False,
            font=dict(size=14, color="#666666"),
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=10,
            bgcolor="#f8f9fa",
        )
        return fig

    if class_tag and class_tag != "All Classes":
        similarity_df = filter_heatmap_by_class_tag(
            similarity_df, class_tag=class_tag, doc_class_map=doc_class_map
        )

    try:
        safe_title = TitleSanitizer.sanitize(title)
    except MatplotlibInjectionError:
        safe_title = "Semantic Similarity Matrix"

    cmap = PLOTLY_CMAP_MAPPING.get(colormap_name, "Viridis")  # noqa: F821

    try:
        clean_df = validate_similarity_matrix(similarity_df)
    except ValueError as error:
        logger.error(error)
        return go.Figure()

    # Issue #839: Handle empty or single document (< 2) input cleanly
    if clean_df.empty or len(clean_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=safe_title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.add_annotation(
            text="No document data available for heatmap visualization",
            showarrow=False,
            font=dict(size=14, color="#666666"),
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=10,
            bgcolor="#f8f9fa",
        )
        return fig
    elif len(clean_df) < 2:
        fig = go.Figure()
        fig.update_layout(
            title=safe_title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.add_annotation(
            text="At least 2 documents are required to build a pairwise heatmap",
            showarrow=False,
            font=dict(size=14, color="#666666"),
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=10,
            bgcolor="#f8f9fa",
        )
        return fig

    names = [TitleSanitizer.sanitize(str(col)) for col in clean_df.columns]
    z_matrix = clean_df.values.tolist()

    if mask_threshold is not None:
        z_matrix = [
            [val if val >= mask_threshold else None for val in row]
            for row in clean_df.values.tolist()
        ]

    if dim_diagonal:
        z_matrix = [
            [None if i == j else val for j, val in enumerate(row)]
            for i, row in enumerate(z_matrix)
        ]

    n = len(names)
    if n > 15:
        show_annotations = False

    hover_text = [
        [
            f"<b>{names[i]}</b> vs <b>{names[j]}</b><br>"
            + (
                "Self-Similarity: Dimmed"
                if (dim_diagonal and i == j)
                else f"Similarity: {clean_df.values[i, j]:.2%}<br>"
                f"Status: {'Flagged' if (i != j and clean_df.values[i, j] >= threshold) else 'Normal'}"
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=names,
            y=names,
            text=hover_text,
            hovertemplate="%{text}",
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Cosine Similarity", thickness=15, tickformat=".0%"),
            xgap=2,
            ygap=2,
        )
    )

    annotations = []
    if show_annotations:
        for i in range(n):
            for j in range(n):
                if dim_diagonal and i == j:
                    continue
                val = clean_df.values[i, j]
                if pd.isna(val) or (
                    mask_threshold is not None and val < mask_threshold
                ):
                    continue
                font_color = (
                    "black"
                    if (0.3 < val < 0.8 and cmap not in ["Viridis", "Plasma"])
                    else "white"
                )
                if cmap == "YlOrRd" and val < 0.6:
                    font_color = "black"

                annotations.append(
                    dict(
                        x=names[j],
                        y=names[i],
                        text=f"{val:.2f}",
                        showarrow=False,
                        font=dict(
                            size=max(9, 14 - n),
                            color=font_color,
                            family=DEFAULT_FONT_FAMILY,
                        ),
                    )
                )

    shapes = []
    for i in range(n):
        for j in range(n):
            if i != j and clean_df.values[i, j] >= threshold:
                if (
                    mask_threshold is not None
                    and clean_df.values[i, j] < mask_threshold
                ):
                    continue
                shapes.append(
                    dict(
                        type="rect",
                        x0=j - 0.5,
                        x1=j + 0.5,
                        y0=i - 0.5,
                        y1=i + 0.5,
                        line=dict(color="#d62728", width=3),
                        fillcolor="rgba(0,0,0,0)",
                    )
                )

    cell_px = max(80, 600 // n)
    bg_color = _get_theme_color(theme_colors, "background", "rgba(0,0,0,0)")
    ink_color = _get_theme_color(theme_colors, "ink", "#0F172A")

    fig.update_layout(
        title=dict(
            text=safe_title,
            font=dict(size=18, family=DEFAULT_FONT_FAMILY, color=ink_color),
        ),
        height=max(500, n * cell_px + 150),
        autosize=True,
        xaxis=dict(
            side="bottom",
            tickangle=-30,
            title="Document ID",
            color=ink_color,
            fixedrange=False,
        ),
        yaxis=dict(
            autorange="reversed", title="Document ID", color=ink_color, fixedrange=False
        ),
        annotations=annotations,
        shapes=shapes,
        margin=dict(l=140, r=60, t=70, b=140),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=ink_color),
        hoverlabel=dict(
            bgcolor=_get_theme_color(theme_colors, "surface", "white"),
            font_size=14,
            font_family=DEFAULT_FONT_FAMILY,
        ),
    )

    return fig


def plot_document_similarity_heatmap(
    similarity_df: pd.DataFrame,
    title: str = "Semantic Similarity Matrix",
    threshold: float = PLAGIARISM_THRESHOLD,
    theme_colors: Optional[dict[str, str]] = None,
    colormap_name: str = DEFAULT_UI_COLORMAP,
    colorscale: str = "Viridis",
    show_annotations: bool = True,
    mask_threshold: Optional[float] = None,
    log_scale: bool = False,
    class_tag: Optional[str] = None,
    doc_class_map: Optional[dict] = None,
    dim_diagonal: bool = False,
):
    """Wrapper function for plot_similarity_heatmap_plotly with empty state handling."""
    return plot_similarity_heatmap_plotly(
        similarity_df=similarity_df,
        title=title,
        threshold=threshold,
        theme_colors=theme_colors,
        colormap_name=colormap_name,
        colorscale=colorscale,
        show_annotations=show_annotations,
        mask_threshold=mask_threshold,
        log_scale=log_scale,
        class_tag=class_tag,
        doc_class_map=doc_class_map,
        dim_diagonal=dim_diagonal,
    )


def plot_similarity_minimap(
    similarity_df: pd.DataFrame,
    colormap_name: str = DEFAULT_UI_COLORMAP,
):
    import plotly.graph_objects as go

    clean_df = validate_similarity_matrix(similarity_df)

    fig = go.Figure(
        data=go.Heatmap(
            z=clean_df.values,
            colorscale=PLOTLY_CMAP_MAPPING.get(colormap_name, "Viridis"),
            showscale=False,
            hoverinfo="skip",
            xgap=0,
            ygap=0,
        )
    )

    fig.update_layout(
        title="Minimap",
        height=220,
        width=220,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(showticklabels=False, fixedrange=True),
        yaxis=dict(showticklabels=False, fixedrange=True, autorange="reversed"),
    )

    return fig


# ── Differential / Delta Heatmap Visualization (#1369) ─────────────────────────


def plot_differential_heatmap(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
    title: str = "Similarity Matrix Delta (Algorithm A - Algorithm B)",
    label_a: str = "Algorithm A",
    label_b: str = "Algorithm B",
    theme_colors: Optional[dict[str, str]] = None,
    colorscale: str = "RdBu",
    show_annotations: bool = True,
    class_tag: Optional[str] = None,
    doc_class_map: Optional[dict] = None,
):
    """Render a differential (delta) Plotly heatmap comparing score variance between two algorithms.

    Parameters
    ----------
    matrix_a : pd.DataFrame
        First square similarity matrix.
    matrix_b : pd.DataFrame
        Second square similarity matrix.
    title : str, default="Similarity Matrix Delta (Algorithm A - Algorithm B)"
        Plot title text.
    label_a : str, default="Algorithm A"
        Label for the first algorithm or model.
    label_b : str, default="Algorithm B"
        Label for the second algorithm or model.
    theme_colors : Optional[Dict[str, str]], default=None
        Theme palette dictionary.
    colorscale : str, default="RdBu"
        Diverging color scale name for Plotly (e.g. 'RdBu', 'Coolwarm', 'PuOr').
    show_annotations : bool, default=True
        Whether to display signed numeric delta values inside cells.
    class_tag : Optional[str], default=None
        Class section tag filter string.
    doc_class_map : Optional[dict], default=None
        Document to class section mapping.

    Returns
    -------
    go.Figure
        Plotly Figure showing the diverging differential heatmap matrix.
    """
    import plotly.graph_objects as go

    if class_tag and class_tag != "All Classes":
        matrix_a = filter_heatmap_by_class_tag(
            matrix_a, class_tag=class_tag, doc_class_map=doc_class_map
        )
        matrix_b = filter_heatmap_by_class_tag(
            matrix_b, class_tag=class_tag, doc_class_map=doc_class_map
        )

    try:
        safe_title = TitleSanitizer.sanitize(title)
    except MatplotlibInjectionError:
        safe_title = "Similarity Matrix Delta"

    bg_color = _get_theme_color(theme_colors, "background", "#FFFFFF")
    ink_color = _get_theme_color(theme_colors, "ink", "#0F172A")

    if matrix_a is None or matrix_b is None or matrix_a.empty or matrix_b.empty:
        fig = go.Figure()
        fig.update_layout(
            title=safe_title,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=ink_color),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.add_annotation(
            text="Similarity matrices are empty or could not be loaded",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color=ink_color),
        )
        return fig

    # Align matrix_a and matrix_b on common document names
    common_docs = [doc for doc in matrix_a.index if doc in matrix_b.index]
    if len(common_docs) < 2:
        if len(matrix_a.index) >= 2 and len(matrix_a.index) == len(matrix_b.index):
            common_docs = list(matrix_a.index)
        else:
            fig = go.Figure()
            fig.update_layout(
                title=safe_title,
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=ink_color),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            fig.add_annotation(
                text="At least 2 matching document pairs are required for differential heatmap",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color=ink_color),
            )
            return fig

    aligned_a = matrix_a.loc[common_docs, common_docs].fillna(0.0)
    aligned_b = matrix_b.loc[common_docs, common_docs].fillna(0.0)

    # Compute delta matrix: (matrix_a - matrix_b)
    delta_df = aligned_a - aligned_b
    delta_matrix = delta_df.values

    # Determine symmetric color scale bounds around 0
    max_abs_delta = (
        float(np.max(np.abs(delta_matrix))) if delta_matrix.size > 0 else 1.0
    )
    if max_abs_delta < 1e-4:
        max_abs_delta = 1.0

    n = len(common_docs)
    hover_text = []
    cell_text = []

    for i, doc_y in enumerate(common_docs):
        hover_row = []
        text_row = []
        for j, doc_x in enumerate(common_docs):
            val_a = float(aligned_a.iloc[i, j])
            val_b = float(aligned_b.iloc[i, j])
            delta = float(delta_matrix[i, j])

            sign_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            hover_row.append(
                f"<b>Pair:</b> {doc_y} ↔ {doc_x}<br>"
                f"<b>{label_a}:</b> {val_a:.2f}<br>"
                f"<b>{label_b}:</b> {val_b:.2f}<br>"
                f"<b>Delta ({label_a} - {label_b}):</b> {sign_str}"
            )
            text_row.append(sign_str)
        hover_text.append(hover_row)
        cell_text.append(text_row)

    valid_colorscales = {
        "RdBu": "RdBu",
        "Coolwarm": "RdBu_r",
        "Diverging": "RdBu_r",
        "Spectral": "Spectral",
        "PuOr": "PuOr",
        "PiYG": "PiYG",
        "PRGn": "PRGn",
    }
    plotly_colorscale = valid_colorscales.get(colorscale, "RdBu")

    heatmap_trace = go.Heatmap(
        z=delta_matrix,
        x=common_docs,
        y=common_docs,
        hoverinfo="text",
        hovertext=hover_text,
        text=cell_text,
        colorscale=plotly_colorscale,
        zmin=-max_abs_delta,
        zmax=max_abs_delta,
        zmid=0.0,
        colorbar=dict(
            title=dict(text=f"Delta<br>({label_a} - {label_b})", font=dict(size=12)),
            ticks="outside",
            tickformat="+.2f",
        ),
        texttemplate="%{text}" if (show_annotations and n <= 20) else None,
        textfont=dict(size=max(8, 12 - n // 2), color=ink_color),
    )

    fig = go.Figure(data=[heatmap_trace])
    fig.update_layout(
        title=dict(
            text=safe_title,
            font=dict(size=16, family=DEFAULT_FONT_FAMILY, color=ink_color),
        ),
        xaxis=dict(
            title="Documents",
            tickangle=-30,
            tickfont=dict(size=max(8, 11 - n // 3), color=ink_color),
        ),
        yaxis=dict(
            title="Documents",
            autorange="reversed",
            tickfont=dict(size=max(8, 11 - n // 3), color=ink_color),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=ink_color),
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig


def plot_differential_heatmap_matplotlib(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
    title: str = "Similarity Matrix Delta (Algorithm A - Algorithm B)",
    label_a: str = "Algorithm A",
    label_b: str = "Algorithm B",
    figsize: Optional[tuple] = None,
    dpi: int = 150,
    theme_colors: Optional[dict[str, str]] = None,
    colormap_name: str = "coolwarm",
) -> Figure:
    """Render a static Matplotlib differential heatmap comparing score variance between two algorithms."""
    try:
        safe_title = TitleSanitizer.sanitize(title)
    except MatplotlibInjectionError:
        safe_title = "Similarity Matrix Delta"

    if matrix_a.empty or matrix_b.empty or len(matrix_a) < 2:
        with matplotlib_figure(figsize=figsize or (6, 4), dpi=dpi) as (fig, ax):
            ax.set_title(safe_title, fontsize=12, fontweight="bold", pad=12)
            ax.text(
                0.5,
                0.5,
                "At least 2 matching documents are required for differential heatmap",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#666666",
            )
            ax.axis("off")
            fig.tight_layout()
            return fig

    common_docs = [doc for doc in matrix_a.index if doc in matrix_b.index]
    if len(common_docs) < 2:
        common_docs = list(matrix_a.index)

    aligned_a = matrix_a.loc[common_docs, common_docs].fillna(0.0)
    aligned_b = matrix_b.loc[common_docs, common_docs].fillna(0.0)
    delta_df = aligned_a - aligned_b

    n = len(common_docs)
    if figsize is None:
        cell_size = max(1.2, 6 / n)
        figsize = (max(6.0, n * cell_size + 2.0), max(5.0, n * cell_size + 1.5))

    max_abs_delta = float(np.max(np.abs(delta_df.values))) if delta_df.size > 0 else 1.0
    if max_abs_delta < 1e-4:
        max_abs_delta = 1.0

    apply_matplotlib_theme(theme_colors)
    with matplotlib_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        sns.heatmap(
            delta_df,
            ax=ax,
            annot=True,
            fmt="+.2f",
            cmap=colormap_name,
            vmin=-max_abs_delta,
            vmax=max_abs_delta,
            center=0.0,
            linewidths=0.6,
            linecolor="#cccccc",
            square=True,
            cbar_kws={"label": f"Delta ({label_a} - {label_b})", "shrink": 0.8},
        )
        ax.set_title(safe_title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel("Documents", fontsize=11)
        ax.set_ylabel("Documents", fontsize=11)
        fig.tight_layout()
        return fig


# ── Granular Analysis (Chunk-Level Heatmap) ────────────────────────────────────


def plot_chunk_similarity_comparison(
    doc_a_name: str,
    doc_b_name: str,
    chunks_a: list,
    chunks_b: list,
    sim_matrix: np.ndarray,
    theme_colors: Optional[dict] = None,
    colormap_name: str = DEFAULT_UI_COLORMAP,
    show_annotations: bool = True,
) -> Figure:
    """Renders a granular, chunk-level similarity heatmap between two specific documents."""
    try:
        safe_doc_a = TitleSanitizer.sanitize(doc_a_name)
        safe_doc_b = TitleSanitizer.sanitize(doc_b_name)
    except MatplotlibInjectionError:
        safe_doc_a, safe_doc_b = "Doc A", "Doc B"

    cmap = MATPLOTLIB_CMAP_MAPPING.get(colormap_name, "viridis")

    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    na, nb = sim_matrix.shape

    def short_label(text, max_chars=40):
        clean_text = " ".join(str(text).split())
        return TitleSanitizer.sanitize(
            clean_text[:max_chars].strip() + "…"
            if len(clean_text) > max_chars
            else clean_text
        )

    row_labels = [f"A{i + 1}: {short_label(c)}" for i, c in enumerate(chunks_a)]
    col_labels = [f"B{j + 1}: {short_label(c)}" for j, c in enumerate(chunks_b)]

    apply_matplotlib_theme(theme_colors)

    with matplotlib_figure(figsize=(max(8, nb * 1.5), max(6, na * 0.8)), dpi=150) as (
        fig,
        ax,
    ):
        sns.heatmap(
            sim_matrix,
            ax=ax,
            annot=show_annotations,
            fmt=".2f" if show_annotations else "",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="#cccccc",
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot_kws={"size": 8},
            cbar_kws={"label": "Cosine Similarity", "shrink": 0.7},
        )

        ax.set_title(
            f"Chunk-Level Similarity: {safe_doc_a}  vs  {safe_doc_b}",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlabel(f"Chunks from {safe_doc_b}", fontsize=10)
        ax.set_ylabel(f"Chunks from {safe_doc_a}", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

        if theme_colors:
            fig.patch.set_facecolor(theme_colors.get("background", "#FFFFFF"))
            ax.set_facecolor(theme_colors.get("surface", "#F8FAFC"))
            ax.tick_params(colors=theme_colors.get("ink", "#0F172A"))
            ax.xaxis.label.set_color(theme_colors.get("ink", "#0F172A"))
            ax.yaxis.label.set_color(theme_colors.get("ink", "#0F172A"))
            ax.title.set_color(theme_colors.get("ink", "#0F172A"))

        fig.tight_layout()
        return fig


def render_heatmap_ui(
    similarity_df: pd.DataFrame,
    threshold: float = PLAGIARISM_THRESHOLD,
    theme_colors: Optional[dict[str, str]] = None,
):
    """Streamlit UI wrapper for similarity heatmap controls."""
    if similarity_df.empty:
        st.warning("No similarity data available.")
        return

    clean_df = validate_similarity_matrix(similarity_df)
    if clean_df.empty:
        st.warning("Validated similarity matrix is empty.")
        return

    if len(clean_df) < 2:
        st.info("At least 2 documents are required to build a pairwise heatmap.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        zoom_mode = st.radio(
            "Heatmap View",
            ["Fit Matrix", "High Similarity Focus", "Reset View"],
            horizontal=True,
            key="heatmap_zoom_mode",
        )

    # Class Tag Filter selector
    unique_classes = ["All Classes"]
    try:
        from src.db.corpus_db import get_unique_class_sections

        unique_classes.extend(get_unique_class_sections())
    except Exception:
        pass

    selected_class_tag = st.selectbox(
        "Filter by Class Tag",
        unique_classes,
        index=0,
        key="heatmap_class_tag_filter",
        help="Filter heatmap rows and columns to documents matching the selected class tag.",
    )

    if selected_class_tag and selected_class_tag != "All Classes":
        clean_df = filter_heatmap_by_class_tag(clean_df, class_tag=selected_class_tag)

    if clean_df.empty or len(clean_df) < 2:
        st.info(
            f"At least 2 document pairs are required matching class tag '{selected_class_tag}'."
        )
        return

    with col2:
        colormap_name = st.selectbox(
            "Color Map",
            UI_COLORMAP_OPTIONS,
            index=(
                UI_COLORMAP_OPTIONS.index(DEFAULT_UI_COLORMAP)
                if DEFAULT_UI_COLORMAP in UI_COLORMAP_OPTIONS
                else 0
            ),
            key="heatmap_colormap",
        )

        log_scale = st.checkbox(
            "Logarithmic Scale",
            value=False,
            key="heatmap_log_scale",
            help="Apply logarithmic color scaling to better visualize highly skewed similarity distributions.",
        )

        dim_diagonal = st.checkbox(
            "Dim Self-Similarity Diagonal",
            value=False,
            key="heatmap_dim_diagonal",
            help="Grey out or dim 100% self-similarity diagonal cells to focus visual attention on cross-document matches.",
        )

    n = len(clean_df)
    show_annotations = st.checkbox(
        "Show Cell Annotations",
        value=True,
        help="Display similarity scores inside each heatmap cell.",
    )

    fig = plot_similarity_heatmap_plotly(
        clean_df,
        threshold=threshold,
        theme_colors=theme_colors,
        colormap_name=colormap_name,
        log_scale=log_scale,
        dim_diagonal=dim_diagonal,
        show_annotations=show_annotations,
    )

    if zoom_mode == "Fit Matrix":
        fig.update_xaxes(range=[-0.5, n - 0.5])
        fig.update_yaxes(range=[n - 0.5, -0.5])
    elif zoom_mode == "High Similarity Focus":
        matrix = clean_df.values
        coords = np.where(matrix >= threshold)
        if len(coords[0]) > 0:
            min_x = max(min(coords[1]) - 1, -0.5)
            max_x = min(max(coords[1]) + 1, n - 0.5)
            min_y = max(min(coords[0]) - 1, -0.5)
            max_y = min(max(coords[0]) + 1, n - 0.5)
            fig.update_xaxes(range=[min_x, max_x])
            fig.update_yaxes(range=[max_y, min_y])
        else:
            st.info("No document pairs found above the similarity threshold.")
    elif zoom_mode == "Reset View":
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)

    main_col, mini_col = st.columns([5, 1])

    with main_col:
        st.plotly_chart(fig, use_container_width=True)

    with mini_col:
        mini_fig = plot_similarity_minimap(
            clean_df,
            colormap_name=colormap_name,
        )
        st.plotly_chart(mini_fig, use_container_width=True)


# ── Multi-Matrix Heatmap Grid Overlay Visualizer (Issue #1504) ─────────────────


def plot_multi_heatmap_grid(
    matrices: dict[str, pd.DataFrame],
    colorscale: str = "Viridis",
    threshold: float = PLAGIARISM_THRESHOLD,
    show_annotations: bool = True,
    shared_colorbar: bool = True,
    theme_colors: Optional[dict[str, str]] = None,
    font_scale: float = 1.0,
):
    """Render a Plotly subplot grid displaying similarity heatmaps side-by-side.

    Instructors analyzing multi-assignment plagiarism can compare similarity
    heatmaps for multiple assignments or semesters simultaneously in a single
    grid layout with a shared color bar.

    Parameters
    ----------
    matrices : dict[str, pd.DataFrame]
        Mapping of assignment/semester label → square similarity DataFrame.
        Each DataFrame must be square and contain float similarity values in [0, 1].
        Empty or invalid DataFrames are rendered as an empty placeholder panel.
    colorscale : str, default="Viridis"
        Plotly colorscale name applied uniformly to all heatmap panels.
    threshold : float, default=PLAGIARISM_THRESHOLD
        Similarity value at or above which cell borders are highlighted red to
        flag potential plagiarism pairs.
    show_annotations : bool, default=True
        Whether to render numeric similarity values inside each heatmap cell.
        Automatically disabled per panel if that panel has more than 15 documents.
    shared_colorbar : bool, default=True
        If True, only the last panel shows its color bar (shared visual reference).
        If False, each panel renders its own individual color bar.
    theme_colors : Optional[Dict[str, str]], default=None
        Theme palette dictionary with keys such as "background", "ink", "surface".
    font_scale : float, default=1.0
        Scaling factor applied to all text elements (title, tick labels, annotations).

    Returns
    -------
    go.Figure
        Plotly Figure containing a subplot grid of heatmap panels.
        - Returns an empty Figure with an explanatory annotation if `matrices` is
          empty or None.
        - Invalid or empty individual DataFrames are rendered as placeholder panels.

    Examples
    --------
    >>> import pandas as pd
    >>> from src.visualization.heatmap import plot_multi_heatmap_grid
    >>> spring = pd.DataFrame(
    ...     [[1.0, 0.8], [0.8, 1.0]], columns=["A", "B"], index=["A", "B"]
    ... )
    >>> fall = pd.DataFrame(
    ...     [[1.0, 0.3], [0.3, 1.0]], columns=["A", "B"], index=["A", "B"]
    ... )
    >>> fig = plot_multi_heatmap_grid({"Spring": spring, "Fall": fall})
    """
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    scale = max(0.5, float(font_scale))
    bg_color = _get_theme_color(theme_colors, "background", "rgba(0,0,0,0)")
    ink_color = _get_theme_color(theme_colors, "ink", "#0F172A")

    # ── Guard: empty or None input ───────────────────────────────────────────
    if not matrices:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Multi-Matrix Heatmap Grid",
                font=dict(
                    size=int(18 * scale), family=DEFAULT_FONT_FAMILY, color=ink_color
                ),
            ),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=ink_color),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig.add_annotation(
            text="No similarity matrices provided for multi-heatmap grid",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=int(14 * scale), color="#666666"),
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=10,
            bgcolor="#f8f9fa",
        )
        return fig

    labels = list(matrices.keys())
    n_panels = len(labels)

    # ── Compute grid dimensions ──────────────────────────────────────────────
    n_cols = min(n_panels, 3)
    n_rows = math.ceil(n_panels / n_cols)

    subplot_titles = [TitleSanitizer.sanitize(lbl) for lbl in labels]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # ── Pre-validate each matrix ─────────────────────────────────────────────
    cleaned: dict[str, Optional[pd.DataFrame]] = {}
    for lbl, df in matrices.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 2:
            cleaned[lbl] = None
            continue
        try:
            cleaned[lbl] = validate_similarity_matrix(df)
        except ValueError:
            cleaned[lbl] = None

    # ── Build per-panel traces ───────────────────────────────────────────────
    all_shapes: list = []

    for panel_idx, lbl in enumerate(labels):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1

        df = cleaned[lbl]

        # Placeholder trace for invalid / empty panels
        if df is None:
            fig.add_trace(
                go.Heatmap(
                    z=[[0]],
                    x=["—"],
                    y=["—"],
                    showscale=False,
                    colorscale=[[0, "#f0f0f0"], [1, "#f0f0f0"]],
                    hoverinfo="skip",
                    xgap=0,
                    ygap=0,
                ),
                row=row,
                col=col,
            )
            # Add "no data" annotation centred in this subplot
            fig.add_annotation(
                text="No data<br>(≥2 documents required)",
                xref=f"x{panel_idx + 1 if panel_idx > 0 else ''} domain",
                yref=f"y{panel_idx + 1 if panel_idx > 0 else ''} domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=int(11 * scale), color="#888888"),
            )
            continue

        names = [TitleSanitizer.sanitize(str(c)) for c in df.columns]
        n = len(names)
        z_vals = df.values.tolist()

        # Hover text
        hover_text = [
            [
                f"<b>{names[i]}</b> vs <b>{names[j]}</b><br>"
                f"Similarity: {df.values[i, j]:.2%}<br>"
                f"Status: {'Flagged ⚠️' if i != j and df.values[i, j] >= threshold else 'Normal'}"
                for j in range(n)
            ]
            for i in range(n)
        ]

        # Annotation texts
        panel_annotations: list = []
        if show_annotations and n <= 15:
            for i in range(n):
                for j in range(n):
                    val = df.values[i, j]
                    if pd.isna(val):
                        continue
                    font_color = "white" if val > 0.5 else "black"
                    panel_annotations.append(
                        dict(
                            x=names[j],
                            y=names[i],
                            text=f"{val:.2f}",
                            showarrow=False,
                            font=dict(
                                size=int(max(7, 11 - n) * scale),
                                color=font_color,
                                family=DEFAULT_FONT_FAMILY,
                            ),
                            xref=f"x{panel_idx + 1 if panel_idx > 0 else ''}",
                            yref=f"y{panel_idx + 1 if panel_idx > 0 else ''}",
                        )
                    )

        all_shapes.extend(panel_annotations)

        # Only show color bar on last panel (shared reference) unless disabled
        show_cb = (not shared_colorbar) or (panel_idx == n_panels - 1)

        fig.add_trace(
            go.Heatmap(
                z=z_vals,
                x=names,
                y=names,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                colorscale=colorscale,
                zmin=0.0,
                zmax=1.0,
                showscale=show_cb,
                colorbar=(
                    dict(
                        title=dict(
                            text="Cosine<br>Similarity",
                            font=dict(size=int(11 * scale)),
                        ),
                        thickness=12,
                        tickformat=".0%",
                        len=0.8,
                        x=1.02,
                    )
                    if show_cb
                    else None
                ),
                xgap=2,
                ygap=2,
            ),
            row=row,
            col=col,
        )

        # Red border shapes for flagged pairs (above threshold)
        x_axis_name = f"x{panel_idx + 1}" if panel_idx > 0 else "x"
        y_axis_name = f"y{panel_idx + 1}" if panel_idx > 0 else "y"
        for i in range(n):
            for j in range(n):
                if i != j and df.values[i, j] >= threshold:
                    all_shapes.append(
                        dict(
                            type="rect",
                            xref=x_axis_name,
                            yref=y_axis_name,
                            x0=j - 0.5,
                            x1=j + 0.5,
                            y0=i - 0.5,
                            y1=i + 0.5,
                            line=dict(color="#d62728", width=2.5),
                            fillcolor="rgba(0,0,0,0)",
                        )
                    )

        # Per-panel axis styling
        tick_sz = int(max(7, 10 - n // 3) * scale)
        fig.update_xaxes(
            tickangle=-30,
            tickfont=dict(size=tick_sz, color=ink_color),
            color=ink_color,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            autorange="reversed",
            tickfont=dict(size=tick_sz, color=ink_color),
            color=ink_color,
            row=row,
            col=col,
        )

    # ── Global layout ────────────────────────────────────────────────────────
    # Separate shape-dicts from annotation-dicts
    shape_items = [s for s in all_shapes if s.get("type") in ("rect", "line", "circle")]
    annotation_items = [a for a in all_shapes if "text" in a]

    panel_height = max(
        350,
        80
        * max(
            (len(cleaned[lbl].columns) if cleaned[lbl] is not None else 2)
            for lbl in labels
        ),
    )
    total_height = n_rows * panel_height + 100

    fig.update_layout(
        title=dict(
            text="Multi-Assignment Similarity Heatmap Grid",
            font=dict(
                size=int(20 * scale),
                family=DEFAULT_FONT_FAMILY,
                color=ink_color,
            ),
        ),
        height=total_height,
        autosize=True,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=ink_color, family=DEFAULT_FONT_FAMILY),
        shapes=shape_items,
        annotations=(list(fig.layout.annotations) + annotation_items),
        margin=dict(l=60, r=80, t=80, b=60),
        hoverlabel=dict(
            bgcolor=_get_theme_color(theme_colors, "surface", "white"),
            font_size=int(12 * scale),
            font_family=DEFAULT_FONT_FAMILY,
        ),
    )

    return fig
