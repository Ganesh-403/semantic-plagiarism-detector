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

# ── Logger Configuration ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Colormap Mappings & Constants ──────────────────────────────────────────────
try:
    from app.theme import (
        UI_COLORMAP_OPTIONS,
        MATPLOTLIB_CMAP_MAPPING,
        PLOTLY_CMAP_MAPPING,
        DEFAULT_UI_COLORMAP,
        apply_matplotlib_theme,
    )
except ImportError:
    # Fallback for standalone testing or isolated environments
    UI_COLORMAP_OPTIONS = ["Viridis", "Plasma", "Coolwarm", "YlOrRd"]
    MATPLOTLIB_CMAP_MAPPING = {"Viridis": "viridis"}
    PLOTLY_CMAP_MAPPING = {"Viridis": "Viridis"}
    DEFAULT_UI_COLORMAP = "Viridis"

    def apply_matplotlib_theme(theme_colors=None):
        pass


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
            raise MatplotlibInjectionError("Provided string contains unauthorized formatting characters.")
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
                fname = d.get("filename") if isinstance(d, dict) else getattr(d, "filename", None)
                csec = d.get("class_section") if isinstance(d, dict) else getattr(d, "class_section", None)
                if fname:
                    doc_class_map[fname] = csec
        except Exception as e:
            logger.warning(f"Could not load document class map from database: {e}")
            doc_class_map = {}

    matching_cols = [
        col for col in similarity_df.columns
        if doc_class_map.get(str(col)) == class_tag
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
    theme_colors: Optional[Dict[str, str]] = None,
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
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc"),
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
            dim_color = theme_colors.get("border", "#cccccc") if theme_colors else "#cccccc"
            for i in range(n):
                ax.add_patch(
                    mpatches.Rectangle(
                        (i, i), 1, 1,
                        facecolor=dim_color,
                        alpha=0.4,
                        zorder=2,
                    )
                )

        data = clean_df.values
        for i in range(n):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (i, i), 1, 1,
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
                            (j, i), 1, 1,
                            boxstyle="square,pad=0",
                            linewidth=2.5,
                            edgecolor="#d62728",
                            facecolor="none",
                            zorder=4,
                        )
                    )

        ax.set_title(safe_title, fontsize=15, fontweight="bold", pad=16, color=title_color)
        ax.set_xlabel("Documents", fontsize=11, labelpad=10)
        ax.set_ylabel("Documents", fontsize=11, labelpad=10)

        safe_labels = [TitleSanitizer.sanitize(str(lbl)) for lbl in clean_df.columns]

        tick_label_fontsize = max(6, 12 - n // 10)

        ax.set_xticklabels(
            safe_labels,
            rotation=30,
            ha="right",
            fontsize=tick_label_fontsize,
        )
        ax.set_yticklabels(
            safe_labels,
            rotation=0,
            fontsize=tick_label_fontsize,
        )

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
                legend.get_frame().set_facecolor(theme_colors.get("background", "#FFFFFF"))
                legend.get_frame().set_edgecolor(theme_colors.get("border", "#E2E8F0"))

        fig.tight_layout()
        return fig


# ── Interactive Visualization (Plotly) ─────────────────────────────────────────
def plot_similarity_heatmap_plotly(
    similarity_df: pd.DataFrame,
    title: str = "Semantic Similarity Matrix",
    threshold: float = PLAGIARISM_THRESHOLD,
    theme_colors: Optional[Dict[str, str]] = None,
    colormap_name: str = DEFAULT_UI_COLORMAP,
    show_annotations: bool = True,
    mask_threshold: Optional[float] = None,
    log_scale: bool = False,
    class_tag: Optional[str] = None,
    doc_class_map: Optional[dict] = None,
    dim_diagonal: bool = False,
):
    """Interactive Plotly heatmap featuring dynamic hover values and custom threshold bounds."""
    import plotly.graph_objects as go

    if class_tag and class_tag != "All Classes":
        similarity_df = filter_heatmap_by_class_tag(
            similarity_df, class_tag=class_tag, doc_class_map=doc_class_map
        )

    try:
        safe_title = TitleSanitizer.sanitize(title)
    except MatplotlibInjectionError:
        safe_title = "Semantic Similarity Matrix"

    cmap = PLOTLY_CMAP_MAPPING.get(colormap_name, "Viridis")

    try:
        clean_df = validate_similarity_matrix(similarity_df)
    except ValueError as error:
        logger.error(error)
        return go.Figure()

    # Issue #839: Handle empty or single document (< 2) input cleanly
    if clean_df.empty or len(clean_df) < 2:
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
            colorscale=cmap,
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
                if pd.isna(val) or (mask_threshold is not None and val < mask_threshold):
                    continue
                font_color = "black" if (0.3 < val < 0.8 and cmap not in ["Viridis", "Plasma"]) else "white"
                if cmap == "YlOrRd" and val < 0.6:
                    font_color = "black"

                annotations.append(
                    dict(
                        x=names[j],
                        y=names[i],
                        text=f"{val:.2f}",
                        showarrow=False,
                        font=dict(size=max(9, 14 - n), color=font_color, family="Arial, sans-serif"),
                    )
                )

    shapes = []
    for i in range(n):
        for j in range(n):
            if i != j and clean_df.values[i, j] >= threshold:
                if mask_threshold is not None and clean_df.values[i, j] < mask_threshold:
                    continue
                shapes.append(
                    dict(
                        type="rect",
                        x0=j - 0.5,
                        x1=j + 0.5,
                        y0=i - 0.5,
                        y1=i + 0.5,
                        line=dict(color="#d62728", width=3),
                        fillcolor="rgba(0,0,0,0)"
                    )
                )

    cell_px = max(80, 600 // n)
    bg_color = _get_theme_color(theme_colors, "background", "rgba(0,0,0,0)")
    ink_color = _get_theme_color(theme_colors, "ink", "#0F172A")

    fig.update_layout(
        title=dict(text=safe_title, font=dict(size=18, family="Arial, sans-serif", color=ink_color)),
        height=max(500, n * cell_px + 150),
        autosize=True,
        xaxis=dict(side="bottom", tickangle=-30, title="Document ID", color=ink_color, fixedrange=False),
        yaxis=dict(autorange="reversed", title="Document ID", color=ink_color, fixedrange=False),
        annotations=annotations,
        shapes=shapes,
        margin=dict(l=140, r=60, t=70, b=140),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=ink_color),
        hoverlabel=dict(bgcolor=_get_theme_color(theme_colors, "surface", "white"), font_size=14, font_family="Arial")
    )

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
        return TitleSanitizer.sanitize(clean_text[:max_chars].strip() + "…" if len(clean_text) > max_chars else clean_text)

    row_labels = [f"A{i + 1}: {short_label(c)}" for i, c in enumerate(chunks_a)]
    col_labels = [f"B{j + 1}: {short_label(c)}" for j, c in enumerate(chunks_b)]

    apply_matplotlib_theme(theme_colors)

    with matplotlib_figure(figsize=(max(8, nb * 1.5), max(6, na * 0.8)), dpi=150) as (fig, ax):
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
    theme_colors: Optional[Dict[str, str]] = None,
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
        st.info(f"At least 2 document pairs are required matching class tag '{selected_class_tag}'.")
        return

    with col2:
        colormap_name = st.selectbox(
            "Color Map",
            UI_COLORMAP_OPTIONS,
            index=UI_COLORMAP_OPTIONS.index(DEFAULT_UI_COLORMAP) if DEFAULT_UI_COLORMAP in UI_COLORMAP_OPTIONS else 0,
            key="heatmap_colormap",
        )

        log_scale = st.checkbox(
            "Logarithmic Scale",
            value=False,
            key="heatmap_log_scale",
            help="Apply logarithmic color scaling to better visualize highly skewed similarity distributions."
        )

        dim_diagonal = st.checkbox(
            "Dim Self-Similarity Diagonal",
            value=False,
            key="heatmap_dim_diagonal",
            help="Grey out or dim 100% self-similarity diagonal cells to focus visual attention on cross-document matches."
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

    st.plotly_chart(fig, use_container_width=True)
    