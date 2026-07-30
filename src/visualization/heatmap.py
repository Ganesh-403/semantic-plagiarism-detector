"""
heatmap.py
----------
Generates similarity heatmaps for Semantic Plagiarism Detector.

This module provides high-quality, customizable heatmap visualizations for 
document similarity matrices. It bridges the gap between backend scoring 
and frontend rendering, offering both static (Matplotlib/Seaborn) and 
interactive (Plotly) options. 

Recent additions (Issue #697):
- Streamlit UI components (`render_heatmap_ui`) to inject interactive controls.
- Dynamic colormap selection (Viridis, Plasma, Coolwarm, YlOrRd) accessible to end-users.
- Enhanced matrix validation and error handling for robust UI behavior.
- Support for both light and dark modes inherited from the application theme system.

Security Updates (Issue #704):
- Added TitleSanitizer and MatplotlibInjectionError to prevent text formatting injection attacks.
"""

import re
from contextlib import contextmanager
from typing import Generator, Optional
import logging

import matplotlib
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
    )
except ImportError:
    # Fallback for standalone testing or isolated environments
    UI_COLORMAP_OPTIONS = ["Viridis", "Plasma", "Coolwarm", "YlOrRd"]
    MATPLOTLIB_CMAP_MAPPING = {"Viridis": "viridis"}
    PLOTLY_CMAP_MAPPING = {"Viridis": "Viridis"}
    DEFAULT_UI_COLORMAP = "Viridis"


# ── Security & Sanitization (Issue #704) ───────────────────────────────────────

class MatplotlibInjectionError(ValueError):
    """Raised when a string contains forbidden formatting or injection tokens."""
    pass


class TitleSanitizer:
    """Sanitizes user-provided titles and labels to prevent injection exploits."""
    
    # Patterns to strip or detect dangerous Matplotlib / formatting injection attempts
    MATHTEXT_PATTERN = re.compile(r"[$_\\^]")
    HTML_TAG_PATTERN = re.compile(r"<[^>]*?>")
    
    @classmethod
    def sanitize(cls, text: Optional[str], strict: bool = False) -> str:
        if not text:
            return ""
        
        # 1. Remove HTML tags
        clean_text = cls.HTML_TAG_PATTERN.sub("", str(text))
        
        # 2. Check for strict mathtext injection if enabled
        if strict and cls.MATHTEXT_PATTERN.search(clean_text):
            logger.error("Potential Matplotlib text injection detected.")
            raise MatplotlibInjectionError("Provided string contains unauthorized formatting characters.")
            
        # 3. Strip raw control or unsafe escape codes
        clean_text = clean_text.replace("\n", " ").replace("\r", " ")
        return clean_text.strip()


# ── Data Validation Helpers ────────────────────────────────────────────────────

def validate_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans the input similarity matrix before visualization.
    
    Ensures that:
    1. The matrix is square.
    2. Values are strictly bounded between 0.0 and 1.0.
    3. Null values are appropriately filled.
    
    Args:
        df (pd.DataFrame): The raw similarity matrix.
        
    Returns:
        pd.DataFrame: A cleaned, safe-to-plot DataFrame.
        
    Raises:
        ValueError: If the dataframe cannot be coerced into a valid square matrix.
    """
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
        clean_df.fillna(0.0, inplace=True)

    clean_df = clean_df.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(clean_df.values, 1.0)

    return clean_df


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
    annotate: bool = True,
    dpi: int = 150,
    theme_colors: Optional[dict] = None,
    colormap_name: str = DEFAULT_UI_COLORMAP,
    mask_threshold: Optional[float] = None,
) -> Figure:
    """
    High-resolution Matplotlib heatmap optimized for static PNG export.
    """
    # Sanitize title input to prevent formatting injection (Issue #704)
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

    if n == 0:
        with matplotlib_figure() as (fig, ax):
            ax.set_title(safe_title)
            return fig

    if figsize is None:
        cell_size = max(1.2, 6 / n)
        width = max(6.0, n * cell_size + 2.0)
        height = max(5.0, n * cell_size + 1.5)
        figsize = (width, height)

    mask = None
    if mask_threshold is not None:
        mask = similarity_df < mask_threshold

    with matplotlib_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        sns.heatmap(
            clean_df,
            ax=ax,
            annot=annotate,
            fmt=".2f" if annotate else "",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
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

        data = clean_df.values

        for i in range(n):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (i, i), 1, 1,
                    boxstyle="square,pad=0",
                    linewidth=2,
                    edgecolor="#555555",
                    facecolor="none",
                    zorder=3,
                )
            )

        for i in range(n):
            for j in range(n):
                if i != j and data[i, j] >= threshold:
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
        
        # Sanitize column/row index labels to prevent injection via document names
        safe_labels = [TitleSanitizer.sanitize(str(lbl)) for lbl in clean_df.columns]
        ax.set_xticklabels(safe_labels, rotation=30, ha="right", fontsize=max(8, 11 - n // 3))
        ax.set_yticklabels(safe_labels, rotation=0, fontsize=max(8, 11 - n // 3))

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
    theme_colors: Optional[dict] = None,
    colormap_name: str = DEFAULT_UI_COLORMAP,
    annotate: bool = True,
    mask_threshold: Optional[float] = None,
):
    """
    Interactive Plotly heatmap featuring dynamic hover values and custom threshold bounds.
    """
    import plotly.graph_objects as go

    try:
        safe_title = TitleSanitizer.sanitize(title)
    except MatplotlibInjectionError:
        safe_title = "Semantic Similarity Matrix"

    cmap = PLOTLY_CMAP_MAPPING.get(colormap_name, "Viridis")

    if similarity_df.empty or len(similarity_df) == 0:
        fig = go.Figure()
        fig.update_layout(title=safe_title)
        fig.add_annotation(text="No data available to plot.", showarrow=False, font=dict(size=14))
        return fig
        
    try:
        clean_df = validate_similarity_matrix(similarity_df)
    except ValueError as error:
        logger.error(error)
        return go.Figure()

    names = [TitleSanitizer.sanitize(str(col)) for col in clean_df.columns]
    z_matrix = clean_df.values.tolist()
    if mask_threshold is not None:
        z_matrix = [
            [
                val if val >= mask_threshold else None
                for val in row
            ]
            for row in clean_df.values.tolist()
        ]
    n = len(names)

    hover_text = [
        [
            f"<b>{names[i]}</b> vs <b>{names[j]}</b><br>"
            f"Similarity: {clean_df.values[i, j]:.2%}<br>"
            f"Status: {'Flagged' if (i != j and clean_df.values[i, j] >= threshold) else 'Normal'}"
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
            hovertemplate="%{text}<extra></extra>",
            colorscale=cmap,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title="Cosine Similarity", 
                thickness=15,
                tickformat=".0%"
            ),
            xgap=2,
            ygap=2,
        )
    )

    annotations = []
    if annotate:
        for i in range(n):
            for j in range(n):
                val = clean_df.values[i, j]
                if mask_threshold is not None and val < mask_threshold:
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
                        font=dict(
                            size=max(9, 14 - n),
                            color=font_color,
                            family="Arial, sans-serif",
                        ),
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
        title=dict(
            text=safe_title,
            font=dict(
                size=18,
                family="Arial, sans-serif",
                color=ink_color
            )
        ),
        height=max(500, n * cell_px + 150),
        autosize=True,

        xaxis=dict(
            side="bottom",
            tickangle=-30,
            title="Document ID",
            color=ink_color,
            fixedrange=False
        ),

        yaxis=dict(
            autorange="reversed",
            title="Document ID",
            color=ink_color,
            fixedrange=False
        ),

        annotations=annotations,
        shapes=shapes,

        margin=dict(
            l=140,
            r=60,
            t=70,
            b=140
        ),

        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,

        font=dict(
            color=ink_color
        ),

        hoverlabel=dict(
            bgcolor=_get_theme_color(
                theme_colors,
                "surface",
                "white"
            ),
            font_size=14,
            font_family="Arial"
        )
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
) -> Figure:
    """
    Renders a granular, chunk-level similarity heatmap between two specific documents.
    """
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

    with matplotlib_figure(figsize=(max(8, nb * 1.5), max(6, na * 0.8)), dpi=150) as (fig, ax):
        sns.heatmap(
            sim_matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
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
    similarity_df,
    threshold=PLAGIARISM_THRESHOLD,
    theme_colors=None,
):
    """
    Streamlit UI wrapper for similarity heatmap controls.

    Provides:
    - Fit Matrix view
    - High Similarity Focus view
    - Reset View
    - Dynamic colormap selection
    """

    if similarity_df.empty:
        st.warning("No similarity data available.")
        return

    clean_df = validate_similarity_matrix(similarity_df)

    if clean_df.empty:
        st.warning("Validated similarity matrix is empty.")
        return

    # Heatmap view controls
    zoom_mode = st.radio(
        "Heatmap View",
        [
            "Fit Matrix",
            "High Similarity Focus",
            "Reset View",
        ],
        horizontal=True,
        key="heatmap_zoom_mode",
    )

    # Colormap selector
    default_index = (
        UI_COLORMAP_OPTIONS.index(DEFAULT_UI_COLORMAP)
        if DEFAULT_UI_COLORMAP in UI_COLORMAP_OPTIONS
        else 0
    )

    colormap_name = st.selectbox(
        "Color Map",
        UI_COLORMAP_OPTIONS,
        index=default_index,
        key="heatmap_colormap",
    )

    n = len(clean_df)

    # Create plot with selected colormap
    fig = plot_similarity_heatmap_plotly(
        clean_df,
        threshold=threshold,
        theme_colors=theme_colors,
        colormap_name=colormap_name,
    )

    # Apply zoom modes
    if zoom_mode == "Fit Matrix":

        fig.update_xaxes(
            range=[-0.5, n - 0.5]
        )

        fig.update_yaxes(
            range=[n - 0.5, -0.5]
        )

    elif zoom_mode == "High Similarity Focus":

        matrix = clean_df.values

        coords = np.where(
            matrix >= threshold
        )

        if len(coords[0]) > 0:

            min_x = max(min(coords[1]) - 1, -0.5)
            max_x = min(max(coords[1]) + 1, n - 0.5)

            min_y = max(min(coords[0]) - 1, -0.5)
            max_y = min(max(coords[0]) + 1, n - 0.5)

            fig.update_xaxes(
                range=[
                    min_x,
                    max_x,
                ]
            )

            fig.update_yaxes(
                range=[
                    max_y,
                    min_y,
                ]
            )

        else:
            st.info(
                "No document pairs found above the similarity threshold."
            )

    elif zoom_mode == "Reset View":

        fig.update_xaxes(
            autorange=True
        )

        fig.update_yaxes(
            autorange=True
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )