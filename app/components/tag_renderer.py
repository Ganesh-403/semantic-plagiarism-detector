"""
app/components/tag_renderer.py
------------------------------
Streamlit UI components for rendering DocumentTags with visual indicators.

Provides functions to render semantic tags with appropriate styling based
on their source (manual vs AI) and confidence level. Low-confidence tags
are rendered with muted colors and warning icons to prompt manual verification.

Issue #2812: Visual indicators for low-confidence tags.
"""

from typing import List, Optional
import html as html_lib

import streamlit as st

from src.core.models.categorization import DocumentTag

# CSS styles for tag badges, supporting both Light and Dark modes
_TAG_CSS = """
<style>
    .tag-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
        line-height: 1.4;
        border: 1px solid transparent;
    }
    
    /* High confidence / Manual tags (Default theme colors) */
    .tag-high-confidence, .tag-source-manual {
        background-color: rgba(37, 99, 235, 0.15); /* Blue-600 with alpha */
        color: #2563eb;
        border-color: rgba(37, 99, 235, 0.3);
    }
    
    /* AI Generated tags (High confidence) */
    .tag-source-ai_generated.tag-high-confidence {
        background-color: rgba(16, 185, 129, 0.15); /* Emerald-500 with alpha */
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    /* Low confidence tags (Muted/Warning styling) - Issue #2812 */
    .tag-low-confidence {
        background-color: rgba(245, 158, 11, 0.15); /* Amber-500 with alpha */
        color: #d97706;
        border-color: rgba(245, 158, 11, 0.4);
        border-style: dashed;
    }
    
    .tag-warning-icon {
        margin-right: 4px;
        font-size: 0.9rem;
    }
    
    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .tag-high-confidence, .tag-source-manual {
            background-color: rgba(96, 165, 250, 0.2);
            color: #93c5fd;
        }
        .tag-source-ai_generated.tag-high-confidence {
            background-color: rgba(52, 211, 153, 0.2);
            color: #6ee7b7;
        }
        .tag-low-confidence {
            background-color: rgba(251, 191, 36, 0.2);
            color: #fbbf24;
            border-color: rgba(251, 191, 36, 0.5);
        }
    }
</style>
"""


def render_tag(tag: DocumentTag, show_confidence: bool = True) -> str:
    """Render a single DocumentTag as an HTML badge with visual indicators.

    Args:
        tag: The DocumentTag instance to render.
        show_confidence: Whether to display the confidence percentage.

    Returns:
        HTML string representing the styled tag badge.
    """
    badge = tag.get_html_badge(show_confidence=show_confidence)
    label = html_lib.escape(tag.name, quote=True)
    # Clickable tag chips need an accessible name for screen readers (#3768).
    return badge.replace(
        "<span ",
        f'<span role="button" aria-label="{label}" ',
        1,
    )


def render_tag_collection(
    tags: list[DocumentTag], title: Optional[str] = None, show_confidence: bool = True
) -> None:
    """Render a collection of DocumentTags in the Streamlit UI.

    Injects the required CSS and renders each tag as an inline HTML badge.
    Low-confidence tags are visually distinct to prompt user verification.

    Args:
        tags: List of DocumentTag instances to render.
        title: Optional title to display above the tag collection.
        show_confidence: Whether to display confidence percentages on AI tags.
    """
    if not tags:
        return

    # Inject CSS once per page load
    st.markdown(_TAG_CSS, unsafe_allow_html=True)

    if title:
        st.markdown(f"**{title}**")

    # Render all tags as a single HTML block for proper inline wrapping
    html_blocks = [render_tag(tag, show_confidence=show_confidence) for tag in tags]
    html_content = " ".join(html_blocks)

    st.markdown(
        f'<div style="margin-bottom: 10px; line-height: 2;">{html_content}</div>',
        unsafe_allow_html=True,
    )


def render_low_confidence_warning(count: int) -> None:
    """Render a summary warning if low-confidence tags are present.

    Args:
        count: Number of low-confidence tags detected.
    """
    if count > 0:
        st.warning(
            f"⚠️ **{count} AI-generated tag{'s' if count > 1 else ''}** have low confidence (<60%) "
            "and require manual verification. These are highlighted with a dashed border."
        )
