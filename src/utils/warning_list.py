"""Search, multi-column sorting, and pagination for plagiarism warnings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from app.theme import badge_html, tier_from_severity_label
from src.core.config import normalize_severity_label, severity_from_score, severity_rank
from src.db.incidents import _normalise_pair, add_false_positive, get_false_positives

SORT_FIELDS = {
    "Similarity": "similarity",
    "Document A": "doc_a",
    "Document B": "doc_b",
    "Severity": "severity_rank",
}


@dataclass(frozen=True)
class WarningPage:
    items: list[dict[str, Any]]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    start_index: int
    end_index: int


def _normalise_warning(
    warning: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        similarity = float(warning.get("similarity", 0.0))
    except (TypeError, ValueError):
        similarity = 0.0

    raw_severity = str(warning.get("severity", "")).strip()
    try:
        severity = normalize_severity_label(raw_severity)
    except ValueError:
        severity = severity_from_score(similarity)

    return {
        **dict(warning),
        "doc_a": str(warning.get("doc_a", "")).strip(),
        "doc_b": str(warning.get("doc_b", "")).strip(),
        "similarity": similarity,
        "severity": severity,
        "severity_rank": severity_rank(severity),
    }


def filter_warnings(
    warnings: Iterable[Mapping[str, Any]],
    search_query: str = "",
) -> list[dict[str, Any]]:
    normalised = [_normalise_warning(item) for item in warnings]
    query = search_query.strip().casefold()

    if not query:
        return normalised

    return [
        item
        for item in normalised
        if query in item["doc_a"].casefold() or query in item["doc_b"].casefold()
    ]


def sort_warnings(
    warnings: Iterable[Mapping[str, Any]],
    *,
    primary_field: str = "similarity",
    primary_descending: bool = True,
    secondary_field: str = "doc_a",
    secondary_descending: bool = False,
) -> list[dict[str, Any]]:
    items = [_normalise_warning(item) for item in warnings]
    allowed = {"similarity", "doc_a", "doc_b", "severity_rank"}

    if primary_field not in allowed:
        primary_field = "similarity"
    if secondary_field not in allowed:
        secondary_field = "doc_a"

    def key_for(field: str):
        def key(item: Mapping[str, Any]):
            value = item[field]
            return value.casefold() if isinstance(value, str) else value

        return key

    items.sort(key=key_for(secondary_field), reverse=secondary_descending)
    items.sort(key=key_for(primary_field), reverse=primary_descending)
    return items


def paginate_warnings(
    warnings: Sequence[Mapping[str, Any]],
    *,
    page: int = 1,
    page_size: int = 10,
) -> WarningPage:
    safe_page_size = max(1, int(page_size))
    total_items = len(warnings)
    total_pages = max(1, math.ceil(total_items / safe_page_size))
    safe_page = min(max(1, int(page)), total_pages)

    start = (safe_page - 1) * safe_page_size
    end = min(start + safe_page_size, total_items)

    return WarningPage(
        items=[dict(item) for item in warnings[start:end]],
        total_items=total_items,
        page=safe_page,
        page_size=safe_page_size,
        total_pages=total_pages,
        start_index=start + 1 if total_items else 0,
        end_index=end,
    )


def prepare_warning_page(
    warnings: Iterable[Mapping[str, Any]],
    *,
    search_query: str = "",
    primary_field: str = "similarity",
    primary_descending: bool = True,
    secondary_field: str = "doc_a",
    secondary_descending: bool = False,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[dict[str, Any]], WarningPage]:
    filtered = filter_warnings(warnings, search_query)
    sorted_items = sort_warnings(
        filtered,
        primary_field=primary_field,
        primary_descending=primary_descending,
        secondary_field=secondary_field,
        secondary_descending=secondary_descending,
    )
    return sorted_items, paginate_warnings(
        sorted_items,
        page=page,
        page_size=page_size,
    )


def _reset_page() -> None:
    st.session_state.warning_page = 1


def _has_exact_match(doc_a: str, doc_b: str) -> bool:
    """Check if two documents share at least one exact matching chunk (ignoring whitespace)."""
    if (
        "analysis_results" not in st.session_state
        or st.session_state.analysis_results is None
    ):
        return False
    chunked_docs = st.session_state.analysis_results[1]
    chunks_a = chunked_docs.get(doc_a, [])
    chunks_b = chunked_docs.get(doc_b, [])

    # Normalize chunks by removing all whitespace
    norm_a = {"".join(c.split()) for c in chunks_a if c.strip()}
    norm_b = {"".join(c.split()) for c in chunks_b if c.strip()}

    return not norm_a.isdisjoint(norm_b)


def render_warning_controls(
    flags: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    ai_probabilities: dict[str, dict[str, Any]] | None = None,
) -> None:
    if "warning_page" not in st.session_state:
        st.session_state.warning_page = 1

    st.caption(f"Pairs with similarity ≥ **{threshold:.2f}**")
    dismissed_pairs = get_false_positives()
    filtered_flags = [
        f
        for f in flags
        if _normalise_pair(f["doc_a"], f["doc_b"]) not in dismissed_pairs
    ]

    if not filtered_flags:
        st.success("✅ No suspicious pairs found above the current threshold.")
        return

    search_col, toggle_col, size_col = st.columns([3, 2, 1])

    with search_col:
        search_query = st.text_input(
            "Search warnings",
            placeholder="Search by either document name…",
            key="warning_search",
            on_change=_reset_page,
        )

    with toggle_col:
        hide_low_severity = st.checkbox(
            "Hide Low Severity",
            key="hide_low_severity",
        )

    with size_col:
        page_size = st.selectbox(
            "Warnings per page",
            [10, 25, 50],
            key="warning_page_size",
            on_change=_reset_page,
        )

    p1, d1, p2, d2 = st.columns([2, 1, 2, 1])

    with p1:
        primary_label = st.selectbox(
            "Primary sort",
            list(SORT_FIELDS),
            key="warning_primary_sort",
            on_change=_reset_page,
        )

    with d1:
        primary_direction = st.selectbox(
            "Direction",
            ["Descending", "Ascending"],
            key="warning_primary_direction",
            on_change=_reset_page,
        )

    with p2:
        secondary_label = st.selectbox(
            "Then sort by",
            list(SORT_FIELDS),
            index=1,
            key="warning_secondary_sort",
            on_change=_reset_page,
        )

    with d2:
        secondary_direction = st.selectbox(
            "Then direction",
            ["Ascending", "Descending"],
            key="warning_secondary_direction",
            on_change=_reset_page,
        )

    # Hide low severity warnings when checkbox is enabled
    display_flags = [_normalise_warning(flag) for flag in filtered_flags]

    if hide_low_severity:
        display_flags = [flag for flag in display_flags if flag["severity"] != "Low"]

    sorted_flags, current_page = prepare_warning_page(
        display_flags,
        search_query=search_query,
        primary_field=SORT_FIELDS[primary_label],
        primary_descending=primary_direction == "Descending",
        secondary_field=SORT_FIELDS[secondary_label],
        secondary_descending=secondary_direction == "Descending",
        page=st.session_state.warning_page,
        page_size=page_size,
    )
    if current_page.page != st.session_state.warning_page:
        st.session_state.warning_page = current_page.page

    export_df = pd.DataFrame(
        [
            {
                "Document A": item["doc_a"],
                "Document B": item["doc_b"],
                "Similarity": item["similarity"],
                "Severity": item["severity"],
            }
            for item in sorted_flags
        ]
    )

    # Generate Markdown Summary of all High & Medium warnings
    summary_flags = [
        _normalise_warning(flag)
        for flag in flags
        if _normalise_warning(flag)["severity"] in ("High", "Medium")
    ]
    if not summary_flags:
        markdown_text = "# 🔍 Plagiarism Report Summary\n\nNo High or Medium severity warnings found."
    else:
        markdown_lines = [
            "# 🔍 Plagiarism Report Summary\n",
            "The following document pairs have been flagged for high or medium similarity:\n",
        ]
        for idx, flag in enumerate(summary_flags, 1):
            matched_words = flag.get("matched_length", 0)
            markdown_lines.append(
                f"{idx}. **{flag['doc_a']}** ↔ **{flag['doc_b']}** — "
                f"**Similarity:** `{flag['similarity'] * 100:.1f}%` ({matched_words} words matched) | "
                f"**Severity:** `{flag['severity']}`"
            )
        markdown_text = "\n".join(markdown_lines)

    escaped_text = (
        markdown_text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\n", "\\n")
    )

    left, middle, right = st.columns([3, 2, 2])
    with left:
        if current_page.total_items:
            st.markdown(
                f"Showing **{current_page.start_index}–{current_page.end_index}** "
                f"of **{current_page.total_items}** matching warnings"
            )
        else:
            st.info("No warnings match the current search.")
    with middle:
        html_code = f"""
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
        </style>
        <button id="copy-btn" style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: white;
            color: #31333f;
            border: 1px solid #d6d6d8;
            padding: 0.35rem 0.75rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-weight: 400;
            font-size: 0.875rem;
            line-height: 1.6;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            width: 100%;
            height: 38px;
            user-select: none;
            box-sizing: border-box;
            transition: background-color 0.2s, color 0.2s, border-color 0.2s;
        " onmouseover="this.style.borderColor='#ff4b4b'; this.style.color='#ff4b4b'" onmouseout="this.style.borderColor='#d6d6d8'; this.style.color='#31333f'">
            📋 Copy Report Summary
        </button>
        <script>
            document.getElementById("copy-btn").addEventListener("click", function() {{
                const text = "{escaped_text}";
                const textArea = document.createElement("textarea");
                textArea.value = text;
                textArea.style.top = "0";
                textArea.style.left = "0";
                textArea.style.position = "fixed";
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                try {{
                    const successful = document.execCommand('copy');
                    if (successful) {{
                        const btn = document.getElementById("copy-btn");
                        btn.innerHTML = "✅ Copied!";
                        btn.style.borderColor = "#28a745";
                        btn.style.color = "#28a745";
                        setTimeout(function() {{
                            btn.innerHTML = "📋 Copy Report Summary";
                            btn.style.borderColor = "#d6d6d8";
                            btn.style.color = "#31333f";
                        }}, 2000);
                    }}
                }} catch (err) {{
                    console.error("Could not copy: ", err);
                }}
                document.body.removeChild(textArea);
            }});
        </script>
        """
        st.components.v1.html(html_code, height=45)
    with right:
        st.download_button(
            "⬇️ Download filtered report (CSV)",
            export_df.to_csv(index=False).encode("utf-8"),
            "plagiarism_warnings_filtered.csv",
            "text/csv",
            use_container_width=True,
            disabled=export_df.empty,
        )

    # ── Warning list container (#369) ────────────────────────────────
    # A stable `key` makes Streamlit attach a `st-key-warning_list_container`
    # class to this container's wrapping div, which theme.py's CSS targets
    # with a transition so re-filtered/re-sorted results animate smoothly
    # instead of snapping instantly.
    with st.container(key="warning_list_container"):
        for flag in current_page.items:
            tier = tier_from_severity_label(flag["severity"])
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    if _has_exact_match(flag["doc_a"], flag["doc_b"]):
                        exact_badge = " <span style='background-color: #E8F5E9; color: #2E7D32; border: 1px solid #2E7D32; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-left: 8px; vertical-align: middle;'>Exact Match</span>"
                        st.markdown(
                            f"**{flag['doc_a']}** ↔ **{flag['doc_b']}**{exact_badge}",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**{flag['doc_a']}** ↔ **{flag['doc_b']}**")

                    # Replaced the standard similarity text with your matched length display logic
                    matched_words = flag.get("matched_length", 0)
                    display_text = f"[{flag['similarity'] * 100:.1f}% Similarity | {matched_words} words matched]"
                    st.progress(
                        min(1.0, max(0.0, float(flag["similarity"]))),
                        text=display_text,
                    )

                    # Display AI probabilities if available
                    if ai_probabilities:
                        ai_a = ai_probabilities.get(flag["doc_a"], {}).get("overall", 0.0)
                        ai_b = ai_probabilities.get(flag["doc_b"], {}).get("overall", 0.0)
                        if ai_a > 0 or ai_b > 0:
                            st.caption(
                                f"🤖 AI Prob: {flag['doc_a']}: {ai_a:.1%} | "
                                f"{flag['doc_b']}: {ai_b:.1%}"
                            )
                with c2:
                    st.markdown(
                        f"<div style='text-align:right;'>{badge_html(tier, flag['severity'])}</div>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    if st.button("Dismiss", key=f"dismiss_{flag['doc_a']}_{flag['doc_b']}"):
                        add_false_positive(flag["doc_a"], flag["doc_b"])
                        st.rerun()

    if current_page.total_items == 0:
        return
    prev_col, page_col, next_col = st.columns([1, 2, 1])

    with prev_col:
        if st.button(
            "← Previous",
            use_container_width=True,
            disabled=current_page.page <= 1,
            key="warning_previous_page",
        ):
            st.session_state.warning_page = current_page.page - 1
            st.rerun()

    with page_col:
        selected_page = st.selectbox(
            "Page",
            list(range(1, current_page.total_pages + 1)),
            index=current_page.page - 1,
            key=f"warning_page_selector_{current_page.total_pages}",
            format_func=lambda value: f"Page {value} of {current_page.total_pages}",
            label_visibility="collapsed",
        )
        if selected_page != current_page.page:
            st.session_state.warning_page = selected_page
            st.rerun()

    with next_col:
        if st.button(
            "Next →",
            use_container_width=True,
            disabled=current_page.page >= current_page.total_pages,
            key="warning_next_page",
        ):
            st.session_state.warning_page = current_page.page + 1
            st.rerun()