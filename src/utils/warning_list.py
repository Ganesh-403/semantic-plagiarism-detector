"""Search, multi-column sorting, and pagination for plagiarism warnings."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from app.theme import badge_html, tier_from_severity_label
from src.core.config import normalize_severity_label, severity_from_score, severity_rank
from src.db.incidents import _normalise_pair, add_false_positive, get_false_positives
from src.i18n.translator import get_text
from src.utils.pagination import PaginationPage, paginate_items

try:
    from thefuzz import fuzz
except ImportError:
    try:
        from fuzzywuzzy import fuzz  # type: ignore[import-untyped,reportMissingImports]
    except ImportError:
        fuzz = None
FUZZY_THRESHOLD = 75
MAX_SEARCH_QUERY_LENGTH = 200

_SORT_KEYS = {
    "warn_sort_similarity": "similarity",
    "warn_sort_doc_a": "doc_a",
    "warn_sort_doc_b": "doc_b",
    "warn_sort_severity": "severity_rank",
}


def _sort_display_names(lang_code: str) -> dict[str, str]:
    return {get_text(k, lang=lang_code): v for k, v in _SORT_KEYS.items()}


WarningPage = PaginationPage[dict[str, Any]]


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


def _truncate_search_query(search_query: str) -> str:
    """Limit search input length to avoid expensive matching on oversized strings."""
    if not isinstance(search_query, str):
        return ""
    return search_query[:MAX_SEARCH_QUERY_LENGTH].strip()


def filter_warnings(
    warnings: Iterable[Mapping[str, Any]],
    search_query: str = "",
    min_match_length: int = 0,
) -> list[dict[str, Any]]:
    """Filter normalized warnings using functional predicate matching."""
    normalised = [_normalise_warning(item) for item in warnings]

    if min_match_length > 0:
        normalised = [
            item for item in normalised if item.get("matched_length", 0) >= min_match_length
        ]

    query = _truncate_search_query(search_query).casefold()
    if not query:
        return normalised

    filtered = []
    for item in normalised:
        doc_a = item["doc_a"].casefold()
        doc_b = item["doc_b"].casefold()

        if query in doc_a or query in doc_b:
            filtered.append(item)
            continue

        if fuzz is not None:
            score_a = max(fuzz.partial_ratio(query, doc_a), fuzz.token_set_ratio(query, doc_a))
            score_b = max(fuzz.partial_ratio(query, doc_b), fuzz.token_set_ratio(query, doc_b))
            if score_a >= FUZZY_THRESHOLD or score_b >= FUZZY_THRESHOLD:
                filtered.append(item)

    return filtered


def build_key_extractor(field: str) -> Callable[[Mapping[str, Any]], Any]:
    """Return a key extraction function suitable for sorting warning items."""
    def extract_key(item: Mapping[str, Any]) -> Any:
        val = item.get(field, "")
        return val.casefold() if isinstance(val, str) else val
    return extract_key


def sort_warnings(
    warnings: Iterable[Mapping[str, Any]],
    *,
    primary_field: str = "similarity",
    primary_descending: bool = True,
    secondary_field: str = "doc_a",
    secondary_descending: bool = False,
) -> list[dict[str, Any]]:
    """Sort warning items using secondary and primary sorting keys."""
    items = [_normalise_warning(item) for item in warnings]
    allowed = {"similarity", "doc_a", "doc_b", "severity_rank"}

    p_field = primary_field if primary_field in allowed else "similarity"
    s_field = secondary_field if secondary_field in allowed else "doc_a"

    items.sort(key=build_key_extractor(s_field), reverse=secondary_descending)
    items.sort(key=build_key_extractor(p_field), reverse=primary_descending)
    return items


def paginate_warnings(
    warnings: Sequence[Mapping[str, Any]],
    *,
    page: int = 1,
    page_size: int = 10,
) -> WarningPage:
    """Return a clamped page of warning dictionaries."""
    normalized_warnings = [dict(item) for item in warnings]
    return paginate_items(
        normalized_warnings,
        page=page,
        page_size=page_size,
        max_page_size=100,
    )


def prepare_warning_page(
    warnings: Iterable[Mapping[str, Any]],
    *,
    search_query: str = "",
    min_match_length: int = 0,
    primary_field: str = "similarity",
    primary_descending: bool = True,
    secondary_field: str = "doc_a",
    secondary_descending: bool = False,
    page: int = 1,
    page_size: int = 10,
) -> tuple[list[dict[str, Any]], WarningPage]:
    filtered = filter_warnings(
        warnings, search_query, min_match_length=min_match_length
    )
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


def render_copy_button(text_to_copy: str, button_id: str = "copy-btn", copy_label: str = "📋 Copy", copied_label: str = "✅ Copied!", height: int = 45) -> None:
    escaped_text = (
        text_to_copy.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\n", "\\n")
    )
    html_code = f"""
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
    </style>
    <button id="{button_id}" style="
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
        {copy_label}
    </button>
    <script>
        document.getElementById("{button_id}").addEventListener("click", function() {{
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
                    const btn = document.getElementById("{button_id}");
                    btn.innerHTML = "{copied_label}";
                    btn.style.borderColor = "#28a745";
                    btn.style.color = "#28a745";
                    setTimeout(function() {{
                        btn.innerHTML = "{copy_label}";
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
    st.components.v1.html(html_code, height=height)


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


def render_compact_warning_row(flag: Mapping[str, Any]) -> None:
    """
    Render warning in compact single-line format.
    """

    doc_a = flag["doc_a"]
    doc_b = flag["doc_b"]

    tier = tier_from_severity_label(flag["severity"])
    similarity = flag["similarity"] * 100

    col1, col2, col3, col4 = st.columns([5, 1, 1, 0.5])

    with col1:
        exact_badge = ""

        if _has_exact_match(doc_a, doc_b):
            exact_badge = (
                " <span style='color:#2E7D32;font-weight:bold;'>✓ Exact</span>"
            )

        st.markdown(
            f"📄 **{doc_a}** ↔ **{doc_b}**{exact_badge}",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(f"**{similarity:.1f}%**")

    with col3:
        st.markdown(
            badge_html(tier, flag["severity"]),
            unsafe_allow_html=True,
        )

    with col4:
        if st.button(
            "❌",
            key=f"compact_dismiss_{doc_a}_{doc_b}",
            help="Dismiss warning",
        ):
            add_false_positive(doc_a, doc_b)
            st.rerun()


def render_warning_controls(
    flags: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    ai_probabilities: dict[str, dict[str, Any]] | None = None,
    lang_code: str = "en",
    expanded: bool = False,
) -> None:
    if "warning_page" not in st.session_state:
        st.session_state.warning_page = 1
    if "compact_view" not in st.session_state:
        st.session_state.compact_view = False

    from src.core.config import DEFAULT_THRESHOLDS

    st.caption(
        get_text("warn_pairs_caption", lang=lang_code).format(
            threshold=f"{threshold:.2f}"
        )
    )

    active_filters = []
    if abs(threshold - DEFAULT_THRESHOLDS.plagiarism) > 0.001:
        active_filters.append(
            {
                "key": "clear_threshold",
                "label": get_text("warn_filter_threshold", lang=lang_code).format(
                    pct=f"{threshold*100:.0f}"
                ),
                "action": "threshold",
            }
        )

    if st.session_state.get("compact_view", False):
        active_filters.append(
            {
                "key": "clear_compact_view",
                "label": "Compact View \u24e7",
                "action": "compact_view",
            }
        )

    if st.session_state.get("hide_low_severity", False):
        active_filters.append(
            {
                "key": "clear_hide_low_severity",
                "label": get_text("warn_filter_severity", lang=lang_code),
                "action": "hide_low_severity",
            }
        )

    warning_search = _truncate_search_query(st.session_state.get("warning_search", ""))
    if warning_search:
        display_search = (
            warning_search if len(warning_search) <= 15 else warning_search[:12] + "..."
        )
        active_filters.append(
            {
                "key": "clear_warning_search",
                "label": get_text("warn_filter_search", lang=lang_code).format(
                    query=display_search
                ),
                "action": "warning_search",
            }
        )

    selected_document_id = st.session_state.get("selected_document_id")
    if selected_document_id:
        display_doc = (
            selected_document_id
            if len(selected_document_id) <= 15
            else selected_document_id[:12] + "..."
        )
        active_filters.append(
            {
                "key": "clear_document_filter",
                "label": get_text("warn_filter_document", lang=lang_code).format(
                    doc=display_doc
                ),
                "action": "selected_document_id",
            }
        )

    selected_class = st.session_state.get("class_filter_selectbox", "All Classes")
    if selected_class and selected_class != "All Classes":
        display_class = (
            selected_class if len(selected_class) <= 15 else selected_class[:12] + "..."
        )
        active_filters.append(
            {
                "key": "clear_class_filter",
                "label": get_text("warn_filter_class", lang=lang_code).format(
                    class_name=display_class
                ),
                "action": "class_filter",
            }
        )

    min_match_len_val = st.session_state.get("warning_min_match_length", 0)
    if min_match_len_val > 0:
        active_filters.append(
            {
                "key": "clear_min_match_length",
                "label": get_text("warn_filter_min_words", lang=lang_code).format(
                    count=min_match_len_val
                ),
                "action": "min_match_length",
            }
        )

    if active_filters:
        st.markdown(
            """<style>
            /* Make buttons look like small pills */
            div[data-testid="column"] button {
                border-radius: 16px !important;
                padding: 2px 12px !important;
                min-height: 28px !important;
                height: 28px !important;
                font-size: 13px !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        cols = st.columns([len(f["label"]) for f in active_filters] + [20])
        for idx, f in enumerate(active_filters):
            with cols[idx]:
                if st.button(f["label"], key=f["key"]):
                    if f["action"] == "threshold":
                        st.session_state.threshold = DEFAULT_THRESHOLDS.plagiarism
                        st.session_state.threshold_slider = (
                            DEFAULT_THRESHOLDS.plagiarism
                        )
                        if "last_seen_threshold_query" in st.session_state:
                            del st.session_state["last_seen_threshold_query"]
                        # In Streamlit >= 1.30, st.query_params is dict-like
                        if "threshold" in st.query_params:
                            del st.query_params["threshold"]
                    elif f["action"] == "hide_low_severity":
                        st.session_state.hide_low_severity = False
                    elif f["action"] == "warning_search":
                        st.session_state.warning_search = ""
                    elif f["action"] == "selected_document_id":
                        st.session_state.selected_document_id = None
                    elif f["action"] == "class_filter":
                        st.session_state.class_filter_selectbox = "All Classes"
                    elif f["action"] == "min_match_length":
                        st.session_state.warning_min_match_length = 0
                    elif f["action"] == "compact_view":
                        st.session_state.compact_view = False
                    st.rerun()

    dismissed_pairs = get_false_positives()
    filtered_flags = [
        f
        for f in flags
        if _normalise_pair(f["doc_a"], f["doc_b"]) not in dismissed_pairs
    ]

    if not filtered_flags:
        st.success(get_text("warn_no_suspicious", lang=lang_code))
        return

    search_col, toggle_col, compact_col, size_col = st.columns([3, 2, 2, 1])

    with search_col:
        search_query = st.text_input(
            get_text("warn_search_label", lang=lang_code),
            placeholder=get_text("warn_search_placeholder", lang=lang_code),
            key="warning_search",
            on_change=_reset_page,
        )
        search_query = _truncate_search_query(search_query)

    with toggle_col:
        hide_low_severity = st.checkbox(
            get_text("warn_hide_low_severity", lang=lang_code),
            key="hide_low_severity",
        )
    with compact_col:
        compact_view = st.checkbox(
            "Compact View",
            key="compact_view",
            help="Show warnings as compact single-line rows",
            on_change=_reset_page,
        )
    with size_col:
        page_size = st.selectbox(
            get_text("warn_per_page", lang=lang_code),
            [10, 25, 50],
            key="warning_page_size",
            on_change=_reset_page,
        )

    min_match_length = st.slider(
        get_text("warn_min_match_length", lang=lang_code),
        min_value=0,
        max_value=250,
        value=0,
        step=5,
        key="warning_min_match_length",
        on_change=_reset_page,
    )

    sort_fields = _sort_display_names(lang_code)
    _desc_text = get_text("warn_descending", lang=lang_code)
    _asc_text = get_text("warn_ascending", lang=lang_code)
    p_dir = st.session_state.get("warning_primary_direction", _desc_text)
    s_dir = st.session_state.get("warning_secondary_direction", _asc_text)

    p_arrow = "\u25bc" if p_dir == _desc_text else "\u25b2"
    s_arrow = "\u25b2" if s_dir == _asc_text else "\u25bc"

    p1, d1, p2, d2 = st.columns([2, 1, 2, 1])

    with p1:
        primary_label = st.selectbox(
            f"{get_text('warn_primary_sort', lang=lang_code)} {p_arrow}",
            list(sort_fields),
            key="warning_primary_sort",
            on_change=_reset_page,
        )

    with d1:
        primary_direction = st.selectbox(
            get_text("warn_direction", lang=lang_code),
            [
                get_text("warn_descending", lang=lang_code),
                get_text("warn_ascending", lang=lang_code),
            ],
            key="warning_primary_direction",
            on_change=_reset_page,
        )

    with p2:
        secondary_label = st.selectbox(
            f"{get_text('warn_secondary_sort', lang=lang_code)} {s_arrow}",
            list(sort_fields),
            index=1,
            key="warning_secondary_sort",
            on_change=_reset_page,
        )

    with d2:
        secondary_direction = st.selectbox(
            get_text("warn_direction", lang=lang_code),
            [
                get_text("warn_ascending", lang=lang_code),
                get_text("warn_descending", lang=lang_code),
            ],
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
        min_match_length=min_match_length,
        primary_field=sort_fields[primary_label],
        primary_descending=primary_direction == _desc_text,
        secondary_field=sort_fields[secondary_label],
        secondary_descending=secondary_direction == _desc_text,
        page=st.session_state.warning_page,
        page_size=page_size,
    )
    if current_page.page != st.session_state.warning_page:
        st.session_state.warning_page = current_page.page

    export_df = pd.DataFrame(
        [
            {
                get_text("warn_col_doc_a", lang=lang_code): item["doc_a"],
                get_text("warn_col_doc_b", lang=lang_code): item["doc_b"],
                get_text("warn_col_similarity", lang=lang_code): item["similarity"],
                get_text("warn_col_severity", lang=lang_code): item["severity"],
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
        markdown_text = get_text("warn_no_summary", lang=lang_code)
    else:
        markdown_lines = [
            get_text("warn_summary_title", lang=lang_code) + "\n",
            get_text("warn_summary_desc", lang=lang_code) + "\n",
        ]
        for idx, flag in enumerate(summary_flags, 1):
            matched_words = flag.get("matched_length", 0)
            sim_label = get_text("warn_summary_similarity_label", lang=lang_code)
            sev_label = get_text("warn_summary_severity_label", lang=lang_code)
            words_text = get_text("warn_summary_words_matched", lang=lang_code).format(
                count=matched_words
            )
            markdown_lines.append(
                f"{idx}. **{flag['doc_a']}** ↔ **{flag['doc_b']}** — "
                f"{sim_label} `{flag['similarity'] * 100:.1f}%` ({words_text}) | "
                f"{sev_label} `{flag['severity']}`"
            )
        markdown_text = "\n".join(markdown_lines)



    left, middle, right = st.columns([3, 2, 2])
    with left:
        if current_page.total_items:
            st.markdown(
                get_text("warn_showing", lang=lang_code).format(
                    start=current_page.start_index,
                    end=current_page.end_index,
                    total=current_page.total_items,
                )
            )
        else:
            st.info(get_text("warn_no_match", lang=lang_code))
    with middle:
        render_copy_button(
            text_to_copy=markdown_text,
            button_id="copy-summary-btn",
            copy_label="📋 Copy Summary",
            copied_label="✅ Copied!"
        )
    with right:
        st.download_button(
            get_text("warn_download_csv", lang=lang_code),
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

            if compact_view:
                render_compact_warning_row(flag)
                st.markdown(
                    "<hr style='margin:4px 0;border:0;border-top:1px solid #eee;'>",
                    unsafe_allow_html=True,
                )

            else:
                tier = tier_from_severity_label(flag["severity"])

                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        exact_match_label = get_text("warn_exact_match", lang=lang_code)
                        if _has_exact_match(flag["doc_a"], flag["doc_b"]):
                            exact_badge = f" <span style='background-color: #E8F5E9; color: #2E7D32; border: 1px solid #2E7D32; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-left: 8px; vertical-align: middle;'>{exact_match_label}</span>"
                            st.markdown(
                                f"**{flag['doc_a']}** ↔ **{flag['doc_b']}**{exact_badge}",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f"**{flag['doc_a']}** ↔ **{flag['doc_b']}**")

                        # Replaced the standard similarity text with your matched length display logic
                        matched_words = flag.get("matched_length", 0)
                        display_text = get_text(
                            "warn_similarity_progress", lang=lang_code
                        ).format(
                            pct=f"{flag['similarity'] * 100:.1f}",
                            words=matched_words,
                        )
                        st.progress(
                            min(1.0, max(0.0, float(flag["similarity"]))),
                            text=display_text,
                        )

                        # Display AI probabilities if available
                        if ai_probabilities:
                            ai_a = ai_probabilities.get(flag["doc_a"], {}).get(
                                "overall", 0.0
                            )
                            ai_b = ai_probabilities.get(flag["doc_b"], {}).get(
                                "overall", 0.0
                            )
                            if ai_a > 0 or ai_b > 0:
                                st.caption(
                                    get_text("warn_ai_prob", lang=lang_code).format(
                                        doc_a=flag["doc_a"],
                                        ai_a=ai_a,
                                        doc_b=flag["doc_b"],
                                        ai_b=ai_b,
                                    )
                                )
                    with c2:
                        st.markdown(
                            f"<div style='text-align:right;'>{badge_html(tier, flag['severity'])}</div>",
                            unsafe_allow_html=True,
                        )
                    with c3:
                        if st.button(
                            get_text("warn_dismiss", lang=lang_code),
                            key=f"dismiss_{flag['doc_a']}_{flag['doc_b']}",
                        ):
                            add_false_positive(flag["doc_a"], flag["doc_b"])
                            st.rerun()

    if current_page.total_items == 0:
        return
    prev_col, page_col, next_col = st.columns([1, 2, 1])

    with prev_col:
        if st.button(
            get_text("warn_prev", lang=lang_code),
            use_container_width=True,
            disabled=current_page.page <= 1,
            key="warning_previous_page",
        ):
            st.session_state.warning_page = current_page.page - 1
            st.rerun()

    with page_col:
        selected_page = st.selectbox(
            get_text("warn_page", lang=lang_code),
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
            get_text("warn_next", lang=lang_code),
            use_container_width=True,
            disabled=current_page.page >= current_page.total_pages,
            key="warning_next_page",
        ):
            st.session_state.warning_page = current_page.page + 1
            st.rerun()
def matches_query_predicate(flag: dict, search_query: str) -> bool:
    """
    Check if a flagged incident matches a search query across document names or text snippets.
    """
    if not search_query or not search_query.strip():
        return True

    query = search_query.strip().lower()
    doc_a = str(flag.get("doc_a", "")).lower()
    doc_b = str(flag.get("doc_b", "")).lower()
    snippet_a = str(flag.get("snippet_a", "")).lower()
    snippet_b = str(flag.get("snippet_b", "")).lower()

    return (
        query in doc_a
        or query in doc_b
        or query in snippet_a
        or query in snippet_b
    )
