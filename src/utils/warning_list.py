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

"""Search, multi-column sorting, and pagination for plagiarism warnings."""

from __future__ import annotations

import html
import re
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


def _truncate_search_query(search_query: Any) -> str:
    """Limit search input length to avoid expensive matching on oversized strings, safely casting numeric inputs."""
    if search_query is None:
        return ""
    if not isinstance(search_query, str):
        search_query = str(search_query)
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
            item
            for item in normalised
            if item.get("matched_length", 0) >= min_match_length
        ]

    predicate = matches_query_predicate(search_query)
    return [item for item in normalised if predicate(item)]


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


DEFAULT_COPY_BUTTON_ID = "copy-btn"

# An HTML id that is also safe to drop into a JavaScript string literal and a
# CSS-free ``getElementById`` lookup: letters, digits, hyphen, underscore.
_SAFE_ELEMENT_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")

# Characters that must not reach a JavaScript string literal verbatim. ``<`` is
# in the list because ``</script>`` inside a literal still closes the block for
# the HTML parser, which is how "it is only a string" becomes script execution.
_JS_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "'": "\\'",
    "`": "\\`",
    "$": "\\$",
    "\n": "\\n",
    "\r": "\\r",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
    "<": "\\u003C",
    ">": "\\u003E",
    "&": "\\u0026",
}


def sanitize_element_id(
    raw_id: Any,
    fallback: str = DEFAULT_COPY_BUTTON_ID,
) -> str:
    """Reduce *raw_id* to characters that are safe as an HTML id.

    The id is written into an HTML attribute *and* into a JavaScript string
    literal inside the same document. Escaping cannot serve both at once —
    the browser un-escapes the attribute but leaves the literal alone, so the
    two stop matching and the button silently dies. Restricting the character
    set instead keeps a single value valid in both places.

    Args:
        raw_id: Caller-supplied id. Any type; non-strings are stringified.
        fallback: Used when nothing survives sanitisation.

    Returns:
        A string of ``[A-Za-z0-9_-]`` only, never empty.

    Examples:
        >>> sanitize_element_id("copy_ca_3")
        'copy_ca_3'
        >>> sanitize_element_id('"><script>alert(1)</script><div id="')
        'scriptalert1scriptdivid'
        >>> sanitize_element_id("<<<>>>")
        'copy-btn'
    """
    if raw_id is None:
        return fallback

    cleaned = _SAFE_ELEMENT_ID_RE.sub("", str(raw_id))
    return cleaned or fallback


def escape_js_string(value: Any) -> str:
    """Escape *value* for use inside a double-quoted JavaScript string literal.

    Args:
        value: Any object; it is stringified first.

    Returns:
        The escaped text, safe to interpolate between two double quotes inside
        a ``<script>`` block.

    Examples:
        >>> escape_js_string('</script><script>alert(1)</script>')
        '\\u003C/script\\u003E\\u003Cscript\\u003Ealert(1)\\u003C/script\\u003E'
    """
    text = str(value)
    return "".join(_JS_STRING_ESCAPES.get(char, char) for char in text)


def render_copy_button(
    text_to_copy: str,
    button_id: str = DEFAULT_COPY_BUTTON_ID,
    copy_label: str = "📋 Copy",
    copied_label: str = "✅ Copied!",
    height: int = 45,
) -> None:
    """Render a clipboard button as an isolated Streamlit HTML component.

    Every caller-supplied value is neutralised for the context it lands in:
    ``button_id`` is reduced to a safe identifier, the labels are HTML-escaped
    where they appear as markup and JS-escaped where they are assigned through
    ``innerHTML``, and the copied text is JS-escaped.

    Args:
        text_to_copy: Text placed on the clipboard when the button is clicked.
        button_id: DOM id for the button. Sanitised; see
            :func:`sanitize_element_id`.
        copy_label: Button caption in its resting state.
        copied_label: Button caption shown for two seconds after a copy.
        height: Height in pixels of the embedded component.
    """
    safe_button_id = sanitize_element_id(button_id)

    # Labels appear twice: literally in the markup, and as a JavaScript string
    # written back through innerHTML. Those are two different contexts and each
    # needs its own escaping.
    safe_copy_label = html.escape(str(copy_label))
    safe_copied_label = html.escape(str(copied_label))
    js_copy_label = escape_js_string(safe_copy_label)
    js_copied_label = escape_js_string(safe_copied_label)

    escaped_text = escape_js_string(text_to_copy)
    html_code = f"""
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
    </style>
    <button id="{safe_button_id}" style="
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
        {safe_copy_label}
    </button>
    <script>
        document.getElementById("{safe_button_id}").addEventListener("click", function() {{
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
                    const btn = document.getElementById("{safe_button_id}");
                    btn.innerHTML = "{js_copied_label}";
                    btn.style.borderColor = "#28a745";
                    btn.style.color = "#28a745";
                    setTimeout(function() {{
                        btn.innerHTML = "{js_copy_label}";
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


def _chunked_docs_from_results(results: Any) -> Mapping[str, Sequence[str]]:
    """Pull the chunk mapping out of whatever shape ``analysis_results`` has.

    The pipeline result has been a plain tuple, a ``NamedTuple`` and a small
    result object over the life of this module, and session state can still be
    holding any of them after a rerun. Reading ``results[1]`` only works for
    the first two; an object exposing ``chunked_docs`` as a plain attribute
    raises ``TypeError: ... is not subscriptable``.

    Args:
        results: The value stored in ``st.session_state.analysis_results``.

    Returns:
        The document-to-chunks mapping, or an empty mapping when the value
        does not carry one.
    """
    chunked_docs = getattr(results, "chunked_docs", None)

    if chunked_docs is None:
        try:
            chunked_docs = results[1]
        except (TypeError, IndexError, KeyError):
            return {}

    if not isinstance(chunked_docs, Mapping):
        return {}

    return chunked_docs


def _has_exact_match(doc_a: str, doc_b: str) -> bool:
    """Check if two documents share at least one exact matching chunk (ignoring whitespace)."""
    if (
        "analysis_results" not in st.session_state
        or st.session_state.analysis_results is None
    ):
        return False
    chunked_docs = _chunked_docs_from_results(st.session_state.analysis_results)
    chunks_a = chunked_docs.get(doc_a, [])
    chunks_b = chunked_docs.get(doc_b, [])

    # Normalize chunks by removing all whitespace
    norm_a = {
        "".join((c.text if hasattr(c, "text") else c).split())
        for c in chunks_a
        if (c.text if hasattr(c, "text") else c).strip()
    }
    norm_b = {
        "".join((c.text if hasattr(c, "text") else c).split())
        for c in chunks_b
        if (c.text if hasattr(c, "text") else c).strip()
    }

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
                    pct=f"{threshold * 100:.0f}"
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
        nf
        for flag in flags
        if (nf := _normalise_warning(flag))["severity"] in ("High", "Medium")
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
            copied_label="✅ Copied!",
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


def matches_query_predicate(search_query: str) -> Callable[[Mapping[str, Any]], bool]:
    """
    Return a predicate that checks whether a warning matches the given search query.
    """
    query = _truncate_search_query(search_query).casefold()

    def predicate(flag: Mapping[str, Any]) -> bool:
        if not query:
            return True
        doc_a = str(flag.get("doc_a", "")).casefold()
        doc_b = str(flag.get("doc_b", "")).casefold()
        if query in doc_a or query in doc_b:
            return True
        if fuzz is not None:
            score_a = max(
                fuzz.partial_ratio(query, doc_a), fuzz.token_set_ratio(query, doc_a)
            )
            score_b = max(
                fuzz.partial_ratio(query, doc_b), fuzz.token_set_ratio(query, doc_b)
            )
            return score_a >= FUZZY_THRESHOLD or score_b >= FUZZY_THRESHOLD
        return False

    return predicate
