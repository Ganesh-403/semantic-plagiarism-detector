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

from unittest.mock import patch

from src.utils.warning_list import (
    build_key_extractor,
    escape_js_string,
    filter_warnings,
    matches_query_predicate,
    paginate_warnings,
    prepare_warning_page,
    render_copy_button,
    sanitize_element_id,
    sort_warnings,
)

WARNINGS = [
    {"doc_a": "Zeta.pdf", "doc_b": "Alpha.pdf", "similarity": 0.91, "severity": "High"},
    {
        "doc_a": "Beta.pdf",
        "doc_b": "Gamma.pdf",
        "similarity": 0.78,
        "severity": "Medium",
    },
    {
        "doc_a": "Alpha.pdf",
        "doc_b": "Delta.pdf",
        "similarity": 0.91,
        "severity": "High",
    },
    {
        "doc_a": "Notes.pdf",
        "doc_b": "Essay.pdf",
        "similarity": 0.81,
        "severity": "Medium",
    },
]


def test_matches_query_predicate():
    predicate_alpha = matches_query_predicate("alpha")
    predicate_empty = matches_query_predicate("   ")

    assert predicate_alpha(WARNINGS[0]) is True  # doc_b matches
    assert predicate_alpha(WARNINGS[1]) is False  # no match
    assert predicate_alpha(WARNINGS[2]) is True  # doc_a matches
    assert predicate_empty(WARNINGS[1]) is True  # empty query matches all


def test_build_key_extractor():
    extractor_doc_a = build_key_extractor("doc_a")
    extractor_sim = build_key_extractor("similarity")

    assert extractor_doc_a(WARNINGS[0]) == "zeta.pdf"
    assert extractor_sim(WARNINGS[0]) == 0.91


def test_search_matches_either_document_case_insensitively():
    results = filter_warnings(WARNINGS, "ALPHA")
    assert len(results) == 2


def test_empty_search_returns_everything():
    assert len(filter_warnings(WARNINGS, " ")) == 4


def test_search_query_is_truncated_to_max_length():
    # Truncation behaviour: 201-char and 200-char queries must produce identical results
    truncated = filter_warnings(WARNINGS, "a" * 201)
    assert truncated == filter_warnings(WARNINGS, "a" * 200)


def test_fuzzy_search_handles_minor_typos():
    try:
        from thefuzz import fuzz  # noqa: F401
    except ImportError:
        try:
            from fuzzywuzzy import fuzz  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("fuzzy library not installed")

    # "Alpaha" is a typo for "Alpha"
    results = filter_warnings(WARNINGS, "Alpaha")
    assert len(results) == 2

    # "Ztaa" is a typo for "Zeta"
    results_zeta = filter_warnings(WARNINGS, "Ztaa")
    assert len(results_zeta) == 1
    assert results_zeta[0]["doc_a"] == "Zeta.pdf"


def test_multi_column_sorting():
    results = sort_warnings(
        WARNINGS,
        primary_field="similarity",
        primary_descending=True,
        secondary_field="doc_a",
        secondary_descending=False,
    )
    assert [item["similarity"] for item in results] == [0.91, 0.91, 0.81, 0.78]
    assert results[0]["doc_a"] == "Alpha.pdf"
    assert results[1]["doc_a"] == "Zeta.pdf"


def test_multi_column_sorting_both_descending():
    results = sort_warnings(
        WARNINGS,
        primary_field="similarity",
        primary_descending=True,
        secondary_field="doc_a",
        secondary_descending=True,
    )
    assert [item["similarity"] for item in results] == [0.91, 0.91, 0.81, 0.78]
    # Both descending: among the two 0.91 items, doc_a descending → Zeta before Alpha
    assert results[0]["doc_a"] == "Zeta.pdf"
    assert results[1]["doc_a"] == "Alpha.pdf"


def test_multi_column_sorting_both_ascending():
    results = sort_warnings(
        WARNINGS,
        primary_field="similarity",
        primary_descending=False,
        secondary_field="doc_a",
        secondary_descending=False,
    )
    assert [item["similarity"] for item in results] == [0.78, 0.81, 0.91, 0.91]
    # Both ascending: among the two 0.91 items, doc_a ascending → Alpha before Zeta
    assert results[2]["doc_a"] == "Alpha.pdf"
    assert results[3]["doc_a"] == "Zeta.pdf"


def test_filename_sorting():
    results = sort_warnings(
        WARNINGS,
        primary_field="doc_a",
        primary_descending=False,
    )
    assert [item["doc_a"] for item in results] == [
        "Alpha.pdf",
        "Beta.pdf",
        "Notes.pdf",
        "Zeta.pdf",
    ]


def test_pagination_and_page_clamping():
    warnings = [
        {
            "doc_a": f"A-{i}.pdf",
            "doc_b": f"B-{i}.pdf",
            "similarity": 0.8,
            "severity": "Medium",
        }
        for i in range(23)
    ]
    page_two = paginate_warnings(warnings, page=2, page_size=10)
    final_page = paginate_warnings(warnings, page=99, page_size=10)

    assert len(page_two.items) == 10
    assert page_two.start_index == 11
    assert page_two.end_index == 20
    assert final_page.page == 3
    assert len(final_page.items) == 3


def test_filtering_occurs_before_pagination():
    warnings = [
        {
            "doc_a": f"target-{i}.pdf" if i < 12 else f"other-{i}.pdf",
            "doc_b": "reference.pdf",
            "similarity": 0.7 + i / 100,
            "severity": "Medium",
        }
        for i in range(20)
    ]

    filtered, page = prepare_warning_page(
        warnings,
        search_query="target",
        page=2,
        page_size=10,
    )
    assert len(filtered) == 12
    assert len(page.items) == 2
    assert page.total_pages == 2


def test_filter_warnings_by_minimum_match_length():
    warnings = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.8,
            "severity": "Medium",
            "matched_length": 5,
        },
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc3.pdf",
            "similarity": 0.85,
            "severity": "High",
            "matched_length": 150,
        },
        {
            "doc_a": "doc2.pdf",
            "doc_b": "doc3.pdf",
            "similarity": 0.75,
            "severity": "Medium",
            "matched_length": 50,
        },
    ]

    # Filter with min_match_length = 50 -> should exclude the 5-word match
    filtered = filter_warnings(warnings, min_match_length=50)
    assert len(filtered) == 2
    assert all(item["matched_length"] >= 50 for item in filtered)

    # Filter with min_match_length = 200 -> should exclude all matches
    filtered_none = filter_warnings(warnings, min_match_length=200)
    assert len(filtered_none) == 0

    # Filter routing in prepare_warning_page
    sorted_items, page = prepare_warning_page(warnings, min_match_length=50)
    assert len(sorted_items) == 2
    assert page.total_items == 2


def test_page_size_clamping_to_max_100():
    """Verify that a page_size parameter larger than 100 is clamped to 100."""
    warnings = [
        {
            "doc_a": f"A-{i}.pdf",
            "doc_b": f"B-{i}.pdf",
            "similarity": 0.8,
            "severity": "Medium",
        }
        for i in range(150)
    ]
    # Request a page size of 200
    page = paginate_warnings(warnings, page=1, page_size=200)
    # The safe_page_size must be clamped to 100
    assert page.page_size == 100
    assert len(page.items) == 100
    assert page.total_pages == 2


def test_has_exact_match_no_results():
    """Verify that _has_exact_match returns False if analysis_results is missing from session state."""
    import streamlit as st

    from src.utils.warning_list import _has_exact_match

    # Ensure analysis_results is not in session state
    if "analysis_results" in st.session_state:
        del st.session_state["analysis_results"]

    assert _has_exact_match("doc_a.pdf", "doc_b.pdf") is False


def test_has_exact_match_with_matching_tuple_results():
    """Verify that _has_exact_match works with legacy tuple format where index 1 is chunked_docs."""
    import streamlit as st

    from src.utils.warning_list import _has_exact_match

    chunked_docs = {
        "doc_a.pdf": ["hello world", "some other chunk"],
        "doc_b.pdf": ["hello world", "different chunk"],
    }
    legacy_results = (None, chunked_docs, None, None, None, None, None, None, None)
    st.session_state.analysis_results = legacy_results

    assert _has_exact_match("doc_a.pdf", "doc_b.pdf") is True


def test_has_exact_match_with_non_matching_tuple_results():
    """Verify that _has_exact_match returns False when no chunks match."""
    import streamlit as st

    from src.utils.warning_list import _has_exact_match

    chunked_docs = {"doc_a.pdf": ["hello world"], "doc_b.pdf": ["different chunk"]}
    legacy_results = (None, chunked_docs, None, None, None, None, None, None, None)
    st.session_state.analysis_results = legacy_results

    assert _has_exact_match("doc_a.pdf", "doc_b.pdf") is False


def test_has_exact_match_with_named_tuple_results():
    """Verify that _has_exact_match works with NamedTuple format, accessing chunked_docs attribute."""
    from collections import namedtuple

    import streamlit as st

    from src.utils.warning_list import _has_exact_match

    MockPipelineResult = namedtuple("MockPipelineResult", ["raw_texts", "chunked_docs"])
    chunked_docs = {
        "doc_a.pdf": ["exact match chunk"],
        "doc_b.pdf": ["exact match chunk"],
    }
    named_results = MockPipelineResult(raw_texts={}, chunked_docs=chunked_docs)
    st.session_state.analysis_results = named_results

    assert _has_exact_match("doc_a.pdf", "doc_b.pdf") is True


def test_has_exact_match_with_pure_attribute():
    """Verify that _has_exact_match works with an object that only has chunked_docs attribute."""
    import streamlit as st

    from src.utils.warning_list import _has_exact_match

    class MockNamedTuple:
        def __init__(self, chunked_docs):
            self.chunked_docs = chunked_docs

    chunked_docs = {"doc_a.pdf": ["exact match"], "doc_b.pdf": ["exact match"]}
    st.session_state.analysis_results = MockNamedTuple(chunked_docs)

    assert _has_exact_match("doc_a.pdf", "doc_b.pdf") is True


def _render(**kwargs) -> str:
    """Render the copy button and return the HTML handed to Streamlit."""
    with patch("streamlit.components.v1.html") as mock_html:
        render_copy_button(**kwargs)

        assert mock_html.called
        return mock_html.call_args[0][0]


def test_render_copy_button_xss_sanitization():
    """Verify that button_id is properly sanitized to prevent XSS.

    The original assertion expected the id to be HTML-escaped. Escaping is
    the wrong tool here: the id is written into an HTML attribute *and* into
    a JavaScript string literal in the same document, and the browser
    un-escapes only the first, so an escaped id no longer matches its own
    getElementById lookup. The id is reduced to a safe character set instead,
    which is valid in both contexts.
    """
    malicious_id = '"><script>alert(1)</script><div id="'

    rendered_html = _render(text_to_copy="Sample text", button_id=malicious_id)

    # No unescaped/raw markup from button_id survives.
    assert 'id=""><script>alert(1)</script>' not in rendered_html
    assert "alert(1)" not in rendered_html
    assert "<script>alert" not in rendered_html

    # The button and its script agree on one safe id.
    assert 'id="scriptalert1scriptdivid"' in rendered_html
    assert 'getElementById("scriptalert1scriptdivid")' in rendered_html


def test_render_copy_button_escapes_copy_label():
    """copy_label lands in the button body and in a JS innerHTML assignment."""
    rendered_html = _render(
        text_to_copy="Sample text",
        copy_label="<img src=x onerror=alert(1)>",
    )

    # The payload survives as inert text, never as a live tag.
    assert "<img src=x" not in rendered_html
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered_html


def test_render_copy_button_escapes_copied_label():
    """copied_label is the more dangerous one: it is written via innerHTML."""
    rendered_html = _render(
        text_to_copy="Sample text",
        copied_label="</script><script>alert(document.cookie)</script>",
    )

    # Only the component's own script block remains; the payload contributed
    # no tag of its own.
    assert rendered_html.count("<script>") == 1
    assert rendered_html.count("</script>") == 1

    # What is left of the payload is an inert JS string literal: every angle
    # bracket and ampersand has been replaced by a \\u escape, so innerHTML
    # renders it as visible text rather than parsing it as markup.
    assert "\\u0026lt;/script\\u0026gt;" in rendered_html


def test_render_copy_button_escapes_text_to_copy():
    """The copied text closes neither the JS literal nor the script block."""
    rendered_html = _render(
        text_to_copy='x"; alert(1); var y = "</script>',
    )

    assert '"; alert(1); var y = "' not in rendered_html
    assert rendered_html.count("</script>") == 1


def test_render_copy_button_keeps_ordinary_ids_intact():
    """Sanitisation must not disturb the ids the app actually passes."""
    rendered_html = _render(text_to_copy="Sample text", button_id="copy_ca_3")

    assert 'id="copy_ca_3"' in rendered_html
    assert 'getElementById("copy_ca_3")' in rendered_html


def test_render_copy_button_falls_back_when_id_is_all_unsafe():
    """An id with nothing safe left in it must not produce id="".

    An empty id would make every button on the page collide on
    getElementById(""), so the first click would drive the wrong button.
    """
    rendered_html = _render(text_to_copy="Sample text", button_id="<<<>>>")

    assert 'id="copy-btn"' in rendered_html
    assert 'id=""' not in rendered_html


def test_render_copy_button_emoji_labels_survive():
    """The default labels are emoji; escaping must leave them readable."""
    rendered_html = _render(text_to_copy="Sample text")

    assert "📋 Copy" in rendered_html
    assert "✅ Copied!" in rendered_html


class TestSanitizeElementId:
    """Unit coverage for the id sanitiser itself."""

    def test_alphanumeric_and_separators_pass_through(self):
        assert sanitize_element_id("copy-btn_2") == "copy-btn_2"

    def test_quotes_and_angle_brackets_are_dropped(self):
        assert sanitize_element_id('a"b<c>d') == "abcd"

    def test_whitespace_is_dropped(self):
        assert sanitize_element_id("copy me") == "copyme"

    def test_empty_result_uses_the_fallback(self):
        assert sanitize_element_id("!!!") == "copy-btn"

    def test_none_uses_the_fallback(self):
        assert sanitize_element_id(None) == "copy-btn"

    def test_custom_fallback_is_honoured(self):
        assert sanitize_element_id("###", fallback="snippet") == "snippet"

    def test_non_string_input_is_stringified(self):
        assert sanitize_element_id(42) == "42"


class TestEscapeJsString:
    """Unit coverage for the JavaScript string-literal escaper."""

    def test_plain_text_is_unchanged(self):
        assert escape_js_string("hello world") == "hello world"

    def test_double_quote_is_escaped(self):
        assert escape_js_string('say "hi"') == 'say \\"hi\\"'

    def test_backslash_is_escaped_before_anything_else(self):
        assert escape_js_string("C:\\path") == "C:\\\\path"

    def test_newlines_become_escape_sequences(self):
        assert escape_js_string("a\nb") == "a\\nb"

    def test_closing_script_tag_cannot_survive(self):
        assert "</script>" not in escape_js_string("</script>")

    def test_line_separators_are_escaped(self):
        assert escape_js_string("a\u2028b") == "a\\u2028b"

    def test_non_string_input_is_stringified(self):
        assert escape_js_string(12) == "12"


def test_truncate_search_query_numeric():
    """Test that _truncate_search_query converts int/float search inputs to strings instead of returning empty."""
    from src.utils.warning_list import _truncate_search_query

    assert _truncate_search_query(12345) == "12345"
    assert _truncate_search_query(98.6) == "98.6"
    assert _truncate_search_query(None) == ""
