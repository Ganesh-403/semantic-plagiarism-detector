"""Search regressions use the production index and real Streamlit rendering."""

from datetime import datetime

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.components.Doc_Search_Filtering import (
    SearchEngine,
    SearchFilter,
    SearchQuery,
    SearchQueryParser,
    SearchResult,
    SmartFilterBuilder,
)


@pytest.fixture
def engine():
    index = SearchEngine()
    index.index_document(
        "excluded", "climate ocean", {"author": "Bob"}, np.array([1.0, 0.0]), ["draft"]
    )
    index.index_document(
        "included",
        "ocean weather",
        {"author": "Alice", "tags": ["research"]},
        np.array([0.8, 0.6]),
        ["research"],
    )
    index.index_document(
        "semantic", "atmosphere", {"author": "Alice"}, np.array([1.0, 0.0])
    )
    return index


def test_filter_before_limit_and_stable_ranking(engine):
    results = engine.search_full_text(
        "climate ocean", [SearchFilter("author", "eq", "Alice")], max_results=1
    )
    assert [(r.document_name, r.score) for r in results] == [("included", 0.5)]
    assert results[0].metadata["author"] == "Alice"
    assert results[0].snippet == "ocean weather"
    assert engine.search_full_text("the and") == []
    assert engine.search_full_text("absent") == []
    assert engine.search_full_text("ocean", max_results=0) == []
    assert [r.document_name for r in engine.search_full_text("ocean")] == [
        "excluded",
        "included",
    ]


def test_reindex_removes_stale_words_tags_and_vectors(engine):
    engine.index_document("excluded", "replacement", {"author": "Carol"})
    assert engine.search_full_text("climate") == []
    assert engine.search_full_text("replacement")[0].metadata == {"author": "Carol"}
    assert "excluded" not in engine.semantic_embeddings
    assert "draft" not in engine.tag_index
    assert engine.get_search_stats()["indexed_documents"] == 3


def test_semantic_and_hybrid_include_nonlexical_matches(engine):
    vector = np.array([1.0, 0.0])
    results = engine.search_semantic(
        vector, [SearchFilter("author", "eq", "Alice")], threshold=0.7
    )
    assert [(r.document_name, round(r.score, 2)) for r in results] == [
        ("semantic", 1.0),
        ("included", 0.8),
    ]
    hybrid = engine.search_hybrid(
        "ocean", vector, [SearchFilter("author", "eq", "Alice")], semantic_weight=0.75
    )
    scores = {r.document_name: r.score for r in hybrid}
    assert scores == pytest.approx({"semantic": 1.0, "included": 0.85})
    assert all(r.match_type == "hybrid" for r in hybrid)
    assert engine.search_hybrid("ocean", max_results=1)[0].document_name == "excluded"
    assert engine.search_hybrid("ocean", max_results=0) == []
    with pytest.raises(ValueError, match="semantic_weight"):
        engine.search_hybrid("ocean", semantic_weight=1.1)


@pytest.mark.parametrize(
    "bad",
    [np.zeros(2), np.array([np.nan, 0.0]), np.array([np.inf, 1.0]), np.ones((1, 2))],
)
def test_semantic_invalid_vectors_never_produce_scores(engine, bad):
    assert engine.search_semantic(bad) == []
    engine.index_document("invalid", "invalid", embeddings=bad)
    assert "invalid" not in [
        r.document_name for r in engine.search_semantic(np.array([1.0, 0.0]))
    ]


def test_semantic_missing_mismatched_and_bounded_results(engine):
    assert SearchEngine().search_semantic(np.ones(2)) == []
    assert engine.search_semantic(None) == []
    assert engine.search_semantic(np.ones(3)) == []
    assert engine.search_semantic(np.ones(2), max_results=-1) == []
    engine.index_document("long", "word " * 100, embeddings=np.array([1.0, 0.0]))
    result = next(
        r
        for r in engine.search_semantic(np.array([1.0, 0.0]))
        if r.document_name == "long"
    )
    assert result.snippet.endswith("...") and len(result.snippet) == 303


@pytest.mark.parametrize(
    "operator,value,expected",
    [
        ("eq", 3, True),
        ("ne", 3, False),
        ("gt", 2, True),
        ("gt", 3, False),
        ("lt", 4, True),
        ("gte", 3, True),
        ("lte", 3, True),
        ("in", [2, 3], True),
        ("contains", "3", True),
        ("unknown", 3, False),
        ("gt", "wrong", False),
    ],
)
def test_metadata_filters(operator, value, expected):
    query_filter = SearchFilter("size", operator, value)
    assert query_filter.matches({"size": 3}) is expected
    assert not query_filter.matches({})
    assert query_filter.to_dict() == {
        "field": "size",
        "operator": operator,
        "value": value,
    }


def test_tag_membership_and_case_insensitive_author():
    assert SearchFilter("tags", "in", ["research"]).matches(
        {"tags": ["draft", "research"]}
    )
    assert not SearchFilter("tags", "in", ["final"]).matches({"tags": ["draft"]})
    assert SearchFilter("author", "contains", "ALICE").matches(
        {"author": "Alice Smith"}
    )


def test_query_parser_preserves_single_word_quotes_and_field_values():
    parsed = SearchQueryParser().parse(
        'author:"Alice Smith" "ocean" climate AND "sea level" -draft OR weather'
    )
    assert parsed == {
        "fields": {"author": "Alice Smith"},
        "exact_phrases": ["ocean", "sea level"],
        "keywords": ["climate", "weather"],
        "excluded": ["draft"],
        "operators": ["AND", "OR"],
    }
    assert SearchQueryParser().parse('"unfinished phrase')["exact_phrases"] == [
        "unfinished phrase"
    ]


def test_filter_builder_rejects_invalid_controls_and_builds_supported_fields():
    builder = SmartFilterBuilder()
    assert builder.create_filter("size", "gt", 10).matches({"size": 11})
    with pytest.raises(ValueError, match="Unknown field"):
        builder.create_filter("invalid", "eq", 1)
    with pytest.raises(ValueError, match="Invalid operator"):
        builder.create_filter("author", "gt", "A")
    assert builder.build_from_dict(
        {"size": {"gte": 3, "bad": 0}, "unknown": {"eq": 1}}
    ) == [SearchFilter("size", "gte", 3)]
    assert "author" in builder.get_filter_options()


def test_saved_search_lifecycle_and_snippets(engine):
    query = SearchQuery(
        "one", "ocean", {}, "full_text", datetime(2026, 1, 1), "alice", 2
    )
    engine.search_history.append(query)
    saved_id = engine.save_search(query, "Ocean research")
    assert engine.get_saved_searches()[saved_id].to_dict()["name"] == "Ocean research"
    assert query.to_dict()["timestamp"] == "2026-01-01T00:00:00"
    assert engine.get_search_history() == [query]
    assert engine.get_search_stats() == {
        "total_searches": 1,
        "saved_searches": 1,
        "indexed_documents": 3,
        "unique_tags": 2,
    }
    assert engine.delete_saved_search(saved_id)
    assert not engine.delete_saved_search(saved_id)
    content = "x" * 400 + " ocean " + "z" * 400
    snippet = engine._generate_snippet(content, ["ocean"])
    assert snippet.startswith("...") and snippet.endswith("...") and "ocean" in snippet
    assert engine._generate_snippet(content, []) == content[:300] + "..."
    assert engine._generate_snippet("short", ["absent"]) == "short"
    assert engine._generate_snippet("", []) == ""
    assert SearchResult("d", 1.0, "s", "full_text").to_dict()["highlights"] == []


def _search_app():
    import streamlit as st
    from app.components.Doc_Search_Filtering import (
        SearchEngine,
        render_advanced_search_dashboard,
    )

    if "search_engine" not in st.session_state:
        index = SearchEngine()
        index.index_document(
            "article",
            'ocean <script>alert("x")</script>',
            {"author": "Alice", "tags": ["research"], "similarity": 0.8},
        )
        st.session_state["search_engine"] = index
    render_advanced_search_dashboard()


def test_search_dashboard_real_filters_history_reset_and_html_escaping():
    at = AppTest.from_function(_search_app, default_timeout=30).run()
    assert not at.exception
    at.text_input(key="search_query_input").set_value("ocean")
    at.text_input(key="author_filter_input").set_value("Alice").run()
    at.multiselect(key="tags_filter_select").set_value(["research"])
    at.checkbox(key="sim_filter_check").check().run()
    at.button(key="search_button").click().run()
    assert not at.exception, [e.message for e in at.exception]
    index = at.session_state["search_engine"]
    assert len(index.search_history) == 1
    assert index.search_history[0].results_count == 1
    assert len(index.search_history[0].filters) == 3
    assert any("&lt;script&gt;" in m.value for m in at.markdown)
    assert not any("<script>alert" in m.value for m in at.markdown)
    at.run()
    assert len(index.search_history) == 1
    at.button(key="view_article_0").click().run()
    assert at.session_state["selected_doc"] == "article"
    at.button(key="annotate_article_0").click().run()
    assert at.session_state["annotate_doc"] == "article"
    at.button(key="compare_article_0").click().run()
    assert at.session_state["compare_doc"] == "article"
    at.button(key="reset_filters_button").click().run()
    assert not at.exception
    assert at.text_input(key="author_filter_input").value == ""
    assert at.multiselect(key="tags_filter_select").value == []
    assert not at.checkbox(key="sim_filter_check").value


def _results_app():
    import streamlit as st
    from app.components.Doc_Search_Filtering import SearchResult, render_search_results

    count = st.number_input("Count", 1, 25, 12)
    render_search_results(
        [SearchResult(str(i), 0.5, "text", "full_text") for i in range(count)]
    )


def test_results_page_clamps_when_new_query_has_fewer_matches():
    at = AppTest.from_function(_results_app, default_timeout=30).run()
    next(b for b in at.button if b.label == "Next →").click().run()
    assert at.session_state["search_page"] == 1
    at.number_input[0].set_value(1).run()
    assert not at.exception
    assert at.session_state["search_page"] == 0
    assert at.expander[0].label.startswith("#1 - 0")


@pytest.mark.parametrize(
    "preset,days",
    [("Today", 0), ("Last 7 Days", 7), ("Last 30 Days", 30), ("Last 90 Days", 90)],
)
def test_date_filter_records_the_selected_boundary(preset, days):
    from datetime import timedelta

    at = AppTest.from_function(_search_app, default_timeout=30).run()
    at.session_state["search_engine"].metadata_index["article"]["date"] = "2099-01-01"
    at.selectbox(key="date_filter_select").set_value(preset).run()
    at.text_input(key="search_query_input").set_value("ocean")
    at.button(key="search_button").click().run()
    assert not at.exception, [e.message for e in at.exception]
    query_filter = at.session_state["search_engine"].search_history[-1].filters["0"]
    assert query_filter == {
        "field": "date",
        "operator": "gt",
        "value": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
    }


def _saved_app():
    from datetime import datetime
    import streamlit as st
    from app.components.Doc_Search_Filtering import (
        SearchEngine,
        SearchQuery,
        render_saved_searches,
    )

    if "engine" not in st.session_state:
        engine = SearchEngine()
        engine.save_search(
            SearchQuery("one", "ocean", {}, "full_text", datetime(2026, 1, 1), "alice"),
            "Ocean",
        )
        st.session_state["engine"] = engine
    render_saved_searches(st.session_state["engine"])


def test_saved_search_can_be_run_and_deleted_in_real_view():
    at = AppTest.from_function(_saved_app, default_timeout=30).run()
    saved_id = next(iter(at.session_state["engine"].get_saved_searches()))
    at.button(key=f"run_saved_{saved_id}").click().run()
    assert not at.exception and at.session_state["active_search"]
    assert at.session_state["search_query"] == "ocean"
    at.button(key=f"delete_saved_{saved_id}").click().run()
    assert not at.exception and not at.session_state["engine"].get_saved_searches()
    assert any("No saved searches" in i.value for i in at.info)


def _empty_search_app():
    from app.components.Doc_Search_Filtering import render_advanced_search_dashboard

    render_advanced_search_dashboard()


def test_empty_dashboard_initialization_and_missing_encoder_feedback():
    at = AppTest.from_function(_empty_search_app, default_timeout=30).run()
    assert not at.exception
    assert (
        at.session_state["search_engine"].get_search_stats()["indexed_documents"] == 0
    )
    at.session_state["active_search"] = True
    at.session_state["search_type"] = "semantic"
    at.session_state["search_query"] = "ocean"
    at.run()
    assert not at.exception
    assert any("configured query encoder" in e.value for e in at.error)
    assert at.session_state["search_results"] == []
