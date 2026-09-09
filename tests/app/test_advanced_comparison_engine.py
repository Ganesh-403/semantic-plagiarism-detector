"""Exercise real heuristic comparisons, score bounds and the comparison view."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from app.components.advanced_comparison_engine import (
    AdvancedComparisonEngine,
    ComparisonMetric,
)


def test_identical_documents_metrics_highlights_history_and_serialization():
    engine = AdvancedComparisonEngine()
    assert engine.get_comparison_stats() == {"total": 0}
    text = "Climate research shows rising sea levels (Smith, 2024). [1]\n\nOcean temperatures affect weather patterns."
    result = engine.compare_documents(text, text, "first", "copy")
    assert result.overall_score == pytest.approx(1.0)
    assert result.risk_level == "critical"
    assert len(result.metrics) == 6 and len(result.recommendations) == 6
    assert all(m.is_passed() and m.score == pytest.approx(1.0) for m in result.metrics)
    assert (0, 0) in result.highlights["a"]
    assert (2, 2) in result.highlights["b"]
    assert json.loads(json.dumps(result.to_dict()))["doc_b"] == "copy"
    assert "proxy" in result.metrics[1].details["method"]
    assert engine.get_comparison_history() == [result.to_dict()]
    assert engine.get_comparison_history(0) == []
    assert engine.get_comparison_history(-1) == []
    stats = engine.get_comparison_stats()
    assert stats["total"] == 1 and stats["avg_score"] == pytest.approx(1.0)
    assert stats["risk_distribution"] == {
        "critical": 1,
        "high": 0,
        "medium": 0,
        "low": 0,
    }


@pytest.mark.parametrize(
    "left,right",
    [
        ("", ""),
        ("...", "words here"),
        ("tiny", ""),
        ("one two three " * 40, "one two three"),
        ("ocean water research", "algorithm compiler software"),
        (
            "First paragraph.\n\nSecond paragraph.",
            "Only one paragraph with more words.",
        ),
    ],
)
def test_comparison_is_bounded_and_symmetric(left, right):
    engine = AdvancedComparisonEngine()
    forward = engine.compare_documents(left, right)
    backward = engine.compare_documents(right, left)
    assert 0 <= forward.overall_score <= 1
    assert forward.overall_score == pytest.approx(backward.overall_score)
    assert all(0 <= m.score <= 1 for m in forward.metrics)
    if not left.strip() or not right.strip():
        assert forward.overall_score == 0
        assert forward.risk_level == "low"
        assert forward.recommendations == [
            "No significant issues detected. Continue monitoring."
        ]


@pytest.mark.parametrize(
    "score,count,risk",
    [
        (0.81, 0, "critical"),
        (0.61, 0, "high"),
        (0.41, 0, "medium"),
        (0.4, 0, "low"),
        (0.2, 4, "critical"),
        (0.2, 3, "high"),
        (0.2, 2, "medium"),
    ],
)
def test_review_risk_boundaries(score, count, risk):
    metrics = [
        ComparisonMetric(str(i), 0.9 if i < count else 0.1, 1 / 6) for i in range(6)
    ]
    assert AdvancedComparisonEngine()._determine_risk_level(metrics, score) == risk


def test_partial_citation_and_ngram_overlap():
    engine = AdvancedComparisonEngine()
    assert engine._calculate_citation_similarity(
        "(Smith, 2024) [1]", "(Smith, 2024) [2]"
    ) == pytest.approx(1 / 3)
    assert engine._ngram_similarity((), ("a",)) == 0
    assert engine._ngram_similarity(
        ("ocean", "rises"), ("ocean", "cools")
    ) == pytest.approx(1 / 3)
    assert engine._extract_keywords("ocean ocean land") == {
        "ocean": 2 / 3,
        "land": 1 / 3,
    }
    assert not ComparisonMetric("low", 0.49, 1.0).to_dict()["passed"]
    assert ComparisonMetric("threshold", 0.5, 1.0).to_dict()["passed"]


def _comparison_app():
    import streamlit as st
    from app.components.advanced_comparison_engine import integrate_comparison_engine

    if st.checkbox("Load documents"):
        st.session_state["document_names"] = ["original", "copy"]
        st.session_state["raw_texts"] = {
            name: "Climate research affects global weather (Smith, 2024)."
            for name in ["original", "copy"]
        }
    integrate_comparison_engine()


def test_comparison_view_renders_once_and_retains_history_on_rerun():
    at = AppTest.from_function(_comparison_app, default_timeout=30).run()
    assert not at.exception and any("2 documents" in w.value for w in at.warning)
    at.checkbox[0].check().run()
    at.button(key="compare_btn").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.get("plotly_chart")) == 1
    assert len(at.session_state["comparison_engine"].comparison_history) == 1
    table = at.dataframe[0].value
    assert set(table["Status"]) == {"⚠️ Above review threshold"}
    assert any("word-frequency proxy" in c.value for c in at.caption)
    at.run()
    assert not at.exception and len(at.get("plotly_chart")) == 1
    assert len(at.session_state["comparison_engine"].comparison_history) == 1
