"""Real pattern detection, risk models, evolution and dashboard regressions."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.components.pattern_recoginition_system import (
    PatternEvolutionTracker,
    PatternRecognitionEngine,
    PredictionEngine,
)


@pytest.fixture
def documents():
    text = "Ocean research studies weather patterns (Smith, 2024).\n\nThese findings describe climate change (Jones, 2025)."
    return {"original": text, "copy": text}


def test_real_detectors_and_rescans_preserve_pattern_identity(documents):
    engine = PatternRecognitionEngine()
    matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
    patterns = engine.detect_patterns(documents, matrix)
    assert {p.pattern_type for p in patterns} == {
        "copy_paste",
        "structural",
        "citation",
        "hybrid",
    }
    assert all(
        p.documents == ["original", "copy"] and 0 <= p.confidence <= 1 for p in patterns
    )
    first = {p.id: p.first_detected for p in patterns}
    repeated = engine.detect_patterns(documents, matrix)
    assert {p.id for p in repeated} == set(first)
    assert len(engine.patterns) == 4
    assert all(p.frequency == 2 and p.first_detected == first[p.id] for p in repeated)
    assert set(engine.pattern_counts.values()) == {2}
    encoded = json.loads(json.dumps(repeated[0].to_dict()))
    assert isinstance(encoded["first_detected"], str)


def test_detectors_honor_selection_and_feature_stats_exclude_self_similarity(documents):
    engine = PatternRecognitionEngine()
    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    features = engine._extract_features(documents, matrix)
    assert features["avg_similarity"] == {"original": 0.2, "copy": 0.2}
    assert features["max_similarity"] == {"original": 0.2, "copy": 0.2}
    assert features["sentence_counts"] == {"original": 2, "copy": 2}
    patterns = engine.detect_patterns(documents, matrix, enabled_types={"citation"})
    assert len(patterns) == 1 and patterns[0].pattern_type == "citation"
    assert engine.detect_patterns(documents, matrix, enabled_types=set()) == []
    single = engine._extract_features({"empty": ""}, np.ones((1, 1)))
    assert single["avg_similarity"]["empty"] == 0
    assert single["sentence_counts"]["empty"] == 0
    assert engine.detect_patterns({}, np.zeros((0, 0))) == []
    assert engine.detect_patterns({"empty": "", "blank": ""}, np.eye(2)) == []
    assert (
        engine._detect_citation(
            {"a": "(Smith, 2024)", "b": "(Other, 2023)", "c": "No references"}
        )
        == []
    )
    assert (
        engine._detect_structural(
            {"a": "word", "b": "very long paragraph " * 50 + "\n\n" + "next " * 50}
        )
        == []
    )


@pytest.mark.parametrize(
    "matrix", [None, [[1.0]], [[1.0, np.nan], [0.2, 1.0]], [[1.0, 2.0], [2.0, 1.0]]]
)
def test_invalid_matrices_fail_without_mutating_patterns(documents, matrix):
    engine = PatternRecognitionEngine()
    with pytest.raises(ValueError, match="matrix"):
        engine.detect_patterns(documents, matrix)
    assert engine.patterns == {}


def test_unknown_pattern_type_is_rejected(documents):
    with pytest.raises(ValueError, match="Unsupported"):
        PatternRecognitionEngine().detect_patterns(
            documents, np.eye(2), enabled_types={"missing"}
        )


def test_heuristic_prediction_handles_numpy_scores_without_fake_confidence(documents):
    engine = PatternRecognitionEngine()
    engine.detect_patterns(documents, np.full((2, 2), 0.95))
    predictor = PredictionEngine(engine)
    result = predictor.predict_risk("copy", "word " * 100, np.array([0.9, 0.95]))
    assert 0 < result.predicted_risk_score <= 1
    assert result.confidence == 0 and result.metadata["method"] == "heuristic"
    assert "Multiple plagiarism patterns detected" in result.contributing_factors
    assert "Low vocabulary diversity" in result.contributing_factors
    assert engine.predictions[result.id] == result
    assert isinstance(json.loads(json.dumps(result.to_dict()))["prediction_date"], str)
    empty = predictor.predict_risk("empty", "", [])
    assert "Document may be too short for reliable analysis" in empty.recommendations


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [-0.1], [1.1], [[0.5]]])
def test_invalid_risk_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="Similarity scores"):
        PredictionEngine(PatternRecognitionEngine()).predict_risk(
            "doc", "content", scores
        )


def _training_data():
    return [
        {
            "similarity_score": score,
            "avg_similarity": score,
            "max_similarity": score,
            "document_length": 100,
            "word_count": 20,
            "unique_words_ratio": 0.5,
            "complexity_score": 0.2,
            "pattern_count": 0,
            "risk_level": label,
        }
        for score, label in [
            (0.05, 0),
            (0.1, 0),
            (0.15, 0),
            (0.9, 1),
            (0.95, 1),
            (1.0, 1),
        ]
        for _ in range(3)
    ]


def test_real_forest_training_probabilities_and_failed_retrain_are_safe():
    predictor = PredictionEngine(PatternRecognitionEngine())
    assert not predictor.train_model([])
    data = _training_data()
    assert predictor.train_model(data)
    low = predictor.predict_risk("low", "short content", [0.1])
    high = predictor.predict_risk("high", "short content", [0.95])
    assert low.predicted_risk_score < high.predicted_risk_score
    assert 0 <= low.predicted_risk_score <= 1 and 0 <= high.predicted_risk_score <= 1
    assert high.metadata["method"] == "model"
    model, scaler = predictor.prediction_model, predictor.scaler
    assert not predictor.train_model([dict(row, risk_level=3) for row in data])
    assert not predictor.train_model(
        [dict(row, document_length="invalid") for row in data]
    )
    assert predictor.prediction_model is model and predictor.scaler is scaler
    assert predictor.train_model([dict(row, risk_level=0) for row in data])
    assert predictor.predict_risk("safe", "text", [0.9]).predicted_risk_score == 0


@pytest.mark.parametrize(
    "score,level",
    [(0.0, "low"), (0.4, "low"), (0.5, "medium"), (0.7, "high"), (0.9, "critical")],
)
def test_risk_level_boundaries_and_recommendations(score, level):
    predictor = PredictionEngine(PatternRecognitionEngine())
    assert predictor._get_risk_level(score) == level
    result = predictor._generate_recommendations(
        [0, 1000, 100, 0.8, 0.5, 0, 0, 0], level
    )
    if level in {"high", "critical"}:
        assert "Conduct immediate document review" in result
    else:
        assert result == ["Document appears low risk. Continue monitoring."]


@pytest.mark.parametrize(
    "frequencies,direction,change,next_value",
    [
        ([2, 4, 6], "increasing", 2.0, 8.0),
        ([6, 4, 2], "decreasing", -2 / 3, 0.0),
        ([3, 3, 3], "stable", 0.0, 3.0),
        ([0, 0, 1], "increasing", 0.0, 1.5),
    ],
)
def test_evolution_percent_change_and_forecasts(
    documents, frequencies, direction, change, next_value
):
    engine = PatternRecognitionEngine()
    pattern = engine.detect_patterns(documents, np.ones((2, 2)))[0]
    tracker = PatternEvolutionTracker(engine)
    tracker._update_trend("unknown")
    tracker.track_evolution("unknown", {})
    tracker.track_evolution("unknown", {})
    for frequency in frequencies:
        tracker.track_evolution(
            pattern.id, {"frequency": frequency, "new_documents": 1}
        )
    trend = tracker.trends[pattern.id]
    assert trend.risk_trend == direction and trend.frequency_change == pytest.approx(
        change
    )
    assert trend.forecast["next_frequency"] == pytest.approx(next_value)
    bucket = {
        "increasing": "trending_up",
        "decreasing": "trending_down",
        "stable": "stable",
    }[direction]
    assert tracker.get_evolution_insights()[bucket] == [pattern.id]
    tracker.track_evolution("new", {"frequency": 1})
    assert "new" in tracker.get_evolution_insights()["new_patterns"]
    tracker.evolution_data["new"][0]["timestamp"] = datetime.now() - timedelta(days=8)
    assert "new" not in tracker.get_evolution_insights()["new_patterns"]
    assert tracker._generate_forecast([]) == {"predictions": "insufficient_data"}


def _pattern_app():
    import numpy as np
    import streamlit as st
    from app.components.pattern_recoginition_system import integrate_pattern_recognition

    count = st.selectbox("Documents", [0, 1, 2])
    text = "Climate research changes weather (Smith, 2024).\n\nEvidence supports this conclusion (Jones, 2025)."
    st.session_state["raw_texts"] = {
        name: text for name in ["original", "<script>copy</script>"][:count]
    }
    st.session_state["similarity_matrix"] = (
        np.full((count, count), 0.95)
        if st.checkbox("Comparison ready", value=True)
        else None
    )
    integrate_pattern_recognition()


def _button(at, label):
    return next(button for button in at.button if button.label == label)


def test_real_pattern_dashboard_options_predictions_and_evolution():
    at = AppTest.from_function(_pattern_app, default_timeout=30).run()
    assert not at.exception and any("No documents" in w.value for w in at.warning)
    next(s for s in at.selectbox if s.label == "Documents").set_value(1).run()
    assert any("2 documents" in w.value for w in at.warning)
    next(s for s in at.selectbox if s.label == "Documents").set_value(2).run()
    for box in at.checkbox:
        if box.label in {
            "Detect Structural Patterns",
            "Detect Citation Patterns",
            "Detect Hybrid Patterns",
        }:
            box.uncheck()
    _button(at, "🔍 Detect Patterns").click().run()
    assert not at.exception, [e.message for e in at.exception]
    engine = at.session_state["pattern_engine"]
    assert [p.pattern_type for p in engine.patterns.values()] == ["copy_paste"]
    _button(at, "🔮 Generate Predictions").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(engine.predictions) == 2
    assert any("&lt;script&gt;copy&lt;/script&gt;" in m.value for m in at.markdown)
    assert any("Not calibrated" in m.value for m in at.markdown)
    tracker = at.session_state["evolution_tracker"]
    pattern_id = next(iter(engine.patterns))
    for frequency in [1, 2, 3]:
        tracker.track_evolution(pattern_id, {"frequency": frequency})
    at.run()
    assert not at.exception and len(at.get("plotly_chart")) == 3
    assert any(
        "Pattern" in frame.value.columns and "Change" in frame.value.columns
        for frame in at.dataframe
    )
    for box in at.checkbox:
        if box.label.startswith("Detect "):
            box.uncheck()
    _button(at, "🔍 Detect Patterns").click().run()
    assert any("No plagiarism patterns" in item.value for item in at.info)
    next(box for box in at.checkbox if box.label == "Comparison ready").uncheck().run()
    assert not at.exception and any(
        "Run document comparison" in w.value for w in at.warning
    )
