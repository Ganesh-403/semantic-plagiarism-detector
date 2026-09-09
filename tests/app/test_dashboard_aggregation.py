"""Dashboard aggregation and rendering use real incident values and export bytes."""
import csv
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.components import analysis_exports, dashboard_stats as dashboard, similarity_score_analytics as analytics


@pytest.mark.parametrize('module', [dashboard, analytics])
@pytest.mark.parametrize('value', [None, 'invalid', object(), float('nan'), float('inf'), float('-inf')])
def test_numeric_helpers_return_finite_defaults(module, value):
    assert module._safe_float(value, 4.0) == 4.0
    assert module._safe_int(value, 4) == 4


def test_dashboard_normalizes_model_and_dict_incidents():
    frame = dashboard._incidents_to_dataframe([
        {'incident_id': 1, 'doc_a': 'a', 'doc_b': 'b', 'similarity_score': 0.8, 'severity': '🔴 High', 'review_status': 'Resolved'},
        SimpleNamespace(incident_id=2, document_a='c', document_b='d', similarity=60, severity_rank='Medium', review_status='Pending'),
        {'incident_id': 3, 'similarity_score': 0, 'similarity': 99, 'review_status': 'unresolved'},
    ])
    assert frame['document_a'].tolist() == ['a', 'c', 'Unknown']
    assert frame['similarity_score'].tolist() == [0.8, 60, 0]
    assert frame['review_status'].tolist() == ['Resolved', 'Pending', 'Pending']
    stats = dashboard._calculate_dashboard_stats(frame, 4)
    assert stats['total_incidents'] == 3 and stats['total_documents'] == 4
    assert stats['avg_similarity'] == pytest.approx(140 / 3)
    assert stats['max_similarity'] == 80
    assert stats['high_severity_count'] == stats['medium_severity_count'] == 1
    assert stats['pending_reviews_count'] == 2
    assert stats['incident_rate'] == 75
    empty = dashboard._calculate_dashboard_stats(dashboard._incidents_to_dataframe([]), 0)
    assert empty['incident_rate'] == empty['avg_similarity'] == empty['pending_review_pct'] == 0


def test_statistics_percentiles_and_outliers_ignore_non_finite_values():
    scores = pd.Series([10, 20, 30, 40, np.nan, np.inf, -np.inf])
    stats = analytics._compute_statistics(scores)
    assert stats['total_samples'] == 4 and stats['mean'] == stats['median'] == 25
    assert stats['variance'] == pytest.approx(500 / 3)
    assert stats['range_val'] == 30
    assert analytics._compute_percentiles(scores)['P25'] == 17.5
    assert analytics._compute_percentiles(scores)['P75'] == 32.5
    outliers = analytics._detect_outliers(pd.Series([10, 10, 10, 10, 100, np.nan]))
    assert outliers['total_outliers'] == 1 and outliers['highest_outlier'] == 100
    assert outliers['outlier_percentage'] == 20
    assert analytics._detect_outliers(pd.Series([1, 2, 3, 4]))['total_outliers'] == 0
    for values in ([], [np.nan, np.inf]):
        series = pd.Series(values, dtype=float)
        assert analytics._compute_statistics(series)['total_samples'] == 0
        assert analytics._compute_percentiles(series)['P99'] == 0
        assert analytics._detect_outliers(series)['total_outliers'] == 0
    assert analytics._compute_statistics(pd.Series([7]))['std_dev'] == 0


def test_analysis_downloads_include_real_matches_and_escape_csv_formulas(monkeypatch):
    button = Mock()
    monkeypatch.setattr(analysis_exports.st, 'download_button', button)
    flags = [{'doc_a': '=HYPERLINK("bad")', 'doc_b': 'Résumé.txt', 'similarity': 0.91}]
    analysis_exports.render_analysis_exports(flags, ['=HYPERLINK("bad")', 'Résumé.txt'], 0.7)
    calls = button.call_args_list
    payload = json.loads(calls[0].args[1])
    assert payload['matches'] == flags and payload['threshold'] == 0.7
    rows = list(csv.reader(io.StringIO(calls[1].args[1])))
    assert rows[1] == ['\'=HYPERLINK("bad")', 'Résumé.txt', '0.91']
    button.reset_mock()
    analysis_exports.render_analysis_exports([], [], 0.7)
    button.assert_not_called()


@pytest.mark.parametrize('populated', [False, True])
@pytest.mark.parametrize('module', ['dashboard_stats', 'similarity_score_analytics'])
def test_analytics_views_render_data_and_empty_states(monkeypatch, module, populated):
    selected = dashboard if module == 'dashboard_stats' else analytics
    rows = [{'incident_id': i, 'document_a': f'a{i}.txt', 'document_b': 'source.txt', 'similarity_score': i / 10, 'severity_rank': 'High' if i > 6 else 'Medium', 'review_status': 'Pending', 'date_flagged': '2026-09-09', 'last_seen': '2026-09-09'} for i in range(1, 10)] if populated else []
    if module == 'dashboard_stats':
        monkeypatch.setattr(selected, '_load_incidents_cached', lambda: rows)
        monkeypatch.setattr(selected, '_load_documents_cached', lambda: [{'filename': 'source.txt'}] if populated else [])
        monkeypatch.setattr(selected, '_load_document_count_cached', lambda: 10 if populated else 0)
        monkeypatch.setattr(selected, '_load_high_severity_trends_cached', lambda **kw: [])
        monkeypatch.setattr(selected, '_load_most_plagiarized_documents_cached', lambda **kw: [])
        monkeypatch.setattr(selected, '_load_storage_footprint_cached', lambda: {'database_bytes': 100, 'embedding_bytes': 40, 'embedding_percentage': 40, 'chunk_count': 1})
    else:
        monkeypatch.setattr(selected, '_load_incidents_data', lambda: rows)
    function = 'render_dashboard_stats' if module == 'dashboard_stats' else 'render_similarity_score_analytics'
    at = AppTest.from_string(f'from app.components.{module} import {function}\n{function}()', default_timeout=20).run()
    assert not at.exception, [(e.message, e.stack_trace) for e in at.exception]
    assert at.markdown
    if populated:
        assert len(at.get('plotly_chart')) >= 2
        if module == 'similarity_score_analytics':
            at.checkbox(key='hist_density').check().run()
            at.checkbox(key='hist_mean').uncheck().run()
            assert not at.exception
