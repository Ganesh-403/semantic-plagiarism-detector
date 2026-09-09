"""Standalone demonstration dashboards identify sample data and survive navigation."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARDS = [
    'ai_content_authenticator', 'citation_integrity_dashboard',
    'cross_lingual_comparison_hub', 'document_fingerprint_vault',
    'plagiarism_risk_scoring_engine',
]


@pytest.mark.parametrize('module', DASHBOARDS)
def test_demo_dashboard_navigation_retains_its_sample_dataset(module):
    path = Path(__file__).resolve().parents[2] / 'app' / 'components' / (module + '.py')
    at = AppTest.from_file(str(path), default_timeout=20).run()
    assert not at.exception, [e.message[:300] for e in at.exception]
    assert any('Demo data' in item.value for item in at.info)
    data_key = 'demo_' + module
    dataset = repr(at.session_state[data_key])
    navigation = next(radio for radio in at.radio if radio.label in ('Tabs', 'Dashboard Tabs'))
    assert len(navigation.options) >= 4
    for choice in navigation.options:
        next(radio for radio in at.radio if radio.label in ('Tabs', 'Dashboard Tabs')).set_value(choice).run()
        assert not at.exception, (module, choice, [e.message[:300] for e in at.exception])
        assert repr(at.session_state[data_key]) == dataset
        assert len(at.markdown) > 3
    at.run()
    assert repr(at.session_state[data_key]) == dataset
