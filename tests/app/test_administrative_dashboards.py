"""Administrative components render and rerun against isolated local state."""
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('module,entrypoint', [
    ('workflow_automation', 'render_workflow_automation'),
    ('audit_compliance_engine', 'render_audit_compliance_engine'),
    ('api_gateway', 'render_api_gateway'),
    ('collaboration_hub', 'render_collaboration_hub'),
    ('advanced_analytics_engine', 'render_analytics_engine'),
])
def test_administrative_dashboard_initialization_and_rerun(tmp_path, module, entrypoint):
    source = f'''
import streamlit as st
st.session_state['data_dir'] = {str(tmp_path)!r}
st.session_state['username'] = 'reviewer'
st.session_state['role'] = 'admin'
from app.components.{module} import {entrypoint}
{entrypoint}()
'''
    at = AppTest.from_string(source, default_timeout=30).run()
    assert not at.exception, [e.message[:300] for e in at.exception]
    assert at.subheader or at.markdown
    assert len(at.tabs) >= 3
    tab_labels = [tab.label for tab in at.tabs]
    state_key = {'workflow_automation': 'workflow_engine', 'audit_compliance_engine': 'audit_compliance_engine', 'api_gateway': 'api_gateway', 'collaboration_hub': 'collaboration_hub', 'advanced_analytics_engine': 'analytics_engine'}[module]
    initialized = at.session_state[state_key]
    at.run()
    assert at.session_state[state_key] is initialized
    assert not at.exception, [e.message[:300] for e in at.exception]
    assert [tab.label for tab in at.tabs] == tab_labels
