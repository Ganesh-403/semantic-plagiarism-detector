from streamlit.testing.v1 import AppTest


def test_storage_widget_renders_usage_for_admin(mock_db):
    at = AppTest.from_string("from app.views.corpus_view import render_document_management_sidebar; render_document_management_sidebar('admin', '/missing/index', 'storage-test', 0)").run()
    assert not at.exception
    assert any(metric.label == "Total Storage Used" for metric in at.metric)


def test_storage_widget_is_hidden_from_non_admin():
    at = AppTest.from_string("from app.views.corpus_view import render_document_management_sidebar; render_document_management_sidebar('teacher', '/missing/index', 'storage-test', 0)").run()
    assert not at.exception
    assert not at.metric
    assert not at.text_input
