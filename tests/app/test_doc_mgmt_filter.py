from streamlit.testing.v1 import AppTest
from src.db.corpus_db import add_document


def test_document_management_filter_ui(mock_db):
    add_document("Alpha.txt", "a")
    add_document("Beta.txt", "b")
    at = AppTest.from_string("from app.views.corpus_view import render_document_management_sidebar; render_document_management_sidebar('admin', '/missing/index', 'filter-test', 0)").run()
    assert not at.exception
    at.text_input(key="doc_mgmt_filter").set_value("ALPHA").run()
    assert not at.exception
    assert list(at.dataframe[0].value["Filename"]) == ["Alpha.txt"]
    at.text_input(key="doc_mgmt_filter").set_value("unknown").run()
    assert not at.exception
    assert not at.dataframe
