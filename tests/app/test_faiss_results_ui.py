from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_faiss_results_use_streamlit_dataframe():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "results_df = faiss_results_dataframe(q_results)" in source
    assert "st.dataframe(" in source
    assert 'key="faiss_search_results_table"' in source


def test_static_faiss_result_loop_is_removed():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "for rec, score in q_results:" not in source


def test_sortable_columns_are_configured():
    source = APP_PATH.read_text(encoding="utf-8")
    assert '"Similarity Score": (' in source
    assert '"Target Document": (' in source
    assert 'format="%.1f%%"' in source
