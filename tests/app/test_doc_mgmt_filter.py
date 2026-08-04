from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_document_management_filter_ui():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'doc_filter = st.text_input("Filter documents by filename", key="doc_mgmt_filter")' in source
    assert 'if not doc_filter or doc_filter.lower() in str(d["filename"]).lower()' in source
    assert 'filtered_docs = [' in source
