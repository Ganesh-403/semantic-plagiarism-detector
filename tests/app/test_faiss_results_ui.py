from unittest.mock import patch

from streamlit.testing.v1 import AppTest


SCRIPT = """
from types import SimpleNamespace
import streamlit as st
from app.views.faiss_view import render_faiss_view
index = SimpleNamespace(ntotal=1) if st.session_state.get("index_loaded", True) else None
render_faiss_view(index, [], 5, .7, {})
"""


def test_faiss_search_results_survive_button_reruns():
    results = [({"doc_name": "essay.pdf", "chunk_text": "A matching passage."}, .92)]
    with (
        patch("app.views.faiss_view.embed_chunks", return_value=[[1., 0.]]) as embed,
        patch("app.views.faiss_view.search_similar_chunks", return_value=results) as search,
        patch("app.views.faiss_view.render_faiss_results_ui") as render,
    ):
        at = AppTest.from_string(SCRIPT).run()
        assert not at.exception
        render.assert_not_called()
        at.text_input(key="faiss_query_input_tab2").set_value("A query")
        at.button(key="run_search_tab2").click().run()
        assert not at.exception
        render.assert_called_once_with(results, "A query", document_pdf_bytes={})
        at.run()
        assert not at.exception
        assert render.call_count == 2
        embed.assert_called_once_with(["A query"])
        search.assert_called_once()


def test_unloaded_index_clears_results():
    at = AppTest.from_string(SCRIPT)
    at.session_state["index_loaded"] = False
    at.session_state["faiss_search_result"] = ("old", [])
    at.run()
    assert not at.exception
    assert "faiss_search_result" not in at.session_state
    assert not at.button
    assert "No FAISS index" in at.info[0].value
