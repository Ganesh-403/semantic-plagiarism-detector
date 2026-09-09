from pathlib import Path
import io
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def upload_dashboard(monkeypatch, mock_db, tmp_path):
    import streamlit as st
    from src.db import auth
    from src.core import embedding_model, app_config
    from app import state_manager
    monkeypatch.setenv("SEMANTIC_PLAGIARISM_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setattr(embedding_model, "_active_model_name", "all-MiniLM-L6-v2")
    monkeypatch.setattr(embedding_model, "embed_chunks", lambda chunks, **kwargs: np.full((len(chunks),384), 1/384**.5, dtype=np.float32))
    monkeypatch.setattr(embedding_model, "get_embedding_model_info", lambda: ("all-MiniLM-L6-v2",384))
    monkeypatch.setattr(app_config, "FAISS_INDEX_PATH", tmp_path / "corpus.index")
    monkeypatch.setattr(state_manager, "init_backup_daemon", lambda: None)
    from src.core import ai_detector
    monkeypatch.setattr(ai_detector, "detect_documents_ai_probability", lambda *args, **kwargs: {})
    queue = {}
    def uploader(*args, **kwargs):
        files = []
        for name, data in queue.items():
            f = io.BytesIO(data)
            f.name, f.size = name, len(data)
            files.append(f)
        return files
    monkeypatch.setattr(st, "file_uploader", uploader)
    auth.add_user("upload_admin", "Upload-Test-Password!984", role="admin")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app/streamlit_app.py"), default_timeout=45).run()
    assert not at.exception, [(e.message,e.stack_trace) for e in at.exception]
    at.text_input[0].set_value("upload_admin")
    at.text_input[1].set_value("Upload-Test-Password!984")
    next(b for b in at.button if b.label == "Login").click().run()
    assert not at.exception, [(e.message,e.stack_trace) for e in at.exception]
    assert at.session_state["authenticated"]
    return at, queue
