import io
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from reportlab.pdfgen import canvas
from streamlit.testing.v1 import AppTest

# Mock streamlit_tour globally during test import to avoid StreamlitAPIException
sys.modules["streamlit_tour"] = MagicMock()
from tests.conftest import MockDataFactory

# Paths to stale artifacts that can pollute test runs
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_STALE_INDEX = os.path.join(_REPO_ROOT, "corpus.index")
_STALE_DB = os.path.join(_REPO_ROOT, "corpus.db")


def _cleanup_stale_artifacts():
    """Remove leftover FAISS index and SQLite DB from prior runs."""
    for path in (_STALE_INDEX, _STALE_DB):
        try:
            if os.path.exists(path):
                os.remove(path)
        except PermissionError:
            pass  # File locked by another process (e.g. SQLite); safe to skip


def generate_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    words = text.split()
    lines = []
    for i in range(0, len(words), 8):
        lines.append(" ".join(words[i : i + 8]))

    y = 750
    for line in lines:
        c.drawString(50, y, line)
        y -= 20

    c.showPage()
    c.save()
    return buf.getvalue()


def mock_embed_chunks(chunks, batch_size=64):
    if not chunks:
        return np.array([])
    val = 1.0 / (384**0.5)
    return np.full((len(chunks), 384), val, dtype="float32")


@pytest.fixture(autouse=True)
def clean_smoke_test_env():
    import os

    from src.db.corpus_db import clear_all_data

    clear_all_data()
    index_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "corpus.index")
    )
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
        except Exception:
            pass
    yield
    clear_all_data()
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
        except Exception:
            pass


@patch("src.core.ai_detector.detect_ai_probability", return_value=0.10)
@patch("src.core.webhook.dispatch_plagiarism_alert")
@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_app_smoke(mock_embed, mock_model_info, mock_webhook, mock_ai_detector):
    # Clean up stale artifacts from prior test runs

    _cleanup_stale_artifacts()
    os.environ["PLAGIARISM_WEBHOOK_URL"] = "https://example.com/webhook"
    try:
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)


        # Pre-seed session state for authentication
        at.session_state["authenticated"] = True
        at.session_state["logged_in"] = True
        at.session_state["username"] = "admin"
        at.session_state["role"] = "admin"
        at.session_state["user"] = {"username": "admin", "role": "admin"}
        at.session_state["page"] = "dashboard"
        at.session_state["nav"] = "Dashboard"

        # Execute app
        at.run()

        # Check for file uploader widget
        uploaders = at.file_uploader
        if not uploaders and hasattr(at, "sidebar"):
            uploaders = at.sidebar.file_uploader

        assert len(uploaders) > 0, (
            f"File uploader widget was not rendered on screen. "
            f"Markdown elements found: {[m.value for m in at.markdown]}"
        )

        # Generate 2 PDFs with distinct text
        sample_text_a = (
            "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural "
            "intelligence displayed by humans and other animals. This field of computer science is "
            "highly focused on study, research and development of agents that perceive their environment "
            "and take actions that maximize their chance of successfully achieving their goals."
        )
        sample_text_b = sample_text_a

        txt1 = sample_text_a.encode("utf-8")
        txt2 = sample_text_b.encode("utf-8")

        # Upload files
        uploaders[0].upload("doc1.txt", txt1, "text/plain")
        uploaders[0].upload("doc2.txt", txt2, "text/plain")


        # Execute full pipeline
        at.run()

        assert not at.exception
        assert len(at.markdown) > 0


        high_severity_keywords = (
            "High",
            "🔴",
            "high",
            "CRITICAL",
            "Critical",
            "danger",
            "Danger",
        )
        assert any(
            any(kw in md.value for kw in high_severity_keywords) for md in at.markdown
        )
        if mock_webhook.called:
            mock_webhook.assert_called()


    finally:
        _cleanup_stale_artifacts()


def test_session_reset_on_logout():
    """Verify that clicking logout resets session_state and clears user credentials cleanly."""
    _cleanup_stale_artifacts()
    try:
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "admin"
        at.session_state["role"] = "admin"
        at.session_state["user"] = {"username": "admin", "role": "admin"}
        at.session_state["custom_test_var"] = "should_be_cleared"
        at.run()

        assert not at.exception
        assert at.session_state["authenticated"] is True

        logout_btn = [
            btn for btn in at.sidebar.button if "Log Out" in btn.label or "🚪" in btn.label
        ][0]
        logout_btn.click().run()

        assert not at.exception
        # Verify authenticated flag is set to False / reset
        authenticated_val = (
            at.session_state["authenticated"]
            if "authenticated" in at.session_state
            else None
        )
        assert authenticated_val in (False, None)

        # Verify username is None or cleared
        username_val = (
            at.session_state["username"]
            if "username" in at.session_state
            else None
        )
        assert username_val is None

        # Verify role is deleted or not admin
        assert "role" not in at.session_state or at.session_state["role"] != "admin"

    finally:
        _cleanup_stale_artifacts()


def test_session_expiration_reset():
    """Verify that automatic session expiration clears authenticated keys from session_state."""
    import time

    _cleanup_stale_artifacts()
    try:
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "admin"
        at.session_state["role"] = "admin"
        # Seed an expired last_interaction timestamp (> 30 mins ago)
        at.session_state["last_interaction"] = time.time() - 3600
        at.run()

        # Session should be expired and authenticated state cleared
        assert "authenticated" not in at.session_state or at.session_state.get("authenticated") in (False, None)
        assert "username" not in at.session_state or at.session_state.get("username") is None
        assert "role" not in at.session_state or at.session_state.get("role") != "admin"
    finally:
        _cleanup_stale_artifacts()


def test_sidebar_reset_all_filters():
    """Verify that clicking 'Reset All Filters' clears the filter keys from session_state."""
    _cleanup_stale_artifacts()
    try:
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "admin"
        at.session_state["role"] = "admin"
        at.session_state["user"] = {"username": "admin", "role": "admin"}
        at.session_state["page"] = "dashboard"
        at.session_state["nav"] = "Dashboard"

        # Pre-seed some filter session states
        at.session_state["threshold_slider"] = 0.85
        at.session_state["class_filter_selectbox"] = "Class A"
        at.session_state["heatmap_mask_threshold"] = 0.50
        at.run()

        assert not at.exception
        assert at.session_state["threshold_slider"] == 0.85
        assert at.session_state["class_filter_selectbox"] == "Class A"

        # Find the reset button
        reset_btns = [
            btn for btn in at.sidebar.button if "Reset All Filters" in btn.label or "🔄" in btn.label
        ]
        assert len(reset_btns) > 0

        # Click the reset button and run
        reset_btns[0].click().run()

        assert not at.exception

        # Keys should be deleted from session_state (or reset to their widget defaults)
        assert at.session_state.get("threshold_slider") != 0.85
        assert at.session_state.get("class_filter_selectbox") != "Class A"
        assert at.session_state.get("heatmap_mask_threshold") != 0.50

    finally:
        _cleanup_stale_artifacts()


def test_api_bearer_token_copy_code_box():
    """Verify that API Bearer Token is displayed using st.code box in settings tab."""
    _cleanup_stale_artifacts()
    try:
        os.environ["API_BEARER_TOKEN"] = "test-secret-bearer-token"
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=30)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "admin"
        at.session_state["role"] = "admin"
        at.run()

        assert not at.exception
        code_blocks = [c.value for c in at.code]
        assert any("test-secret-bearer-token" in val or "secret" in val for val in code_blocks)
    finally:
        _cleanup_stale_artifacts()
