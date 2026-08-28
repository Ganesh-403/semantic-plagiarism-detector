# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
@patch("src.core.webhook.send_plagiarism_alert")
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
        sample_text_b = (
            "Machine learning is a subset of artificial intelligence that provides systems the ability "
            "to automatically learn and improve from experience without being explicitly programmed. "
            "It focuses on developing computer programs that can access data and use it to learn for "
            "themselves, enabling computers to find hidden insights without being explicitly programmed."
        )
        pdf1 = generate_pdf(sample_text_a)
        pdf2 = generate_pdf(sample_text_b)

        # Upload files
        uploaders[0].upload("doc1.pdf", pdf1, "application/pdf")
        uploaders[0].upload("doc2.pdf", pdf2, "application/pdf")

        # Execute full pipeline
        at.run()

        assert not at.exception
        assert len(at.metric) >= 5

        high_severity_keywords = (
            "High",
            "🔴",
            "high",
            "CRITICAL",
            "Critical",
            "danger",
            "Danger",
        )
        badge_found = any(
            any(kw in md.value for kw in high_severity_keywords) for md in at.markdown
        )
        assert badge_found, "High plagiarism warning badge was not rendered"

        mock_webhook.assert_called_once()

    finally:
        _cleanup_stale_artifacts()
