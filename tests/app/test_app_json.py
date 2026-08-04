import os
import sys
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from tests.conftest import MockDataFactory

# Mock googleapiclient modules to avoid ModuleNotFoundError in environments without them installed
sys.modules["googleapiclient"] = MagicMock()
sys.modules["googleapiclient.discovery"] = MagicMock()
sys.modules["googleapiclient.http"] = MagicMock()
sys.modules["google.oauth2"] = MagicMock()
sys.modules["google.oauth2.service_account"] = MagicMock()

# Mock ML libraries to prevent pytest segmentation faults on Apple Silicon
sys.modules["transformers"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

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
            pass


@patch(
    "src.core.ai_detector.detect_documents_ai_probability",
    return_value={
        "assignment1.txt": {"overall": 0.1},
        "assignment2.txt": {"overall": 0.1},
    },
)
@patch("src.core.webhook.send_plagiarism_alert")
@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_app_json_export_integration(
    mock_embed, mock_model_info, mock_webhook, mock_ai
):
    _cleanup_stale_artifacts()

    try:
        # Instantiate AppTest
        at = AppTest.from_file("app/streamlit_app.py")

        # Simulate authentication in session state
        at.session_state["authenticated"] = True
        at.session_state["username"] = "admin"
        at.session_state["role"] = "admin"
        at.session_state["page"] = "dashboard"

        # Initial run to display uploader
        at.run(timeout=30)

        # Assert uploader is found
        assert len(at.file_uploader) > 0

        # Upload two files to run the plagiarism pipeline
        at.file_uploader[0].upload(
            "doc1.txt", b"First student assignment text.", "text/plain"
        )
        at.file_uploader[0].upload(
            "doc2.txt", b"Second student assignment text.", "text/plain"
        )

        # Execute full pipeline
        at.run(timeout=30)

        # Ensure no exceptions occurred during pipeline execution
        assert not at.exception

        # Locate download buttons
        csv_btn = None
        json_btn = None
        for btn in at.download_button:
            if "CSV" in btn.label:
                csv_btn = btn
            elif "JSON" in btn.label:
                json_btn = btn

        # Ensure both buttons are rendered
        assert csv_btn is not None, "CSV download button not found"
        assert json_btn is not None, "JSON download button not found"

        # Verify JSON download button configuration
        assert json_btn.key == "json_export_button"

    finally:
        _cleanup_stale_artifacts()
