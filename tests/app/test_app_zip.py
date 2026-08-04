import io
import os
import sys
import zipfile
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
def test_app_zip_upload_integration(mock_embed, mock_model_info, mock_webhook, mock_ai):
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

        # Construct a ZIP archive in memory containing two text files
        zip_stream = io.BytesIO()
        with zipfile.ZipFile(zip_stream, "w") as zf:
            zf.writestr(
                "assignment1.txt",
                b"First student assignment text for similarity checking.",
            )
            zf.writestr(
                "assignment2.txt",
                b"Second student assignment text for similarity checking.",
            )
        zip_bytes = zip_stream.getvalue()

        # Upload the ZIP via the file uploader widget
        at.file_uploader[0].upload("assignments.zip", zip_bytes, "application/zip")

        # Execute full pipeline
        at.run(timeout=30)

        # Ensure no exceptions occurred during pipeline execution
        assert not at.exception

        # Check if analysis results are rendered correctly in the UI tabs
        assert any("Index total:" in info.body for info in at.info)

    finally:
        _cleanup_stale_artifacts()
