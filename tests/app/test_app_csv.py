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


@patch("src.core.ai_detector.detect_documents_ai_probability", return_value={})
@patch("src.core.webhook.send_plagiarism_alert")
@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_app_csv_upload_integration(mock_embed, mock_model_info, mock_webhook, mock_ai):
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

        # Construct CSV data in memory
        csv_content = (
            "student_name,essay_response\n"
            "Alice,This is the first essay answer that has some content for analysis.\n"
            "Bob,This is the second essay response that contains similar matching text for analysis.\n"
        )
        csv_bytes = csv_content.encode("utf-8")

        # Upload the CSV via the file uploader widget
        at.file_uploader[0].upload("assignments.csv", csv_bytes, "text/csv")

        # Execute run to initialize CSV dropdown selectors
        at.run(timeout=30)

        # Ensure no exceptions occurred during UI initialization
        assert not at.exception

        # Verify selectors are rendered
        assert len(at.selectbox) > 0

        # Execute full pipeline
        at.run(timeout=30)

        # Ensure no exceptions occurred during pipeline execution
        assert not at.exception

        # Check if analysis results are rendered correctly in the UI tabs
        assert any("Index total:" in info.body for info in at.info)

    finally:
        _cleanup_stale_artifacts()
