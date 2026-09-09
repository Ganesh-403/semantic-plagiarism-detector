from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_staged_files_counter_badge():
    """Verify that uploading files displays the staged files counter banner and clearing files hides it."""
    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app/streamlit_app.py"))
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["role"] = "admin"
    at.run()

    # 1. Initially, there should be no staged files and no info banner containing 'Staged'
    assert not any("Staged" in info.body for info in at.info)

    # 2. Simulate staging two files (e.g. total 1.5 MB in size)
    class MockUploadedFile:
        def __init__(self, name, size):
            self.name = name
            self.size = size

        def getvalue(self):
            return b"x" * self.size

    file1 = MockUploadedFile("doc1.pdf", 1024 * 1024)  # 1.0 MB
    file2 = MockUploadedFile("doc2.docx", 512 * 1024)  # 0.5 MB

    # In AppTest, we can set the file_uploader session state directly
    at.session_state["file_uploader"] = [file1, file2]
    # Trigger the callback manually or run the app so the callback is executed
    at.run()

    # Verify that the callback set the session state variables
    assert at.session_state["staged_files_count"] == 2
    assert at.session_state["staged_files_size"] == 1024 * 1024 + 512 * 1024

    # Verify that the info banner displays correct information: "📁 Staged 2 files (Total Size: 1.5 MB)"
    banner_text = "📁 Staged 2 files (Total Size: 1.5 MB)"
    assert any(banner_text in info.body for info in at.info)

    # 3. Simulate clearing/emptying the upload queue
    at.session_state["file_uploader"] = []
    at.run()

    # Verify that session state was reset to 0
    assert at.session_state["staged_files_count"] == 0
    assert at.session_state["staged_files_size"] == 0

    # Verify that the banner is cleared/hidden
    assert not any("Staged" in info.body for info in at.info)
