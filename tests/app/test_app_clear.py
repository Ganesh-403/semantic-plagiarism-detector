import os

from streamlit.testing.v1 import AppTest

# Paths to index and DB
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_INDEX_PATH = os.path.join(_REPO_ROOT, "corpus.index")
_DB_PATH = os.path.join(_REPO_ROOT, "corpus.db")


def test_clear_all_button_visibility():
    """Verify that Clear All Documents button is visible ONLY for administrators."""
    # 1. Admin user - should see the button
    at_admin = AppTest.from_file("app/streamlit_app.py")
    at_admin.session_state["authenticated"] = True
    at_admin.session_state["username"] = "admin"
    at_admin.session_state["role"] = "admin"
    at_admin.run()

    admin_btn = any("Clear All Documents" in btn.label for btn in at_admin.button)
    assert admin_btn is True

    # 2. Regular user (teacher) - should NOT see the button
    at_teacher = AppTest.from_file("app/streamlit_app.py")
    at_teacher.session_state["authenticated"] = True
    at_teacher.session_state["username"] = "teacher1"
    at_teacher.session_state["role"] = "teacher"
    at_teacher.run()

    teacher_btn = any("Clear All Documents" in btn.label for btn in at_teacher.button)
    assert teacher_btn is False


def test_clear_all_confirmation_modal_interaction():
    """Verify that clicking Clear All Documents opens the dialog with Cancel and Clear All buttons."""
    at = AppTest.from_file("app/streamlit_app.py")
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["role"] = "admin"
    at.run()

    # Find and click "Clear All Documents" button
    clear_btn = None
    for btn in at.button:
        if "Clear All Documents" in btn.label:
            clear_btn = btn
            break

    assert clear_btn is not None
    clear_btn.click().run()

    # Check if the modal buttons (Cancel, Clear All) appear
    cancel_btn = None
    confirm_btn = None
    for btn in at.button:
        if "Cancel" in btn.label:
            cancel_btn = btn
        elif (
            "Clear All" == btn.label
        ):  # Exact label of the confirmation button in dialog
            confirm_btn = btn

    assert cancel_btn is not None
    assert confirm_btn is not None

    # Clicking cancel should not crash and should rerun/close
    cancel_btn.click().run()
    assert not at.exception
