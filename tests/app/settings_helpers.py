from pathlib import Path
from streamlit.testing.v1 import AppTest


def settings_page(role="admin"):
    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app/pages/2_Settings.py"))
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["role"] = role
    return at
