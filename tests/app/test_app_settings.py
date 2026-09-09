from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_settings_reset_to_defaults(mock_db):
    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app/pages/2_Settings.py"))
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["role"] = "admin"
    at.run()
    assert not at.exception
    at.slider(key="threshold_slider").set_value(.7)
    at.checkbox(key="chunk_matrix_checkbox").check()
    at.slider(key="faiss_top_k_slider").set_value(10).run()
    assert not at.exception
    at.button(key="reset_defaults_button").click().run()
    assert not at.exception
    assert at.slider(key="threshold_slider").value == .59
    assert not at.checkbox(key="chunk_matrix_checkbox").value
    assert at.slider(key="faiss_top_k_slider").value == 5
