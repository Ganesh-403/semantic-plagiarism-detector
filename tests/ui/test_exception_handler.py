import pytest
from streamlit.testing.v1 import AppTest
from src.utils.ui_helpers import ui_exception_handler  # type: ignore


def test_ui_exception_handler_catches_runtime_error():
    """Test that ui_exception_handler intercepts a RuntimeError and displays st.error without crashing."""

    at = AppTest.from_function(target_app_code)
    at.run()

    assert len(at.error) > 0
    assert "Simulated runtime error" in at.error[0].value


def target_app_code():
    import streamlit as st

    @ui_exception_handler
    def faulty_function():
        raise RuntimeError("Simulated runtime error")

    st.write("Before exception")
    faulty_function()
    st.write("After exception")
