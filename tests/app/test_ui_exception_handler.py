"""
tests/app/test_ui_exception_handler.py
--------------------------------------
Unit tests for ui_exception_handler decorator (Issue #2787).
Verifies that Streamlit StopException is re-raised to maintain control flow,
while standard exceptions are caught, logged, and return None.
"""

from unittest.mock import patch

import pytest
import streamlit as st

from app.state_manager import ui_exception_handler


class TestUIExceptionHandler:
    """Tests for ui_exception_handler decorator."""

    def test_ui_exception_handler_reraises_stop_exception(self):
        """Verify StopException is re-raised and not intercepted by the error handler."""

        @ui_exception_handler("TestComponent")
        def component_with_stop():
            raise st.runtime.scriptrunner.StopException()

        with (
            pytest.raises(st.runtime.scriptrunner.StopException),
            patch("streamlit.error") as mock_st_error,
        ):
            component_with_stop()

        mock_st_error.assert_not_called()

    def test_ui_exception_handler_catches_generic_exception(self):
        """Verify generic Exception is caught, logged, st.error is called, and None is returned."""

        @ui_exception_handler("FailingComponent")
        def failing_component():
            raise ValueError("Something went wrong during rendering")

        with (
            patch("streamlit.error") as mock_st_error,
            patch("app.state_manager.logger.error") as mock_log_error,
        ):
            result = failing_component()
            assert result is None
            mock_st_error.assert_called_once_with(
                "⚠️ Failed to load component: FailingComponent"
            )
            mock_log_error.assert_called_once()
            assert "Component '%s' failed to render" in mock_log_error.call_args[0][0]
            assert mock_log_error.call_args[0][1] == "FailingComponent"

    def test_ui_exception_handler_successful_execution(self):
        """Verify normal execution passes through return value."""

        @ui_exception_handler("SuccessComponent")
        def success_component(x, y):
            return x + y

        assert success_component(2, 3) == 5
