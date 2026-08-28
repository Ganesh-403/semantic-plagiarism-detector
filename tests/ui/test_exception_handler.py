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
