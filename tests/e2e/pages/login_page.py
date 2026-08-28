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

"""
tests/e2e/pages/login_page.py
-----------------------------
Page Object wrapping the Streamlit login screen rendered by
``app/views/auth_view.py``.

Streamlit DOM notes
~~~~~~~~~~~~~~~~~~~
- ``st.text_input("Username")`` renders an ``<input type="text">`` whose
  preceding ``<label>`` contains the literal text "Username".
- ``st.text_input("Password", type="password")`` renders an
  ``<input type="password">`` labelled "Password".
- ``st.button("Login")`` renders a ``<button>`` whose visible text is
  "Login".

We target these elements by their **label text** so the test is
resilient to Streamlit's internal class-name churn.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect


class LoginPage:
    URL_PATH = "/"

    def __init__(self, page: Page) -> None:
        self.page = page

    @property
    def username_input(self):
        return self.page.locator(
            "div[data-testid='stWidgetLabel']:has-text('Username') "
            "~ div input, "
            "label:has-text('Username') ~ div input, "
            "label:has-text('Username') + div input"
        ).first

    @property
    def password_input(self):
        return self.page.locator(
            "label:has-text('Password') + div input, "
            "label:has-text('Password') ~ div input"
        ).first

    @property
    def login_button(self):
        return self.page.get_by_role("button", name="Login")

    @property
    def error_message(self):
        return self.page.locator("text='Invalid username or password.'")

    def goto(self) -> None:
        self.page.goto(
            self.page.url.split("?")[0] + self.URL_PATH, wait_until="domcontentloaded"
        )

    def fill_username(self, value: str) -> None:
        self.username_input.wait_for(state="visible")
        self.username_input.fill(value)

    def fill_password(self, value: str) -> None:
        self.password_input.wait_for(state="visible")
        self.password_input.fill(value)

    def submit(self) -> None:
        self.login_button.scroll_into_view_if_needed()
        self.login_button.click()
        self.page.wait_for_load_state("networkidle")

    def login(self, username: str, password: str) -> None:
        """Fill + submit + assert we left the login screen."""
        self.fill_username(username)
        self.fill_password(password)
        self.submit()
        # After successful login the app calls st.stop() inside
        # render_login_view, then reruns. The login button must
        # disappear.
        expect(self.login_button).to_be_hidden(timeout=20_000)

    def assert_login_visible(self) -> None:
        expect(self.login_button).to_be_visible()

    def assert_error_visible(self) -> None:
        expect(self.error_message).to_be_visible(timeout=10_000)
