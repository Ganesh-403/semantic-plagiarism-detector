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
test_playwright_e2e_issue_2945.py
----------------------------------
End-to-End Playwright integration test suite for Issue #2945:
Launches the Streamlit application against an isolated DB instance,
logs in as a valid user, uploads dummy assignment files, runs quick verification,
and asserts that the similarity/plagiarism score is computed and displayed on screen.
"""

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import TEST_PASSWORD, TEST_USERNAME
from tests.e2e.fixtures.sample_docs import STUDENT_A_TEXT, write_sample_docs
from tests.e2e.pages.login_page import LoginPage
from tests.e2e.pages.upload_page import UploadPage

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.e2e,
]


def test_playwright_e2e_login_upload_and_assert_similarity_score(
    page: Page,
    streamlit_url: str,
    tmp_path: Path,
) -> None:
    """E2E Playwright test for Issue #2945: Launch app, login, upload dummy file, and assert similarity score."""
    # 1. Navigate to the Streamlit application URL
    page.goto(streamlit_url)

    # 2. Log in using seeded test user credentials
    login = LoginPage(page)
    login.login(TEST_USERNAME, TEST_PASSWORD)

    # 3. Wait for post-login upload interface
    upload = UploadPage(page)
    upload.wait_for_upload_section()

    # 4. Upload dummy sample assignment documents
    sample_docs = write_sample_docs(tmp_path)
    upload.upload_files([sample_docs["student_a.txt"], sample_docs["student_b.txt"]])

    # Assert files are staged
    staged_text = upload.staged_files_info.inner_text()
    assert re.search(r"Staged\s+(\d+)\s+files?", staged_text), staged_text

    # 5. Run Quick Verification against an indexed text snippet
    snippet = STUDENT_A_TEXT.split(".")[0]
    upload.run_quick_verification(snippet)

    # 6. Assert similarity score is rendered and displayed on-screen
    upload.assert_result_present()
    top_score = upload.get_top_similarity_percent()
    assert top_score is not None, "Similarity score was not displayed on screen"
    assert 0.0 <= top_score <= 100.0, f"Unexpected similarity score value: {top_score}"
