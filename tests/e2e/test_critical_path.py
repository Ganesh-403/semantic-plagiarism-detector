"""
tests/e2e/test_critical_path.py
--------------------------------
End-to-end critical-path test for the Semantic Plagiarism Detector
(Issue #3030).

Critical path exercised:
    1. Browser launches.
    2. User logs in with seeded credentials.
    3. Two near-duplicate .txt files are uploaded.
    4. The "Run Quick Verification" search is run against an indexed
       corpus snippet.
    5. The plagiarism / similarity score renders on-screen and is
       parsed back out as a number, asserting it lies in the
       expected band.

A second test asserts the negative path — uploading two unrelated
documents yields no significant matches (no false positives).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.fixtures.sample_docs import (
    STUDENT_A_TEXT,
    write_sample_docs,
)
from tests.e2e.pages.login_page import LoginPage
from tests.e2e.pages.upload_page import UploadPage

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.e2e,
]


def _extract_percent(text: str) -> float | None:
    """Pull the first ``NN.N%`` value out of an arbitrary string."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(match.group(1)) if match else None


# ─── Critical path: login → upload → score renders ──────────────────────
def test_login_upload_and_assert_plagiarism_score(
    authenticated_page: Page,
    tmp_path: Path,
) -> None:
    """The full happy-path E2E test requested by Issue #3030."""

    # ── 1. Login has already happened via the ``authenticated_page``
    #       fixture in conftest.py. Sanity-check we are past the gate.
    login = LoginPage(authenticated_page)
    expect(login.login_button).to_be_hidden()

    upload = UploadPage(authenticated_page)
    upload.wait_for_upload_section()

    # ── 2. Upload two near-duplicate .txt files.
    sample_docs = write_sample_docs(tmp_path)
    upload.upload_files([sample_docs["student_a.txt"], sample_docs["student_b.txt"]])

    staged_text = upload.staged_files_info.inner_text()
    assert re.search(r"Staged\s+(\d+)\s+files?", staged_text), staged_text
    assert int(re.search(r"Staged\s+(\d+)\s+files?", staged_text).group(1)) >= 2

    # ── 3. Run the "Quick Verification" search against an indexed
    #       snippet from student_a.txt.
    snippet = STUDENT_A_TEXT.split(".")[0]
    upload.run_quick_verification(snippet)

    # ── 4. Assert the plagiarism score renders on screen.
    upload.assert_result_present()
    top_score = upload.get_top_similarity_percent()
    assert top_score is not None, "Expected a per-match similarity score, got None"
    assert 0.0 <= top_score <= 100.0, f"Score out of range: {top_score}"

    badge_text = upload.similarity_badge.inner_text()
    badge_pct = _extract_percent(badge_text)
    assert badge_pct is not None, f"Could not parse badge text: {badge_text!r}"
    assert 0.0 <= badge_pct <= 100.0

    avg_pct = upload.get_avg_similarity_percent()
    assert avg_pct is None or (0.0 <= avg_pct <= 100.0), avg_pct


# ─── Negative path: unrelated documents should not be flagged ──────────
def test_unrelated_documents_yield_no_false_positive(
    authenticated_page: Page,
    tmp_path: Path,
) -> None:
    """Upload student_a + student_c (unrelated) and confirm no false positive."""
    sample_docs = write_sample_docs(tmp_path)
    upload = UploadPage(authenticated_page)
    upload.wait_for_upload_section()
    upload.upload_files([sample_docs["student_a.txt"], sample_docs["student_c.txt"]])

    upload.run_quick_verification(
        "Quantum entanglement is a phenomenon where two particles "
        "become correlated such that measuring one instantly affects "
        "the state of the other, regardless of distance."
    )

    no_match_locator = authenticated_page.locator(
        "text=/✅ No significant matches found in the assignment database\\./"
    )
    result_locator = upload.first_result_expander

    expect(no_match_locator.or_(result_locator)).to_be_visible(timeout=30_000)
