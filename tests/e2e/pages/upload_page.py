"""
tests/e2e/pages/upload_page.py
------------------------------
Page Object wrapping the post-login upload + scan flow.

The main Streamlit app renders:
- ``st.file_uploader("📂 Upload Assignments", accept_multiple_files=True,
                     type=["pdf","docx","txt","md","markdown","mdown"],
                     key="file_uploader")``
- (for non-admin roles) ``st.text_area("Paste a text snippet to check against index:")``
- ``st.button("🔍 Run Quick Verification", key="user_query")``

The plagiarism score is rendered as:
- A top-level metric: ``st.metric("Avg Similarity %", f"{avg_sim*100:.1f}%")``
- A per-match result inside an expander labeled:
  ``#{rank} · {anon_doc_name} (chunk #{n}) — {score:.1%}``
  followed by an HTML badge:
  ``<span ...>Similarity: {score*100:.1f}%</span>``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from playwright.sync_api import Page, expect


class UploadPage:
    def __init__(self, page: Page) -> None:
        self.page = page

    @property
    def file_uploader_input(self):
        return self.page.locator(
            "div[data-testid='stFileUploader'] input[type='file']"
        ).first

    @property
    def staged_files_info(self):
        return self.page.locator("text=/📁 Staged \\d+ files?/")

    @property
    def query_text_area(self):
        return self.page.locator(
            "label:has-text('Paste a text snippet to check against index:') "
            "+ div textarea, "
            "label:has-text('Paste a text snippet to check against index:') "
            "~ div textarea"
        ).first

    @property
    def run_quick_verification_button(self):
        return self.page.get_by_role("button", name="🔍 Run Quick Verification")

    @property
    def avg_similarity_metric(self):
        return self.page.locator(
            "div[data-testid='stMetric']:has(label:has-text('Avg Similarity %')) "
            "[data-testid='stMetricValue']"
        ).first

    @property
    def first_result_expander(self):
        return self.page.locator(
            "div[data-testid='stExpander'] details:has-text('Document-')"
        ).first

    @property
    def similarity_badge(self):
        return self.page.locator("span:has-text('Similarity:')").first

    def wait_for_upload_section(self) -> None:
        """Block until the upload widget is on the page (post-login)."""
        self.file_uploader_input.wait_for(state="visible", timeout=20_000)

    def upload_file(self, file_path: Path) -> None:
        """Upload a single file via the hidden file input."""
        self.wait_for_upload_section()
        self.file_uploader_input.set_input_files(str(file_path))
        expect(self.staged_files_info).to_be_visible(timeout=20_000)

    def upload_files(self, file_paths: list[Path]) -> None:
        """Upload multiple files at once."""
        self.wait_for_upload_section()
        self.file_uploader_input.set_input_files([str(p) for p in file_paths])
        expect(self.staged_files_info).to_be_visible(timeout=20_000)

    def run_quick_verification(self, query: str) -> None:
        """Paste a query and click the 'Run Quick Verification' button."""
        self.query_text_area.wait_for(state="visible")
        self.query_text_area.fill(query)
        self.run_quick_verification_button.scroll_into_view_if_needed()
        self.run_quick_verification_button.click()
        self.page.wait_for_load_state("networkidle")

    def get_avg_similarity_percent(self) -> Optional[float]:
        """Return the dashboard 'Avg Similarity %' metric as a float.

        Returns None if the metric is not rendered yet (no incidents
        have been recorded, which is the case for a fresh DB).
        """
        try:
            value_el = self.avg_similarity_metric
            if not value_el.is_visible(timeout=5_000):
                return None
            text = value_el.inner_text().strip()
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            return float(match.group(1)) if match else None
        except Exception:
            return None

    def get_top_similarity_percent(self) -> Optional[float]:
        """Return the similarity % of the top-ranked result expander.

        Returns None when no result expanders are present.
        """
        try:
            expander = self.first_result_expander
            if not expander.is_visible(timeout=5_000):
                return None
            summary = expander.locator("summary").inner_text()
            match = re.search(r"—\s*(\d+(?:\.\d+)?)\s*%", summary)
            return float(match.group(1)) if match else None
        except Exception:
            return None

    def assert_no_significant_matches(self) -> None:
        expect(
            self.page.locator(
                "text=/✅ No significant matches found in the assignment database\\./"
            )
        ).to_be_visible(timeout=20_000)

    def assert_result_present(self) -> None:
        expect(self.first_result_expander).to_be_visible(timeout=30_000)
        expect(self.similarity_badge).to_be_visible(timeout=10_000)
