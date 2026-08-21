"""
tests/app/test_multi_page_structure.py
--------------------------------------
Unit tests verifying the Streamlit multi-page app structure (Issue #2810).

Ensures that the logical views have been correctly decomposed into the
app/pages/ directory and follow Streamlit's naming conventions.
"""

import pytest
import os
from pathlib import Path


class TestMultiPageAppStructure:
    """Test suite for verifying the decomposed Streamlit pages."""

    @pytest.fixture
    def pages_dir(self):
        """Return the path to the app/pages directory."""
        root_dir = Path(__file__).resolve().parents[2]
        return root_dir / "app" / "pages"

    def test_pages_directory_exists(self, pages_dir):
        """Verify the app/pages directory exists."""
        assert pages_dir.exists(), "app/pages directory must exist for multi-page apps"
        assert pages_dir.is_dir()

    def test_dashboard_page_exists(self, pages_dir):
        """Verify the Dashboard page file exists with correct naming."""
        dashboard_files = list(pages_dir.glob("*Dashboard*.py"))
        assert len(dashboard_files) >= 1, "Dashboard page must exist"

        # Verify it starts with a number for ordering
        assert any(f.name[0].isdigit() for f in dashboard_files)

    def test_settings_page_exists(self, pages_dir):
        """Verify the Settings page file exists."""
        settings_files = list(pages_dir.glob("*Settings*.py"))
        assert len(settings_files) >= 1, "Settings page must exist"

    def test_audit_logs_page_exists(self, pages_dir):
        """Verify the Audit Logs page file exists."""
        audit_files = list(pages_dir.glob("*Audit*.py"))
        assert len(audit_files) >= 1, "Audit Logs page must exist"

    def test_pages_have_streamlit_imports(self, pages_dir):
        """Verify all page files import streamlit."""
        py_files = list(pages_dir.glob("*.py"))
        assert len(py_files) > 0, "At least one page file must exist"

        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")
            assert (
                "import streamlit as st" in content or "import streamlit" in content
            ), f"{py_file.name} must import streamlit"

    def test_pages_have_render_functions(self, pages_dir):
        """Verify pages define a main render function or entry point."""
        py_files = list(pages_dir.glob("*.py"))

        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")
            # Should have either a render function or direct st calls
            assert (
                "def render_" in content
                or "st.title" in content
                or "st.set_page_config" in content
            ), f"{py_file.name} must contain Streamlit UI logic"

    def test_pages_have_docstrings(self, pages_dir):
        """Verify all page files have module-level docstrings."""
        py_files = list(pages_dir.glob("*.py"))

        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")
            assert '"""' in content or "'''" in content, (
                f"{py_file.name} must have a module docstring"
            )
