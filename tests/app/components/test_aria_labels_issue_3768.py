"""Tests for accessible ARIA attrs on custom HTML controls (Issue #3768)."""

import os

from src.core.models.categorization import DocumentTag
from app.components.tag_renderer import render_tag

_COMPONENTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "app", "components")
)


def test_render_tag_includes_role_and_aria_label():
    html_out = render_tag(DocumentTag(name="Physics", confidence=0.95))
    assert 'role="button"' in html_out
    assert 'aria-label="Physics"' in html_out


def test_notification_badge_markup_includes_aria():
    path = os.path.join(_COMPONENTS, "smart_notifications.py")
    with open(path, encoding="utf-8") as f:
        source = f.read()
    assert 'role="button"' in source
    assert "aria-label=" in source
