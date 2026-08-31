"""Tests for Accessible High Contrast theme (Issue #3761)."""

import os
from unittest.mock import patch

import streamlit as st

from app.theme import HIGH_CONTRAST_THEME, THEMES, set_theme


def test_high_contrast_theme_registered():
    assert HIGH_CONTRAST_THEME["background"] == "#000000"
    assert HIGH_CONTRAST_THEME["ink"] == "#ffffff"
    assert HIGH_CONTRAST_THEME["accent"] == "#ffff00"
    assert THEMES["Accessible High Contrast"] == HIGH_CONTRAST_THEME


def test_sidebar_exposes_theme_selector():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "app", "streamlit_app.py")
    )
    with open(path, encoding="utf-8") as f:
        source = f.read()
    assert 'key="theme_selector"' in source
    assert "THEMES.keys()" in source


def test_set_theme_accepts_high_contrast():
    st.session_state.clear()
    with patch("app.theme.st.query_params", {}):
        set_theme("Accessible High Contrast")
    assert st.session_state.theme == "Accessible High Contrast"
    assert st.session_state.theme_colors["background"] == "#000000"
