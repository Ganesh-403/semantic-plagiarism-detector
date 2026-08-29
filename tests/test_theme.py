import pytest
from app.theme import HIGH_CONTRAST_THEME, get_theme_config

def test_high_contrast_theme_definition():
    """Verify that HIGH_CONTRAST_THEME meets WCAG AAA specifications."""
    assert HIGH_CONTRAST_THEME["background"] == "#000000"
    assert HIGH_CONTRAST_THEME["text"] == "#ffffff"
    assert HIGH_CONTRAST_THEME["highlight"] == "#ffff00"

def test_get_theme_config_selection():
    """Verify theme config retrieval handles high contrast selection correctly."""
    config = get_theme_config("Accessible High Contrast")
    assert config == HIGH_CONTRAST_THEME
