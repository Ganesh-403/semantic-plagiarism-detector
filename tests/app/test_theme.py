from app.theme import (COLORS, badge_html, severity_tier, tier_color,
                       tier_from_severity_label)


def test_severity_tier():
    # Test with threshold 0.75
    assert severity_tier(0.95, 0.75) == "high"
    assert severity_tier(0.90, 0.75) == "high"
    assert severity_tier(0.85, 0.75) == "medium"
    assert severity_tier(0.75, 0.75) == "medium"
    assert severity_tier(0.70, 0.75) == "low"
    assert severity_tier(0.00, 0.75) == "low"

    # Test with threshold 0.59
    assert severity_tier(0.65, 0.59) == "medium"
    assert severity_tier(0.59, 0.59) == "medium"
    assert severity_tier(0.58, 0.59) == "low"


def test_tier_from_severity_label():
    assert tier_from_severity_label("🔴 High") == "high"
    assert tier_from_severity_label("🟡 Medium") == "medium"
    assert tier_from_severity_label("HIGH") == "high"
    assert tier_from_severity_label("Warning") == "medium"
    assert tier_from_severity_label("Low") == "low"
    assert tier_from_severity_label("unknown") == "low"


def test_tier_color():
    assert tier_color("high") == COLORS["danger"]
    assert tier_color("medium") == COLORS["warning"]
    assert tier_color("low") == COLORS["success"]
    assert tier_color("unknown") == COLORS["neutral_soft"]


def test_badge_html_default():
    html = badge_html("high")
    assert "background-color: " + COLORS["danger_soft"] in html
    assert "color: " + COLORS["danger"] in html
    assert "🔴 High" in html

    html_med = badge_html("medium")
    assert "background-color: " + COLORS["warning_soft"] in html_med
    assert "color: " + COLORS["warning"] in html_med
    assert "🟡 Medium" in html_med

    html_low = badge_html("low")
    assert "background-color: " + COLORS["success_soft"] in html_low
    assert "color: " + COLORS["success"] in html_low
    assert "🟢 Low" in html_low


def test_badge_html_custom_label():
    html = badge_html("high", "Similarity: 95.0%")
    assert "Similarity: 95.0%" in html
    assert "🔴 High" not in html
