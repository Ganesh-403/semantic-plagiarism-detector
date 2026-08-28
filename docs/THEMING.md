# UI Customization and Theme Guide

## Overview

(explain that app/theme.py + app/css_constants.py control all visual styling)

## CSS Architecture

(explain inject_css() builds one big <style> block injected via st.markdown,
 driven by CSS variables like --background, --ink, --accent, and list the
 major sections: Sidebar, Badge, Login container, Empty state, Pipeline
 progress, Skeletons, Responsive rules, etc.)

## Theme Tokens

(explain the THEMES dict has "Light" and "Dark" palettes, list the token
 names: background, surface, card, ink, muted, accent, border, input,
 neutral_soft, danger, danger_soft, warning, warning_soft, success,
 success_soft — and mention get_colors(), set_theme(), sanitize_theme_colors())

## Component Styling / CSS Class Naming Convention

(explain the constants in css_constants.py, e.g. BADGE = "badge",
 and how helper functions like badge_html(), empty_state_html() reference them
 instead of hardcoding class name strings)

## Adding a New CSS Class (code example)

(show a short example: add a constant to css_constants.py, add a matching
 CSS rule inside inject_css(), then use it in a helper function)
