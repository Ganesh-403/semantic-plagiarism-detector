"""Encoding fallback utilities for document parsing."""

This module provides helpers for normalizing common encoding issues
such as mojibake and replacement characters.
from __future__ import annotations

def normalize_encoding(text: str) -> str:
    """Replace common mojibake with the intended characters."""
    replacements = {
        "\ufffd": "",
        "Ã©": "é",
        "Ã¡": "á",
        "Ã¼": "ü",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text
