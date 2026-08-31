"""
test_translation_cache_primary_key_collision_issue_2983.py
------------------------------------------------------------
Unit tests for Issue #2983: Primary Key collision in Translation Cache when the same
source text is translated to multiple target languages.
"""

from __future__ import annotations

import hashlib
from src.db.translation_cache import _hash_text_simple


def test_hash_text_simple_payload_includes_language_pair():
    """Verify _hash_text_simple includes source_lang and target_lang in hash payload."""
    text = "Hello"
    source_lang = "en"
    target_lang = "es"

    expected_payload = f"{source_lang}:{target_lang}:{text}"
    expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()

    assert _hash_text_simple(text, source_lang=source_lang, target_lang=target_lang) == expected_hash


def test_hash_text_simple_differs_by_target_language():
    """Verify _hash_text_simple produces distinct hashes when translating same text to different target languages."""
    text = "Hello"

    hash_es = _hash_text_simple(text, source_lang="en", target_lang="es")
    hash_fr = _hash_text_simple(text, source_lang="en", target_lang="fr")

    assert hash_es != hash_fr


def test_hash_text_simple_differs_by_source_language():
    """Verify _hash_text_simple produces distinct hashes for different source languages."""
    text = "Chat"

    hash_en = _hash_text_simple(text, source_lang="en", target_lang="es")
    hash_fr = _hash_text_simple(text, source_lang="fr", target_lang="es")

    assert hash_en != hash_fr
