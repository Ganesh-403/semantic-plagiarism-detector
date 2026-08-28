# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class TranslationService:
    """Service for handling translation operations."""

    def __init__(self, storage_file: str = "translations.json"):
        self.storage_file = storage_file
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Ensure the storage file exists."""
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, "w") as f:
                json.dump([], f)

    def save_translation(
        self,
        text: str,
        translated_text: str,
        source_lang: str = "en",
        target_lang: str = "es",
        confidence: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Save a translation record to storage.

        Args:
            text: Original text to translate
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            confidence: Translation confidence score (optional)

        Returns:
            The saved translation record

        Example:
            >>> service = TranslationService()
            >>> record = service.save_translation("Hello", "Hola", "en", "es")
            >>> print(record['timestamp'])
            '2026-01-21T10:30:00.123456+00:00'
        """
        # Create translation record with timezone-aware timestamp
        translation_record = {
            "text": text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "timestamp": datetime.now(timezone.utc).isoformat(),  # ✅ FIXED
            "confidence": confidence,
        }

        # Load existing translations
        try:
            with open(self.storage_file) as f:
                translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            translations = []

        # Append new translation
        translations.append(translation_record)

        # Save back to file
        with open(self.storage_file, "w") as f:
            json.dump(translations, f, indent=2)

        return translation_record

    def get_translations(
        self, limit: Optional[int] = None, source_lang: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve saved translations.

        Args:
            limit: Maximum number of translations to return
            source_lang: Filter by source language

        Returns:
            List of translation records
        """
        try:
            with open(self.storage_file) as f:
                translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        # Filter by source language if specified
        if source_lang:
            translations = [
                t for t in translations if t.get("source_lang") == source_lang
            ]

        # Sort by timestamp (newest first)
        translations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Apply limit
        if limit:
            translations = translations[:limit]

        return translations

    def get_translation_by_text(self, text: str) -> Optional[dict[str, Any]]:
        """Retrieve a translation by original text."""
        try:
            with open(self.storage_file) as f:
                translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        for record in translations:
            if record.get("text") == text:
                return record

        return None

    def delete_translation(self, text: str) -> bool:
        """Delete a translation record by text."""
        try:
            with open(self.storage_file) as f:
                translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        initial_count = len(translations)
        translations = [t for t in translations if t.get("text") != text]

        if len(translations) < initial_count:
            with open(self.storage_file, "w") as f:
                json.dump(translations, f, indent=2)
            return True

        return False

    def clear_all_translations(self) -> None:
        """Clear all translation records."""
        with open(self.storage_file, "w") as f:
            json.dump([], f)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about saved translations."""
        try:
            with open(self.storage_file) as f:
                translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"total": 0}

        # Count by language
        lang_stats = {}
        for record in translations:
            target = record.get("target_lang", "unknown")
            lang_stats[target] = lang_stats.get(target, 0) + 1

        return {
            "total": len(translations),
            "by_language": lang_stats,
            "has_confidence": sum(
                1 for t in translations if t.get("confidence") is not None
            ),
            "oldest": translations[-1].get("timestamp") if translations else None,
            "newest": translations[0].get("timestamp") if translations else None,
        }


def save_translation_simple(
    text: str, translated_text: str, **kwargs
) -> dict[str, Any]:
    """
    Simple function wrapper for saving translations.

    This is the function referenced in issue #2988 (line 149).

    Args:
        text: Original text
        translated_text: Translated text
        **kwargs: Additional arguments (source_lang, target_lang, confidence)

    Returns:
        Saved translation record

    Example:
        >>> record = save_translation_simple("Hello", "Hola")
        >>> print(record['timestamp'])
        '2026-01-21T10:30:00.123456+00:00'
    """
    service = TranslationService()
    return service.save_translation(text, translated_text, **kwargs)


class TranslationRecord:
    """Data class for translation records."""

    def __init__(
        self,
        text: str,
        translated_text: str,
        source_lang: str = "en",
        target_lang: str = "es",
        confidence: Optional[float] = None,
    ):
        self.text = text
        self.translated_text = translated_text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.confidence = confidence
        self.timestamp = datetime.now(timezone.utc)  # ✅ FIXED (timezone-aware)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "text": self.text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslationRecord":
        """Create record from dictionary."""
        record = cls(
            text=data["text"],
            translated_text=data["translated_text"],
            source_lang=data.get("source_lang", "en"),
            target_lang=data.get("target_lang", "es"),
            confidence=data.get("confidence"),
        )
        # Parse timestamp if present
        if "timestamp" in data:
            record.timestamp = datetime.fromisoformat(data["timestamp"])
        return record

    def __repr__(self) -> str:
        return f"TranslationRecord(text='{self.text[:20]}...', target='{self.target_lang}')"


# Export for CLI usage
__all__ = ["TranslationService", "save_translation_simple", "TranslationRecord"]
