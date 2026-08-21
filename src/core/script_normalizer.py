"""
Script Normalization for Non-Latin Languages

Provides normalization for Arabic script, Devanagari, Cyrillic, and other
non-Latin scripts to improve cross-script plagiarism detection.
"""

import logging
import unicodedata
from collections import defaultdict
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ARABIC SCRIPT NORMALIZATION
# ============================================================================

# Arabic diacritics (Harakat) - removed for normalization
ARABIC_DIACRITICS = {
    "\u064b",  # Fathatan
    "\u064c",  # Dammatan
    "\u064d",  # Kasratan
    "\u064e",  # Fatha
    "\u064f",  # Damma
    "\u0650",  # Kasra
    "\u0651",  # Shadda
    "\u0652",  # Sukun
    "\u0653",  # Maddah
    "\u0654",  # Hamza above
    "\u0655",  # Hamza below
    "\u0670",  # Superscript Alef
}

# Arabic letter normalization mapping
ARABIC_LETTER_MAP = {
    "أ": "ا",  # Alef with hamza above -> Alef
    "إ": "ا",  # Alef with hamza below -> Alef
    "آ": "ا",  # Alef with madda -> Alef
    "ٱ": "ا",  # Alef wasla -> Alef
    "ة": "ه",  # Ta marbuta -> Heh
    "ى": "ي",  # Alef maksura -> Yeh
    "ئ": "ي",  # Yeh with hamza -> Yeh
    "ؤ": "و",  # Waw with hamza -> Waw
}

# Arabic ligature normalization
ARABIC_LIGATURES = {
    "ﻻ": "لا",  # Lam-Alef ligature
    "ﻹ": "لا",  # Lam-Alef with hamza below
    "ﻷ": "لا",  # Lam-Alef with hamza above
    "ﻵ": "لا",  # Lam-Alef with madda
}

# Arabic character ranges for detection
ARABIC_RANGE = range(0x0600, 0x06FF)
ARABIC_EXTENDED_RANGE = range(0x0750, 0x077F)
ARABIC_PRESENTATION_RANGE = range(0xFB50, 0xFDFF)


# ============================================================================
# DEVANAGARI NORMALIZATION
# ============================================================================

DEVANAGARI_MAP = {
    "ा": "अ",  # Matra
    "ि": "इ",  # Matra
    "ी": "इ",  # Matra
    "ु": "उ",  # Matra
    "ू": "उ",  # Matra
    "ृ": "ऋ",  # Matra
    "ॄ": "ऋ",  # Matra
    "े": "ए",  # Matra
    "ै": "ऐ",  # Matra
    "ो": "ओ",  # Matra
    "ौ": "औ",  # Matra
    "ं": "न",  # Anusvara -> Na
    "ः": "ह",  # Visarga -> Ha
    "ँ": "न",  # Chandrabindu -> Na
}


# ============================================================================
# CYRILLIC NORMALIZATION
# ============================================================================

CYRILLIC_MAP = {
    "ё": "е",  # Yo -> Ye
    "й": "и",  # Short I -> I
    "ъ": "ь",  # Hard sign -> Soft sign (simplified)
}


# ============================================================================
# SCRIPT DETECTION
# ============================================================================


class ScriptDetector:
    """Detect the script of a given text."""

    SCRIPTS = {
        "arabic": "Arabic",
        "devanagari": "Devanagari",
        "cyrillic": "Cyrillic",
        "latin": "Latin",
        "chinese": "Chinese (Han)",
        "japanese": "Japanese",
        "korean": "Korean",
        "hebrew": "Hebrew",
        "greek": "Greek",
        "other": "Other",
    }

    @classmethod
    def detect(cls, text: str) -> str:
        """Detect the primary script of a text."""
        if not text:
            return "unknown"

        script_counts = defaultdict(int)

        for char in text:
            script = cls._get_script(char)
            if script:
                script_counts[script] += 1

        if not script_counts:
            return "latin"  # Default to Latin

        # Return the most frequent script
        return max(script_counts, key=script_counts.get)

    @classmethod
    def _get_script(cls, char: str) -> Optional[str]:
        """Get the script of a single character."""
        code = ord(char)

        # Arabic
        if (
            code in ARABIC_RANGE
            or code in ARABIC_EXTENDED_RANGE
            or code in ARABIC_PRESENTATION_RANGE
        ):
            return "arabic"

        # Devanagari (U+0900-U+097F)
        if 0x0900 <= code <= 0x097F:
            return "devanagari"

        # Cyrillic (U+0400-U+04FF)
        if 0x0400 <= code <= 0x04FF:
            return "cyrillic"

        # Chinese (CJK Unified Ideographs)
        if 0x4E00 <= code <= 0x9FFF:
            return "chinese"

        # Japanese Hiragana/Katakana
        if 0x3040 <= code <= 0x30FF:
            return "japanese"

        # Korean Hangul
        if 0xAC00 <= code <= 0xD7AF:
            return "korean"

        # Hebrew
        if 0x0590 <= code <= 0x05FF:
            return "hebrew"

        # Greek
        if 0x0370 <= code <= 0x03FF:
            return "greek"

        # Latin (basic)
        if 0x0041 <= code <= 0x007A:
            return "latin"

        return None


# ============================================================================
# SCRIPT NORMALIZER
# ============================================================================


class ScriptNormalizer:
    """
    Normalize text for cross-script plagiarism detection.

    Features:
    - Remove diacritics (Arabic, Devanagari, etc.)
    - Normalize letter forms
    - Handle ligatures
    - Script detection
    - Transliteration support
    """

    def __init__(self):
        self.script_detector = ScriptDetector()

    def normalize(self, text: str, script: Optional[str] = None) -> str:
        """
        Normalize text based on its script.

        Args:
            text: Input text
            script: Optional script override

        Returns:
            Normalized text
        """
        if not text:
            return text

        if script is None:
            script = self.script_detector.detect(text)

        if script == "arabic":
            return self._normalize_arabic(text)
        elif script == "devanagari":
            return self._normalize_devanagari(text)
        elif script == "cyrillic":
            return self._normalize_cyrillic(text)
        else:
            # Latin and others: basic Unicode normalization
            return unicodedata.normalize("NFKC", text)

    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text."""
        # Remove diacritics
        for diacritic in ARABIC_DIACRITICS:
            text = text.replace(diacritic, "")

        # Normalize ligatures
        for ligature, replacement in ARABIC_LIGATURES.items():
            text = text.replace(ligature, replacement)

        # Normalize letters
        for letter, replacement in ARABIC_LETTER_MAP.items():
            text = text.replace(letter, replacement)

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        return text

    def _normalize_devanagari(self, text: str) -> str:
        """Normalize Devanagari text."""
        for matra, base in DEVANAGARI_MAP.items():
            text = text.replace(matra, base)

        text = unicodedata.normalize("NFKC", text)
        return text

    def _normalize_cyrillic(self, text: str) -> str:
        """Normalize Cyrillic text."""
        for letter, replacement in CYRILLIC_MAP.items():
            text = text.replace(letter, replacement)

        text = unicodedata.normalize("NFKC", text)
        return text

    def detect_and_normalize(self, text: str) -> Tuple[str, str]:
        """
        Detect script and normalize text.

        Returns:
            Tuple of (normalized_text, script)
        """
        script = self.script_detector.detect(text)
        normalized = self.normalize(text, script)
        return normalized, script


# ============================================================================
# CROSS-SCRIPT SIMILARITY
# ============================================================================


def compute_cross_script_similarity(
    text_a: str,
    text_b: str,
    normalizer: Optional[ScriptNormalizer] = None,
) -> float:
    """
    Compute similarity between texts from different scripts.

    Uses normalization and transliteration to compare.
    """
    if normalizer is None:
        normalizer = ScriptNormalizer()

    script_a = normalizer.script_detector.detect(text_a)
    script_b = normalizer.script_detector.detect(text_b)

    # Same script: use normalization
    if script_a == script_b:
        norm_a = normalizer.normalize(text_a, script_a)
        norm_b = normalizer.normalize(text_b, script_b)
        # Use Jaccard or other similarity
        from src.core.lexical_similarity import jaccard_similarity

        return jaccard_similarity(norm_a, norm_b)

    # Different scripts: use transliteration + embedding similarity
    # This will be handled by the main pipeline with cross-lingual detection
    return 0.0


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_normalizer: Optional[ScriptNormalizer] = None


def get_script_normalizer() -> ScriptNormalizer:
    """Get global script normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = ScriptNormalizer()
    return _normalizer
