"""
src/security/obfuscation_detector.py
------------------------------------
Security module for detecting adversarial text obfuscation techniques.

Students increasingly use adversarial techniques to bypass plagiarism and AI
detectors. These include injecting zero-width characters, replacing Latin
letters with Cyrillic homoglyphs, or hiding text using invisible Unicode
control characters. This module provides pattern matching and scoring
logic to identify these anomalies before the main similarity pipeline runs.
"""

import re
import logging
import unicodedata
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Zero-width and invisible Unicode character ranges
# U+200B to U+200F: Zero-width spaces, joiners, LTR/RTL marks
# U+FEFF: Zero-width no-break space (BOM)
# U+202A to U+202E: Bidi formatting controls
# U+2060 to U+206F: Invisible operators and word joiners
ZERO_WIDTH_PATTERN = re.compile(
    r"[\u200B-\u200F\uFEFF\u202A-\u202E\u2060-\u206F\u00AD]"
)

# Cyrillic homoglyphs that visually resemble Latin characters
# Mapping Cyrillic char -> Latin equivalent it is trying to mimic
CYRILLIC_HOMOGLYPHS: dict[str, str] = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    "і": "i",
    "ј": "j",
    "ѕ": "s",
    "А": "A",
    "Е": "E",
    "О": "O",
    "Р": "P",
    "С": "C",
    "У": "Y",
    "Х": "X",
    "І": "I",
    "Ј": "J",
    "Ѕ": "S",
}

# Build regex for quick homoglyph detection
HOMOGLYPH_PATTERN = re.compile(f"[{''.join(CYRILLIC_HOMOGLYPHS.keys())}]")


@dataclass
class ObfuscationReport:
    """Detailed report of obfuscation patterns found in a text string."""

    total_characters: int = 0
    zero_width_count: int = 0
    homoglyph_count: int = 0
    control_char_count: int = 0
    flagged_indices: list[int] = field(default_factory=list)
    obfuscation_score: float = 0.0
    is_suspicious: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a dictionary for JSON/DB storage."""
        return {
            "total_characters": self.total_characters,
            "zero_width_count": self.zero_width_count,
            "homoglyph_count": self.homoglyph_count,
            "control_char_count": self.control_char_count,
            "flagged_indices_count": len(self.flagged_indices),
            "obfuscation_score": self.obfuscation_score,
            "is_suspicious": self.is_suspicious,
        }


def detect_zero_width_chars(text: str) -> tuple[int, list[int]]:
    """Identify zero-width and invisible characters in the text.

    Args:
        text: The input text to scan.

    Returns:
        A tuple containing the total count of zero-width characters
        and a list of their specific character indices.
    """
    count = 0
    indices = []
    for i, char in enumerate(text):
        if ZERO_WIDTH_PATTERN.match(char):
            count += 1
            indices.append(i)
    return count, indices


def detect_homoglyphs(text: str) -> tuple[int, list[int]]:
    """Identify Cyrillic homoglyphs masquerading as Latin characters.

    Args:
        text: The input text to scan.

    Returns:
        A tuple containing the total count of homoglyphs and their indices.
    """
    count = 0
    indices = []
    for i, char in enumerate(text):
        if char in CYRILLIC_HOMOGLYPHS:
            count += 1
            indices.append(i)
    return count, indices


def detect_control_chars(text: str) -> tuple[int, list[int]]:
    """Identify invisible Unicode control characters (categories Cc and Cf).

    Characters already reported by :func:`detect_zero_width_chars` are excluded.
    Every member of ``ZERO_WIDTH_PATTERN`` is also category Cf, so counting them
    here as well would report each one twice and make the three counts on an
    ObfuscationReport sum to more than the document's length.

    Args:
        text: The input text to scan.

    Returns:
        A tuple containing the count of control characters and their indices.
    """
    count = 0
    indices = []
    for i, char in enumerate(text):
        if ZERO_WIDTH_PATTERN.match(char):
            # Already accounted for as a zero-width character.
            continue
        category = unicodedata.category(char)
        # Cc = Control, Cf = Format (excluding standard whitespace like \n, \t)
        if category in ("Cc", "Cf") and char not in ("\n", "\r", "\t"):
            count += 1
            indices.append(i)
    return count, indices


# Per-signal weights applied to the density of each obfuscation pattern.
ZERO_WIDTH_WEIGHT = 10
HOMOGLYPH_WEIGHT = 15
CONTROL_CHAR_WEIGHT = 12

# Ceiling on the density term. High enough that a saturated document clears any
# realistic suspicion threshold on density alone, while leaving room for the
# absolute penalties below to push a document the rest of the way to 1.0.
MAX_DENSITY_SCORE = 0.8


def calculate_obfuscation_score(report: ObfuscationReport) -> float:
    """Calculate a normalized obfuscation score between 0.0 and 1.0.

    The score is weighted based on the severity and density of the
    obfuscation patterns relative to the total text length. All three
    detected signals contribute: zero-width characters, Cyrillic homoglyphs,
    and invisible control characters.

    Args:
        report: The populated ObfuscationReport object.

    Returns:
        A float between 0.0 (clean) and 1.0 (highly obfuscated).
    """
    if report.total_characters == 0:
        return 0.0

    # Weighted density calculation.
    # Zero-width, homoglyphs and control characters are all high severity.
    zw_density = report.zero_width_count / report.total_characters
    hg_density = report.homoglyph_count / report.total_characters
    ctrl_density = report.control_char_count / report.total_characters

    base_score = min(
        MAX_DENSITY_SCORE,
        (zw_density * ZERO_WIDTH_WEIGHT)
        + (hg_density * HOMOGLYPH_WEIGHT)
        + (ctrl_density * CONTROL_CHAR_WEIGHT),
    )

    # Absolute threshold penalty: a large raw count is suspicious regardless of
    # how long the surrounding document is.
    absolute_penalty = 0.0
    if report.zero_width_count > 10:
        absolute_penalty += 0.3
    if report.homoglyph_count > 5:
        absolute_penalty += 0.2
    if report.control_char_count > 10:
        absolute_penalty += 0.2

    final_score = min(1.0, base_score + absolute_penalty)
    return round(final_score, 4)


def analyze_text_obfuscation(text: str, threshold: float = 0.15) -> ObfuscationReport:
    """Run all obfuscation detectors and generate a comprehensive report.

    Args:
        text: The raw extracted text from the document.
        threshold: The score threshold above which the document is
                   flagged as suspicious. Defaults to 0.15.

    Returns:
        A fully populated ObfuscationReport.
    """
    if not text:
        return ObfuscationReport()

    report = ObfuscationReport(total_characters=len(text))

    zw_count, zw_indices = detect_zero_width_chars(text)
    hg_count, hg_indices = detect_homoglyphs(text)
    ctrl_count, ctrl_indices = detect_control_chars(text)

    report.zero_width_count = zw_count
    report.homoglyph_count = hg_count
    report.control_char_count = ctrl_count

    # Merge and deduplicate indices
    all_indices = set(zw_indices + hg_indices + ctrl_indices)
    report.flagged_indices = sorted(list(all_indices))

    report.obfuscation_score = calculate_obfuscation_score(report)
    report.is_suspicious = report.obfuscation_score >= threshold

    if report.is_suspicious:
        logger.warning(
            "Obfuscation detected! Score: %.4f | ZW: %d | HG: %d | Ctrl: %d",
            report.obfuscation_score,
            zw_count,
            hg_count,
            ctrl_count,
        )

    return report
import re
import unicodedata

class ObfuscationDetector:
    def __init__(self):
        # 1. Zero-width and invisible control character tracking filters
        # Matches ZWSP, ZWNJ, ZWJ, BOM, and general format control characters (\p{Cf})
        self.invisible_chars_regex = re.compile(r'[\u200b-\u200d\ufeff\u200e\u200f\u202a-\u202e]')
        
        # 2. Cyrillic Homoglyphs frequently substituted into English text
        # Example: Cyrillic 'а' (U+0430) vs Latin 'a' (U+0061)
        self.cyrillic_homoglyphs = set(range(0x0400, 0x04FF))

    def detect_invisible_characters(self, text: str) -> list[int]:
        """Finds string indexes containing hidden Unicode format markers."""
        return [match.start() for match in self.invisible_chars_regex.finditer(text)]

    def detect_homoglyphs(self, text: str) -> list[int]:
        """Pins down mixed-script homoglyph character substitutions."""
        flagged_indices = []
        has_latin = any(unicodedata.name(c).startswith('LATIN') for c in text if c.isalpha())
        
        if has_latin:
            for idx, char in enumerate(text):
                if ord(char) in self.cyrillic_homoglyphs:
                    flagged_indices.append(idx)
        return flagged_indices

    def analyze_text(self, text: str) -> dict:
        """Runs the complete suite of text analysis sub-checks."""
        if not text:
            return {"obfuscation_score": 0.0, "invisible_indices": [], "homoglyph_indices": [], "is_flagged": False}

        invisible_indices = self.detect_invisible_characters(text)
        homoglyph_indices = self.detect_homoglyphs(text)
        
        total_violations = len(invisible_indices) + len(homoglyph_indices)
        total_chars = len(text)
        
        # Calculate percentage footprint density score
        obfuscation_score = round((total_violations / total_chars) * 100, 2) if total_chars > 0 else 0.0
        
        # Flag automatically if more than 1% of the document uses suspicious formatting
        is_flagged = obfuscation_score >= 1.0 or total_violations > 10

        return {
            "obfuscation_score": obfuscation_score,
            "invisible_indices": invisible_indices,
            "homoglyph_indices": homoglyph_indices,
            "is_flagged": is_flagged
        }
