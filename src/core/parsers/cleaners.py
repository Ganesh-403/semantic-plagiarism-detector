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

"""src/core/parsers/cleaners.py - Text normalization, cleaning, and preprocessing strategies."""

import logging
import os
import re
import string
import unicodedata
from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect

from src.core.translator import translate_text

DetectorFactory.seed = 0
logger = logging.getLogger(__name__)

ZERO_WIDTH_CHARS_PATTERN = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u202a\u202b\u202c\u202d\u202e\u2060\u2061\u2062\u2063\u2064]"
)

ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "did",
        "didn",
        "do",
        "does",
        "doesn",
        "doing",
        "don",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "has",
        "hasn",
        "have",
        "haven",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "mightn",
        "more",
        "most",
        "mustn",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "same",
        "shan",
        "she",
        "should",
        "shouldn",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn",
        "we",
        "were",
        "weren",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "wouldn",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_BIBLIOGRAPHY_HEADERS = re.compile(
    r"^\s*(References|Works\s+Cited|Bibliography|Citations|Reference\s+List|Sources)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_DATE_PATTERNS = [
    re.compile(
        r"\b(?:\d{1,2}[-/th|st|nd|rd ]+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/ ,]+\d{2,4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
]
_ORG_PATTERNS = [
    re.compile(
        r"\b[A-Z][a-zA-Z0-9&]+ (?:Inc|Corp|LLC|Ltd|University|College|Department|Association|Group|Foundation)\b"
    ),
]
_PERSON_PATTERNS = [
    re.compile(r"\b(?:Prof\.|Dr\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
]


def load_custom_stopwords(file_path: Optional[str] = None) -> frozenset:
    """Load custom domain-specific stopwords from a text file (one word per line).

    Error Recovery & Fault Tolerance:
    --------------------------------
    If `file_path` is not explicitly provided, the system checks the `STOPWORDS_FILE` environment variable.
    If `STOPWORDS_FILE` is not set or points to an empty string, an empty `frozenset()` is returned.

    If `STOPWORDS_FILE` points to a non-existent file or raises an `OSError` (e.g. permission denied, missing file,
    broken file descriptor), the error is caught, a warning is logged, and an empty `frozenset()` is returned
    to ensure the text extraction and cleaning pipeline does not crash.

    Args:
        file_path: Optional path to custom stopwords file. If None, reads from STOPWORDS_FILE environment variable.

    Returns:
        frozenset: Lowercase custom stopwords loaded from file, or empty frozenset on missing file or read failure.
    """
    path = file_path if file_path is not None else os.environ.get("STOPWORDS_FILE")

    if not path:
        return frozenset()

    try:
        with open(path, "r", encoding="utf-8") as f:
            return frozenset(line.strip().lower() for line in f if line.strip())
    except OSError as exc:
        logger.warning(
            f"[document_parser] Could not read custom stopwords file '{path}': {exc}"
        )
        return frozenset()


def get_stopwords() -> frozenset:
    """Return the combined set of standard and custom (domain-specific) stopwords."""
    return ENGLISH_STOPWORDS | load_custom_stopwords()


def sanitize_zero_width_characters(text: str, filename: Optional[str] = None) -> str:
    """Strips zero-width unicode characters (e.g. \\u200b) often used to bypass plagiarism checkers."""
    if not text:
        return text

    matches = ZERO_WIDTH_CHARS_PATTERN.findall(text)
    if matches:
        count = len(matches)
        target = f"in file '{filename}'" if filename else "in document text"
        logger.warning(
            f"[document_parser] Security warning: Found and stripped {count} zero-width unicode character(s) {target}."
        )
        return ZERO_WIDTH_CHARS_PATTERN.sub("", text)
    return text


def normalize_unicode_spaces(text: str) -> str:
    """Normalize special Unicode whitespace, zero-width characters, and full-width punctuation."""
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    unicode_mapping = {
        0x00A0: " ",
        0x2009: " ",
        0x200A: " ",
        0x202F: " ",
        0x205F: " ",
        0x3000: " ",
        0x00AD: "",
        0x200B: "",
        0x200C: "",
        0x200D: "",
        0xFEFF: "",
        0x2060: "",
        0x2028: "\n",
        0x2029: "\n\n",
    }

    text = text.translate(unicode_mapping)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def sanitize_unicode_spaces(text: str) -> str:
    """Replace special Unicode spaces with standard ASCII spaces."""
    if not text:
        return text
    return text.replace("\u00a0", " ").replace("\u2009", " ")


def detect_text_language(text: str) -> str:
    """Detect the language of a text chunk."""
    cleaned_text = text.strip()
    if len(cleaned_text) < 20:
        return "unknown"

    try:
        return detect(cleaned_text)
    except LangDetectException:
        return "unknown"


def strip_bibliography(text: str) -> str:
    """Remove everything from the first standalone bibliography header onward."""
    from src.core.parsers.docx_parser import ParsedDocxText

    structured_headings = getattr(text, "headings", None)
    plain_text = text.text if isinstance(text, ParsedDocxText) else text
    match = _BIBLIOGRAPHY_HEADERS.search(plain_text)
    if match:
        sliced_text = plain_text[: match.start()].rstrip()
        if structured_headings is not None:
            words_in_sliced = len(sliced_text.split())
            return ParsedDocxText(
                text=sliced_text, headings=structured_headings[:words_in_sliced]
            )
        return sliced_text
    return text


def clean_text(raw_text: str, remove_stopwords: bool = False) -> str:
    """Normalize whitespace and remove unwanted Unicode characters."""
    text = raw_text

    text = text.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "‘": "'",
                "’": "'",
                "—": "-",
                "–": "-",
            }
        )
    )

    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\u00a0\u200b]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    if remove_stopwords:
        words = text.split()
        stopwords = get_stopwords()
        filtered_words = [
            word
            for word in words
            if word.lower().strip(string.punctuation) not in stopwords
        ]
        text = " ".join(filtered_words)
    return text.strip()


def remove_ignore_phrases(text: str, ignore_phrases: str) -> str:
    """Remove specified ignore phrases from text."""
    if not ignore_phrases or not ignore_phrases.strip():
        return text

    phrases = [line.strip() for line in ignore_phrases.split("\n") if line.strip()]
    if not phrases:
        return text

    result = text
    for phrase in phrases:
        result = result.replace(phrase, "")

    return clean_text(result)


def prepare_text_for_embedding(text: str) -> dict:
    """Preserve the original text and prepare English text for embeddings."""
    original_text = text.strip()
    detected_language = detect_text_language(original_text)

    translated_text = original_text
    was_translated = False

    if detected_language not in {"en", "unknown"}:
        translated_result = translate_text(
            original_text,
            target_lang="en",
        )

        if translated_result and not translated_result.startswith(
            "(Translation Error:"
        ):
            translated_text = translated_result
            was_translated = True

    cleaned_text = clean_text(translated_text)

    return {
        "original_text": original_text,
        "detected_language": detected_language,
        "embedding_text": cleaned_text,
        "was_translated": was_translated,
    }


def mask_named_entities_in_text(text: str) -> str:
    """Replace recognized PERSON, ORGANIZATION, and DATE entities with [ENTITY_MASKED]."""
    if not text:
        return text

    masked = text

    try:
        import nltk

        try:
            tokens = nltk.word_tokenize(masked)
            pos_tags = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(pos_tags)
            entities = []
            for chunk in chunks:
                if hasattr(chunk, "label") and chunk.label() in (
                    "PERSON",
                    "ORGANIZATION",
                    "ORGANISATION",
                    "GPE",
                    "DATE",
                ):
                    entity_str = " ".join(c[0] for c in chunk)
                    entities.append(entity_str)
            for ent in sorted(entities, key=len, reverse=True):
                if len(ent) > 1:
                    masked = masked.replace(ent, "[ENTITY_MASKED]")
        except Exception:
            pass
    except ImportError:
        pass

    for pat in _DATE_PATTERNS:
        masked = pat.sub("[ENTITY_MASKED]", masked)
    for pat in _ORG_PATTERNS:
        masked = pat.sub("[ENTITY_MASKED]", masked)
    for pat in _PERSON_PATTERNS:
        masked = pat.sub("[ENTITY_MASKED]", masked)

    return masked


def normalize_extended_punctuation(text: str) -> str:
    """Replace curly quotes, em-dashes, and ellipsis with standard ASCII."""
    if not text:
        return text

    translation_table = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "…": "..."}
    )
    return text.translate(translation_table)


def normalize_unicode_nfc(text: str) -> str:
    """Convert input text to Unicode NFC canonical composition form."""
    if not text or not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def _strip_inline_markdown(line: str) -> str:
    """Strip bold, italic, strikethrough, inline code, and link markdown syntax."""
    line = re.sub(r"(\*\*|__|[*_])(.*?)\1", r"\2", line)
    line = re.sub(r"~~(.*?)~~", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    return line


def strip_markdown_syntax(raw_text: str) -> str:
    """Remove markdown syntax markers while preserving text content."""
    if not raw_text or not raw_text.strip():
        return ""

    lines = raw_text.splitlines()
    cleaned_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            cleaned_lines.append(line)
            continue

        if re.match(r"^#{1,6}\s+", stripped):
            line = re.sub(r"^#{1,6}\s+", "", stripped)

        if re.match(r"^[-*+]\s+", stripped):
            line = re.sub(r"^[-*+]\s+", "", stripped)

        if re.match(r"^\d+\.\s+", stripped):
            line = re.sub(r"^\d+\.\s+", "", stripped)

        if stripped.startswith(">"):
            line = re.sub(r"^>\s*", "", stripped)

        if re.match(r"^[*-]{3,}$", stripped):
            continue

        line = _strip_inline_markdown(line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
