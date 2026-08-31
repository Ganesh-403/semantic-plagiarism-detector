"""
src/core/encoding.py
--------------------
Utilities for detecting, normalizing, and repairing text encoding issues
commonly encountered when extracting text from PDFs, DOCX files, and
legacy web scrapes.

Real-world documents often suffer from "mojibake" — character garbling
caused by interpreting text in the wrong encoding (e.g., reading UTF-8
bytes as Windows-1252 or Latin-1). This module provides lookup tables
and helper functions to reverse these transformations.

Repair strategy
~~~~~~~~~~~~~~~
Repair happens in two passes, most reliable first:

1. **Round-trip repair.** Genuine Windows-1252 mojibake is, by definition,
   UTF-8 bytes that were decoded with the wrong single-byte codec. Encoding
   the text back to ``cp1252`` and decoding it as UTF-8 reverses the damage
   exactly. If either step fails, the text was not uniformly garbled and the
   original is left untouched. This pass is inherently safe: correctly encoded
   text almost never survives the UTF-8 decode.

2. **Table-driven repair.** Documents that mix clean and garbled runs (a
   correctly decoded ``é`` next to a garbled ``Ã»``) cannot round-trip,
   because the clean bytes are not valid UTF-8. For those, the
   ``MOJIBAKE_REPLACEMENTS`` table rewrites the individual garbled sequences.

Both passes are conservative: a sequence is only rewritten when it is
unambiguously mojibake, so correctly encoded documents pass through unchanged.
"""

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Mojibake Replacement Dictionary ───────────────────────────────────────────
# When a UTF-8 encoded character (which may use 2-4 bytes) is read by a system
# expecting single-byte Windows-1252, each byte is mapped to a distinct
# character. The UTF-8 bytes for 'ñ' (0xC3 0xB1), for instance, are interpreted
# as 'Ã' (0xC3) and '±' (0xB1), producing the garbled string "Ã±".
#
# Rather than transcribing those byte pairs by hand — which is how the wrong
# entries crept in previously — the table is derived from that definition. For
# each character we care about we encode it as UTF-8 and decode those bytes as
# Windows-1252; the result is exactly the mojibake a reader would see.
#
# Characters whose UTF-8 bytes include one of the five positions Windows-1252
# leaves undefined (0x81, 0x8D, 0x8F, 0x90, 0x9D) cannot be produced by this
# corruption path at all, so they are skipped automatically.
#
# Every generated key is at least two characters long. Single characters are
# deliberately impossible here: a bare "Â" or "Ã" is ordinary text in French,
# Portuguese, Romanian and Vietnamese, so rewriting one on sight corrupts clean
# documents.

# Latin-1 supplement (accented letters, symbols) plus the punctuation that
# Microsoft Word inserts automatically and that dominates real-world mojibake.
_MOJIBAKE_SOURCE_CODEPOINTS = (
    tuple(range(0x00A0, 0x0100))  # Latin-1 supplement: ¡ ¢ £ © « ° » À-ÿ
    + tuple(range(0x0100, 0x0180))  # Latin Extended-A: ā ē ī ł ő ū ž …
    + (
        0x2013,  # – en dash
        0x2014,  # — em dash
        0x2018,  # ' left single quote
        0x2019,  # ' right single quote / apostrophe
        0x201A,  # ‚ single low quote
        0x201C,  # " left double quote
        0x201D,  # " right double quote
        0x201E,  # „ double low quote
        0x2020,  # † dagger
        0x2021,  # ‡ double dagger
        0x2022,  # • bullet
        0x2026,  # … horizontal ellipsis
        0x2030,  # ‰ per mille
        0x2039,  # ‹ single left angle quote
        0x203A,  # › single right angle quote
        0x20AC,  # € euro sign
        0x2122,  # ™ trade mark
    )
)


def _build_mojibake_table() -> dict[str, str]:
    """Derive the mojibake lookup table from the UTF-8/Windows-1252 mismatch.

    Returns:
        A mapping of garbled sequence to the character it should be restored
        to. Keys are always two or more characters long.
    """
    table: dict[str, str] = {}

    for codepoint in _MOJIBAKE_SOURCE_CODEPOINTS:
        character = chr(codepoint)
        try:
            garbled = character.encode("utf-8").decode("cp1252")
        except UnicodeDecodeError:
            # The UTF-8 bytes hit one of the undefined Windows-1252 positions,
            # so this character cannot be garbled this way.
            continue

        # A one-character "garble" would mean the sequence is indistinguishable
        # from ordinary text; never add those.
        if len(garbled) < 2:
            continue

        table[garbled] = character

    return table


# Windows-1252 leaves 0x81, 0x8D, 0x8F, 0x90 and 0x9D undefined. Many extraction
# pipelines drop those bytes rather than raising, which truncates the garbled
# sequence and puts it out of reach of the generated table. The common casualty
# is the right double quotation mark (U+201D, UTF-8 E2 80 9D): the closing half
# of every quoted passage in a Word document arrives as a bare "â€".
#
# Only sequences that stay two characters or longer after the drop are listed.
# The two-byte accented capitals (Á, Í, Ï, Ð, Ý) collapse to a lone "Ã" and are
# therefore indistinguishable from ordinary text, so they are left alone.
_TRUNCATED_MOJIBAKE_REPLACEMENTS: dict[str, str] = {
    "â€": "”",  # U+201D with its undefined 0x9D byte dropped
}

MOJIBAKE_REPLACEMENTS: dict[str, str] = {
    **_build_mojibake_table(),
    **_TRUNCATED_MOJIBAKE_REPLACEMENTS,
}

# Compile a regex pattern for efficient bulk replacement.
# We sort the keys by length (longest first) to ensure three-character
# sequences like "â€œ" are matched before two-character prefixes like "â€".
_SORTED_MOJIBAKE_KEYS = sorted(MOJIBAKE_REPLACEMENTS.keys(), key=len, reverse=True)
_MOJIBAKE_PATTERN = re.compile(
    "|".join(re.escape(key) for key in _SORTED_MOJIBAKE_KEYS)
)

# ── Contextual "Â" (U+00C2) Prefix Removal ────────────────────────────────────
# Code points U+0080–U+00BF encode in UTF-8 as the two bytes 0xC2 0xNN, so read
# back as Windows-1252 they become "Â" followed by whatever 0xNN maps to. The
# table above already restores the printable members of that range (Â© → ©,
# Â£ → £). What is left over are the sequences whose target is a C1 control
# character, which is never meaningful in prose — there the "Â" is simply a
# stray artifact and the following symbol is the real content, e.g. "100Â€".
#
# The tell is always the *second* character. A "Â" followed by an ordinary
# letter — "Âme", "Ângela" — is real text and must survive untouched.
_C2_CONTINUATION_CHARS = bytes(range(0x80, 0xC0)).decode("cp1252", errors="ignore")
_C2_PREFIX_RE = re.compile(f"Â([{re.escape(_C2_CONTINUATION_CHARS)}])")

# Control characters that should never appear in repaired prose. Their presence
# means a round trip "succeeded" into nonsense rather than into real text.
_SUSPICIOUS_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f�]")

# Upper bound on repair passes. Convergence is guaranteed well before this
# because every pass that changes the text also shortens it; the cap exists so
# an unforeseen table entry can never spin the loop.
_MAX_REPAIR_PASSES = 8


def _round_trip_repair(text: str) -> Optional[str]:
    """Reverse uniform Windows-1252 mojibake via an encode/decode round trip.

    Text that is genuinely garbled consists of UTF-8 bytes that were decoded
    with Windows-1252. Re-encoding to ``cp1252`` recovers those original bytes,
    and decoding them as UTF-8 recovers the original characters.

    Args:
        text: The text to attempt to repair.

    Returns:
        The repaired text, or ``None`` when the round trip does not apply —
        either because the text is not representable in Windows-1252, because
        the recovered bytes are not valid UTF-8, or because the result contains
        control characters that indicate a spurious match.
    """
    try:
        recovered = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None

    # A round trip that lands on control or replacement characters has not
    # recovered prose; treat it as a false positive and decline.
    if _SUSPICIOUS_CONTROL_RE.search(recovered):
        return None

    return recovered


def _table_repair(text: str) -> str:
    """Rewrite individual mojibake sequences using the replacement table.

    Applied to documents that mix clean and garbled runs, where a whole-string
    round trip is impossible. The contextual "Â" prefix rule runs afterwards to
    clear stray prefixes the table does not cover.
    """
    repaired = _MOJIBAKE_PATTERN.sub(
        lambda match: MOJIBAKE_REPLACEMENTS[match.group(0)],
        text,
    )
    return _C2_PREFIX_RE.sub(r"\1", repaired)


def normalize_encoding(text: str) -> str:
    """Repair common mojibake patterns in extracted text.

    Attempts a Windows-1252 → UTF-8 round trip first, which reverses uniformly
    garbled text exactly. Falls back to table-driven substitution for documents
    that contain a mix of clean and garbled runs.

    This function is idempotent — applying it multiple times to the same text
    yields the same result — and text that is already correctly encoded is
    returned unchanged.

    Args:
        text: The raw extracted text potentially containing mojibake.

    Returns:
        The cleaned text with mojibake patterns replaced. Returns an empty
        string if the input is None or not a string.

    Examples:
        >>> normalize_encoding("The cafÃ© was beautifÃ»l.")
        'The café was beautifûl.'

        >>> normalize_encoding("He said, â€œHello!â€")
        'He said, “Hello!”'

        >>> normalize_encoding("Âme et Ângela")
        'Âme et Ângela'

    Handled Patterns:
        See the module-level ``MOJIBAKE_REPLACEMENTS`` dictionary for the
        complete list of supported garbling patterns.
    """
    if not text or not isinstance(text, str):
        return ""

    # Repairing one layer can expose another: text that was encoded wrongly
    # twice needs two passes, and a substitution can leave a stray prefix next
    # to the symbol it belongs with. Iterate until the text stops changing so
    # the result is a fixed point and the function is genuinely idempotent.
    #
    # Every repair replaces a multi-character sequence with a single character,
    # so each pass that changes anything strictly shortens the text and the
    # loop is guaranteed to terminate. The cap is a defensive backstop only.
    repaired = text
    for _ in range(_MAX_REPAIR_PASSES):
        round_tripped = _round_trip_repair(repaired)
        candidate = (
            round_tripped if round_tripped is not None else _table_repair(repaired)
        )

        if candidate == repaired:
            return repaired
        repaired = candidate

    logger.debug(
        "normalize_encoding hit the %d-pass repair cap; returning the latest result.",
        _MAX_REPAIR_PASSES,
    )
    return repaired


def detect_mojibake(text: str, threshold: float = 0.05) -> bool:
    """Heuristically detect if a text string contains significant mojibake.

    Calculates the ratio of known mojibake sequences to the total text length.
    If the ratio exceeds the threshold, the text is likely garbled and should
    be passed through ``normalize_encoding()``.

    Only sequences that are unambiguously mojibake are counted. A lone "Â" or
    "Ã" is ordinary text in several Latin-script languages and never counts
    toward the ratio on its own.

    Args:
        text: The text to inspect.
        threshold: Minimum ratio of mojibake characters to trigger a positive
                   detection. Defaults to 0.05 (5%).

    Returns:
        True if the text appears to contain mojibake, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    # Count occurrences of known mojibake patterns.
    matches = _MOJIBAKE_PATTERN.findall(text)
    prefix_matches = _C2_PREFIX_RE.findall(text)

    if not matches and not prefix_matches:
        return False

    # Total characters consumed by mojibake patterns. Each contextual "Â"
    # prefix match consumes two characters: the prefix and its continuation.
    mojibake_chars = sum(len(m) for m in matches) + 2 * len(prefix_matches)
    ratio = mojibake_chars / len(text)

    return ratio >= threshold
