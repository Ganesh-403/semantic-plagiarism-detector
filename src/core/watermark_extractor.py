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

"""
src/core/watermark_extractor.py
-------------------------------
Watermark Extraction and Decoding Engine.

Scans incoming text documents for invisible Zero-Width Character (ZWC)
watermarks embedded by the watermark_engine. Decodes the binary payload
to recover the original watermark ID, which can then be looked up in the
database to identify the leak source.
"""

import logging
import re
from typing import Optional

from src.security.watermark_engine import ZWC_END, ZWC_ONE, ZWC_START, ZWC_ZERO

logger = logging.getLogger(__name__)

# Regex pattern to detect the presence of any zero-width characters
ZWC_PATTERN = re.compile(f"[{ZWC_ZERO}{ZWC_ONE}{ZWC_START}{ZWC_END}]")


def _zwc_to_binary(zwc_str: str) -> str:
    """Convert a string of Zero-Width Characters back into a binary string."""
    binary_chars = []
    for char in zwc_str:
        if char == ZWC_ZERO:
            binary_chars.append("0")
        elif char == ZWC_ONE:
            binary_chars.append("1")
        # Ignore start/end markers during binary conversion
    return "".join(binary_chars)


def _binary_to_hex(binary_str: str) -> str:
    """Convert a binary string back into a hexadecimal string."""
    # Ensure the binary string is exactly 128 bits (32 hex chars)
    if len(binary_str) < 128:
        binary_str = binary_str.zfill(128)
    elif len(binary_str) > 128:
        binary_str = binary_str[:128]

    int_val = int(binary_str, 2)
    return hex(int_val)[2:].zfill(32)


def extract_watermark(text: str) -> Optional[str]:
    """Scan text for an embedded watermark and return the decoded ID.

    Args:
        text: The text content to scan.

    Returns:
        The 32-character hexadecimal watermark ID if found, else None.
    """
    if not text:
        return None

    # Quick check: does the text contain any ZWC characters at all?
    if not ZWC_PATTERN.search(text):
        return None

    # Extract the payload between START and END markers
    # We use a regex to find the exact sequence
    pattern = re.compile(f"{ZWC_START}([{ZWC_ZERO}{ZWC_ONE}]+){ZWC_END}")
    match = pattern.search(text)

    if not match:
        # Fallback: try to decode any contiguous block of ZWC_ZERO and ZWC_ONE
        # This handles cases where the end marker was stripped by a platform
        raw_zwc = ZWC_PATTERN.findall(text)
        cleaned_zwc = "".join([c for c in raw_zwc if c in (ZWC_ZERO, ZWC_ONE)])
        if len(cleaned_zwc) >= 128:
            binary_payload = _zwc_to_binary(cleaned_zwc[:128])
            return _binary_to_hex(binary_payload)
        return None

    zwc_payload = match.group(1)
    binary_payload = _zwc_to_binary(zwc_payload)
    watermark_id = _binary_to_hex(binary_payload)

    logger.info("Successfully extracted watermark ID: %s", watermark_id)
    return watermark_id


def strip_watermarks(text: str) -> str:
    """Remove all zero-width watermark characters from the text.

    Useful for cleaning text before passing it to the semantic similarity
    pipeline, ensuring the invisible characters don't affect tokenization
    or embedding generation.
    """
    if not text:
        return text
    return ZWC_PATTERN.sub("", text)
