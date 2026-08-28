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
src/security/watermark_engine.py
--------------------------------
Automated Document Watermarking Engine.

Embeds invisible, unique digital watermarks into text documents using
Zero-Width Character (ZWC) binary encoding. This allows the system to
cryptographically identify the original recipient of a leaked document.

The engine converts a unique identifier (e.g., UUID or User ID hash) into
a binary string, then maps each bit to a specific zero-width Unicode character.
These characters are invisible to the human eye and most standard text editors,
but can be extracted by the watermark_extractor module.
"""

import hashlib
import logging
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# Zero-Width Characters used for binary encoding
# U+200B: Zero Width Space (represents '0')
# U+200C: Zero Width Non-Joiner (represents '1')
# U+200D: Zero Width Joiner (used as a delimiter/start marker)
# U+FEFF: Zero Width No-Break Space (used as an end marker)
ZWC_ZERO = "\u200B"
ZWC_ONE = "\u200C"
ZWC_START = "\u200D"
ZWC_END = "\uFEFF"


def generate_watermark_id(user_id: str, document_hash: str) -> str:
    """Generate a deterministic, unique watermark ID for a user-document pair.

    Args:
        user_id: The ID of the user receiving the document.
        document_hash: The SHA-256 hash of the original document content.

    Returns:
        A 32-character hexadecimal string serving as the watermark payload.
    """
    # Combine user_id and document_hash to create a unique seed
    seed = f"{user_id}:{document_hash}"
    # Use SHA-256 and take the first 16 bytes (32 hex chars) for a compact payload
    hash_obj = hashlib.sha256(seed.encode("utf-8"))
    return hash_obj.hexdigest()[:32]


def _text_to_binary(text: str) -> str:
    """Convert a hexadecimal string into a binary string."""
    # Convert hex to integer, then to binary, stripping the '0b' prefix
    # Pad with zeros to ensure exact length (32 hex chars = 128 bits)
    int_val = int(text, 16)
    binary_str = bin(int_val)[2:].zfill(128)
    return binary_str


def _binary_to_zwc(binary_str: str) -> str:
    """Map a binary string to Zero-Width Characters."""
    zwc_chars = []
    for bit in binary_str:
        if bit == "0":
            zwc_chars.append(ZWC_ZERO)
        elif bit == "1":
            zwc_chars.append(ZWC_ONE)
    return "".join(zwc_chars)


def embed_watermark(
    text: str, user_id: str, document_hash: str, strategy: str = "append"
) -> tuple[str, str]:
    """Embed an invisible watermark into the provided text.

    Args:
        text: The original text content to watermark.
        user_id: The ID of the recipient.
        document_hash: The hash of the document.
        strategy: Embedding strategy ('append', 'prepend', or 'distribute').
                  'append' adds the watermark to the end of the text.
                  'prepend' adds it to the beginning.
                  'distribute' spreads it across paragraph breaks.

    Returns:
        A tuple containing (watermarked_text, watermark_id).
    """
    if not text or not text.strip():
        logger.warning("Cannot embed watermark in empty text.")
        return text, ""

    watermark_id = generate_watermark_id(user_id, document_hash)
    binary_payload = _text_to_binary(watermark_id)
    zwc_payload = ZWC_START + _binary_to_zwc(binary_payload) + ZWC_END

    if strategy == "prepend":
        watermarked_text = zwc_payload + text
    elif strategy == "distribute":
        # Insert watermark at the middle of the text to survive truncation
        mid_point = len(text) // 2
        watermarked_text = text[:mid_point] + zwc_payload + text[mid_point:]
    else:
        # Default: append
        watermarked_text = text + zwc_payload

    logger.info(
        "Embedded watermark %s for user %s using strategy '%s'.",
        watermark_id,
        user_id,
        strategy,
    )
    return watermarked_text, watermark_id
