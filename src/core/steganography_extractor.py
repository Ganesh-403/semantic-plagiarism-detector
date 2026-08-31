"""
src/core/steganography_extractor.py
-----------------------------------
Document Steganography Extraction Engine.

Parses DOCX/PDF for hidden text (white-on-white, micro-fonts, zero-width
payloads) to detect concealed cheating or AI prompt injections.
"""

import re
import zipfile
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Zero-width characters used for steganography
ZERO_WIDTH_CHARS = [
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non-Joiner
    "\u200d",  # Zero Width Joiner
    "\ufeff",  # Zero Width No-Break Space
    "\u2060",  # Word Joiner
]


def extract_zero_width_payloads(text: str) -> List[str]:
    """Extract zero-width character sequences from text.

    Args:
        text: The input document text.

    Returns:
        List of extracted zero-width payloads.
    """
    if not text:
        return []

    payloads = []
    current_payload = []

    for char in text:
        if char in ZERO_WIDTH_CHARS:
            current_payload.append(char)
        else:
            if current_payload:
                payloads.append("".join(current_payload))
                current_payload = []

    if current_payload:
        payloads.append("".join(current_payload))

    return payloads


def extract_hidden_docx_text(file_bytes: bytes) -> List[str]:
    """Extract white-on-white and micro-font text from DOCX files.

    Parses the document.xml inside the DOCX zip to find text runs with
    white color (#FFFFFF) or extremely small font sizes (< 2pt).

    Args:
        file_bytes: Raw bytes of the DOCX file.

    Returns:
        List of hidden text strings.
    """
    hidden_texts = []
    try:
        with zipfile.ZipFile(file_bytes) as zf:
            if "word/document.xml" not in zf.namelist():
                return []

            xml_content = zf.read("word/document.xml")
            root = ET.fromstring(xml_content)  # nosec

            # DOCX namespaces
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

            # Find all text runs
            for run in root.findall(".//w:r", ns):
                rPr = run.find("w:rPr", ns)
                is_hidden = False

                if rPr is not None:
                    # Check for white color
                    color = rPr.find("w:color", ns)
                    if (
                        color is not None
                        and color.get(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val",
                            "",
                        ).upper()
                        == "FFFFFF"
                    ):
                        is_hidden = True

                    # Check for micro-font (< 2pt = 4 half-points)
                    sz = rPr.find("w:sz", ns)
                    if sz is not None:
                        try:
                            if (
                                int(
                                    sz.get(
                                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val",
                                        "0",
                                    )
                                )
                                < 4
                            ):
                                is_hidden = True
                        except ValueError:
                            pass

                    # Check for vanishing text
                    vanish = rPr.find("w:vanish", ns)
                    if vanish is not None:
                        is_hidden = True

                if is_hidden:
                    t = run.find("w:t", ns)
                    if t is not None and t.text:
                        hidden_texts.append(t.text)

    except Exception as e:
        logger.error("Failed to parse DOCX for hidden text: %s", e)

    return hidden_texts
