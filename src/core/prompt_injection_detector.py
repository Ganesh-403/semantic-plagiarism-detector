"""
src/core/prompt_injection_detector.py
-------------------------------------
Prompt Injection and Cheat Sheet Detector.

Analyzes extracted hidden payloads for AI prompt injection patterns
(e.g., "Ignore previous instructions") and cheat-sheet heuristics.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Common AI prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior\s+prompts",
    r"forget\s+(everything|all)\s+(above|before)",
    r"act\s+as\s+an?\s+(expert|AI|assistant)",
    r"you\s+are\s+now\s+an?\s+",
    r"generate\s+an?\s+essay\s+about",
    r"write\s+a\s+paper\s+on",
]


def detect_prompt_injections(payloads: List[str]) -> Dict[str, Any]:
    """Analyze hidden payloads for AI prompt injection patterns.

    Args:
        payloads: List of extracted hidden text payloads.

    Returns:
        Dictionary containing injection flags and matched patterns.
    """
    if not payloads:
        return {"is_injection": False, "matched_patterns": [], "risk_score": 0.0}

    combined_payload = " ".join(payloads).lower()
    matched_patterns = []

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, combined_payload, re.IGNORECASE):
            matched_patterns.append(pattern)

    is_injection = len(matched_patterns) > 0
    risk_score = min(1.0, len(matched_patterns) * 0.3)

    return {
        "is_injection": is_injection,
        "matched_patterns": matched_patterns,
        "risk_score": round(risk_score, 4),
    }


def analyze_steganography(
    text: str, file_bytes: bytes = None, file_type: str = "txt"
) -> Dict[str, Any]:
    """Analyze document for steganography and prompt injections.

    Args:
        text: The extracted plain text.
        file_bytes: Raw file bytes (for DOCX parsing).
        file_type: File extension.

    Returns:
        Dictionary containing steganography metrics and injection flags.
    """
    from src.core.steganography_extractor import (
        extract_zero_width_payloads,
        extract_hidden_docx_text,
    )

    all_payloads = []

    # Extract zero-width payloads from plain text
    zw_payloads = extract_zero_width_payloads(text)
    all_payloads.extend(zw_payloads)

    # Extract hidden DOCX text if applicable
    if file_type.lower() in ["docx", "doc"] and file_bytes:
        docx_payloads = extract_hidden_docx_text(file_bytes)
        all_payloads.extend(docx_payloads)

    from src.core.prompt_injection_detector import detect_prompt_injections

    injection_result = detect_prompt_injections(all_payloads)

    return {
        "zero_width_payloads": len(zw_payloads),
        "hidden_docx_payloads": len(all_payloads) - len(zw_payloads),
        "is_injection": injection_result["is_injection"],
        "risk_score": injection_result["risk_score"],
        "matched_patterns": injection_result["matched_patterns"],
    }
