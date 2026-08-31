"""
src/core/adversarial_analysis.py
--------------------------------
Integration layer for adversarial obfuscation detection.

Hooks the obfuscation detector into the main document parsing pipeline,
allowing the system to flag, log, and optionally reject documents that
exhibit signs of adversarial manipulation before they consume expensive
embedding and similarity computation resources.
"""

import logging
import hashlib
from typing import Optional, Dict, Any

from src.security.obfuscation_detector import (
    analyze_text_obfuscation,
    ObfuscationReport,
)
from src.db.obfuscation_logs_db import (
    log_obfuscation_attempt,
    initialize_obfuscation_db,
)

logger = logging.getLogger(__name__)

# Ensure the DB is initialized when this module is imported
initialize_obfuscation_db()


def analyze_document_for_obfuscation(
    text: str,
    document_id: str,
    user_id: Optional[str] = None,
    threshold: float = 0.15,
    strict_mode: bool = False,
) -> dict[str, Any]:
    """Analyze a document for adversarial obfuscation and log the results.

    This function acts as the primary entry point for the main pipeline.
    It computes the document hash, runs the obfuscation analysis, logs
    the attempt to the database if suspicious, and returns a structured
    dictionary for the pipeline to act upon.

    Args:
        text: The extracted text content of the document.
        document_id: Unique identifier or filename of the document.
        user_id: Optional ID of the user who uploaded the document.
        threshold: Obfuscation score threshold for flagging.
        strict_mode: If True, raises a ValueError when obfuscation is detected,
                     halting the pipeline immediately.

    Returns:
        A dictionary containing:
        - 'is_suspicious' (bool)
        - 'obfuscation_score' (float)
        - 'report' (dict representation of ObfuscationReport)
        - 'document_hash' (str)

    Raises:
        ValueError: If strict_mode is True and the document is suspicious.
    """
    # Compute SHA-256 hash of the raw text for logging and deduplication
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Run the core analysis
    report = analyze_text_obfuscation(text, threshold=threshold)

    result = {
        "is_suspicious": report.is_suspicious,
        "obfuscation_score": report.obfuscation_score,
        "report": report.to_dict(),
        "document_hash": doc_hash,
    }

    # Log to database if suspicious
    if report.is_suspicious:
        log_obfuscation_attempt(
            document_id=document_id,
            document_hash=doc_hash,
            user_id=user_id or "anonymous",
            score=report.obfuscation_score,
            zero_width_count=report.zero_width_count,
            homoglyph_count=report.homoglyph_count,
            flagged_indices_count=len(report.flagged_indices),
        )

        if strict_mode:
            raise ValueError(
                f"Document '{document_id}' rejected in strict mode due to "
                f"adversarial obfuscation (score: {report.obfuscation_score:.4f})."
            )

    return result


def strip_obfuscation_chars(text: str) -> str:
    """Utility to strip detected obfuscation characters from the text.

    This can be used to "clean" a document before passing it to the
    embedding model, ensuring that zero-width spaces and homoglyphs
    do not corrupt the semantic vector representation.

    Args:
        text: The raw text containing obfuscation characters.

    Returns:
        The cleaned text with zero-width and homoglyph characters removed
        or replaced with their Latin equivalents.
    """
    from src.security.obfuscation_detector import (
        ZERO_WIDTH_PATTERN,
        CYRILLIC_HOMOGLYPHS,
    )

    # Replace Cyrillic homoglyphs with their Latin equivalents
    cleaned = text
    for cyrillic, latin in CYRILLIC_HOMOGLYPHS.items():
        cleaned = cleaned.replace(cyrillic, latin)

    # Remove all zero-width and invisible control characters
    cleaned = ZERO_WIDTH_PATTERN.sub("", cleaned)

    return cleaned
import hashlib
from src.security.obfuscation_detector import ObfuscationDetector
# from src.db.obfuscation_logs_db import log_obfuscation_incident

class AdversarialAnalysisPipeline:
    def __init__(self):
        self.detector = ObfuscationDetector()

    def process_document_text(self, raw_text: str, document_id: str) -> dict:
        """
        Intercepts and evaluates raw text before sending it to similarity engines.
        Quarantines documents that cross the safety threshold.
        """
        # Calculate consistent SHA-256 fingerprint hash for auditing records
        doc_hash = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()
        
        # Execute security inspection check
        metrics = self.detector.analyze_text(raw_text)
        
        if metrics["is_flagged"]:
            # Construct log profile structure
            log_payload = {
                "document_id": document_id,
                "document_hash": doc_hash,
                "obfuscation_score": metrics["obfuscation_score"],
                "patterns_found": {
                    "invisible_count": len(metrics["invisible_indices"]),
                    "homoglyph_count": len(metrics["homoglyph_indices"])
                }
            }
            # Commit incident to the database
            # log_obfuscation_incident(log_payload)
            print(f"🚨 [Security Alert]: Adversarial pattern discovered on file {document_id}. Hash: {doc_hash}")

        return {
            "allow_pipeline_execution": not metrics["is_flagged"],
            "security_metrics": metrics,
            "document_hash": doc_hash
        }
