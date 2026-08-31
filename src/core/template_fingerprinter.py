"""
src/core/template_fingerprinter.py
----------------------------------
Template Fingerprinter for Document Cloning Detection.

Generates structural fingerprints of document templates and style
definitions to detect template plagiarism.
"""

import hashlib
import logging
from typing import List, Dict, Any
from src.core.formatting_entropy_extractor import compute_formatting_entropy

logger = logging.getLogger(__name__)


def generate_template_fingerprint(
    styles: List[str], file_type: str = "docx"
) -> Dict[str, Any]:
    """Generate a structural fingerprint for a document template.

    Combines the sorted list of unique styles with the formatting entropy
    to create a deterministic hash that identifies the underlying template,
    regardless of the textual content.

    Args:
        styles: List of extracted style tags/macros.
        file_type: Type of document ('docx' or 'latex').

    Returns:
        Dictionary containing the template hash and entropy metrics.
    """
    if not styles:
        return {"template_hash": "", "entropy": 0.0, "style_count": 0}

    # Sort styles to ensure deterministic hashing regardless of extraction order
    sorted_styles = sorted(set(styles))

    # Compute entropy
    entropy = compute_formatting_entropy(styles)

    # Generate hash
    canonical_str = "|".join(sorted_styles)
    template_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    return {
        "template_hash": template_hash,
        "entropy": entropy,
        "style_count": len(sorted_styles),
        "file_type": file_type,
    }


def compare_template_fingerprints(
    fp_a: Dict[str, Any], fp_b: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two template fingerprints to detect cloning.

    Args:
        fp_a: Fingerprint from document A.
        fp_b: Fingerprint from document B.

    Returns:
        Dictionary containing match flags and entropy deltas.
    """
    is_exact_match = (
        fp_a.get("template_hash") == fp_b.get("template_hash")
        and fp_a.get("template_hash") != ""
    )

    entropy_delta = abs(fp_a.get("entropy", 0.0) - fp_b.get("entropy", 0.0))

    # If hashes don't match exactly, check if entropy and style counts are highly similar
    # This catches templates that are slightly modified (e.g., added one custom macro)
    is_structural_match = False
    if not is_exact_match:
        style_count_a = fp_a.get("style_count", 0)
        style_count_b = fp_b.get("style_count", 0)

        if style_count_a > 0 and style_count_b > 0:
            count_ratio = min(style_count_a, style_count_b) / max(
                style_count_a, style_count_b
            )
            if count_ratio > 0.85 and entropy_delta < 0.5:
                is_structural_match = True

    return {
        "is_exact_match": is_exact_match,
        "is_structural_match": is_structural_match,
        "entropy_delta": round(entropy_delta, 4),
        "is_template_plagiarism": is_exact_match or is_structural_match,
    }
