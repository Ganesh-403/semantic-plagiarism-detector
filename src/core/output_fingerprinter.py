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
src/core/output_fingerprinter.py
--------------------------------
Code Execution Output Fingerprinting Engine.

Generates execution output fingerprints across mutated and fuzzed inputs
to detect hardcoded logic and test-case evasion. If a student's code produces
identical outputs for vastly different inputs, it indicates hardcoded logic.
"""

import hashlib
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def compute_output_fingerprint(outputs: List[str]) -> str:
    """Compute a structural fingerprint for a list of execution outputs.

    Args:
        outputs: List of output strings from mutated test inputs.

    Returns:
        SHA-256 hash representing the output pattern.
    """
    if not outputs:
        return ""

    # Normalize outputs by stripping whitespace
    normalized = [out.strip() for out in outputs]
    canonical_str = "|".join(normalized)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def compute_output_variance(outputs: List[str]) -> float:
    """Compute the variance in output lengths across mutated inputs.

    If the variance is near 0, it suggests the code is ignoring the input
    and returning a hardcoded string.

    Args:
        outputs: List of output strings.

    Returns:
        Normalized variance score between 0.0 and 1.0.
    """
    if len(outputs) < 2:
        return 1.0  # Cannot compute variance for < 2 outputs

    lengths = [len(out) for out in outputs]
    mean_len = sum(lengths) / len(lengths)

    if mean_len == 0:
        return 0.0

    variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)

    # Normalize variance (heuristic: variance > mean_len^2 is high variance)
    normalized_var = min(1.0, variance / (mean_len**2 + 1e-6))
    return round(normalized_var, 4)


def generate_output_fingerprints(
    original_output: str, mutated_outputs: List[str]
) -> Dict[str, Any]:
    """Generate fingerprints and variance metrics for execution outputs.

    Args:
        original_output: Output from the original test case.
        mutated_outputs: Outputs from fuzzed/mutated inputs.

    Returns:
        Dictionary containing fingerprints and variance metrics.
    """
    all_outputs = [original_output] + mutated_outputs

    fingerprint = compute_output_fingerprint(all_outputs)
    variance = compute_output_variance(all_outputs)

    # Check if all outputs are identical (strong indicator of hardcoding)
    unique_outputs = set(out.strip() for out in all_outputs)
    is_hardcoded = len(unique_outputs) == 1 and len(all_outputs) > 1

    return {
        "fingerprint": fingerprint,
        "output_variance": variance,
        "unique_outputs": len(unique_outputs),
        "total_outputs": len(all_outputs),
        "is_hardcoded": is_hardcoded,
    }
