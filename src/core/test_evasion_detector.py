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
src/core/test_evasion_detector.py
---------------------------------
Test-Case Evasion Detection Engine.

Analyzes execution traces and code structure to detect conditional logic
that sniffs the testing environment or bypasses hidden tests.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Patterns that indicate environment sniffing or test evasion
EVASION_PATTERNS = [
    r"os\.environ",
    r"os\.getenv",
    r"sys\.argv",
    r'__name__\s*==\s*["\']__main__["\']',
    r"pytest",
    r"unittest",
    r"hidden_test",
    r"secret_test",
]


def detect_evasion_patterns(code: str) -> List[str]:
    """Scan code for patterns that indicate test-case evasion.

    Args:
        code: The submitted source code.

    Returns:
        List of matched evasion patterns.
    """
    if not code:
        return []

    matches = []
    for pattern in EVASION_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            matches.append(pattern)

    return matches


def analyze_test_evasion(code: str, output_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze code and execution outputs for test-case evasion.

    Args:
        code: The submitted source code.
        output_metrics: Metrics from output_fingerprinter.

    Returns:
        Dictionary containing evasion flags and confidence scores.
    """
    evasion_patterns = detect_evasion_patterns(code)
    is_hardcoded = output_metrics.get("is_hardcoded", False)
    output_variance = output_metrics.get("output_variance", 1.0)

    # Compute evasion risk score
    # Hardcoded outputs and environment sniffing are strong indicators
    risk_score = 0.0
    if is_hardcoded:
        risk_score += 0.6
    if len(evasion_patterns) > 0:
        risk_score += 0.4

    risk_score = min(1.0, risk_score)

    return {
        "evasion_patterns": evasion_patterns,
        "is_hardcoded": is_hardcoded,
        "output_variance": output_variance,
        "evasion_risk_score": round(risk_score, 4),
        "is_suspicious": risk_score > 0.5,
    }
