"""
tests/core/test_code_evasion.py
-------------------------------
Unit tests for Code Execution Output Fingerprinting and Evasion Detection.
"""

import pytest
from src.core.output_fingerprinter import (
    compute_output_fingerprint,
    compute_output_variance,
    generate_output_fingerprints,
)
from src.core.test_evasion_detector import detect_evasion_patterns, analyze_test_evasion


class TestOutputFingerprinter:
    """Test suite for output fingerprinting."""

    def test_compute_output_fingerprint_identical(self):
        """Verify identical outputs produce the same fingerprint."""
        outputs = ["Result A", "Result A"]
        fp1 = compute_output_fingerprint(outputs)
        fp2 = compute_output_fingerprint(outputs)
        assert fp1 == fp2

    def test_compute_output_variance_hardcoded(self):
        """Verify variance is 0.0 for identical outputs."""
        outputs = ["Hardcoded", "Hardcoded", "Hardcoded"]
        var = compute_output_variance(outputs)
        assert var == 0.0

    def test_generate_output_fingerprints_hardcoded(self):
        """Verify is_hardcoded flag is set for identical outputs."""
        metrics = generate_output_fingerprints("A", ["A", "A"])
        assert metrics["is_hardcoded"] is True


class TestTestEvasionDetector:
    """Test suite for test-case evasion detection."""

    def test_detect_evasion_patterns_env_sniffing(self):
        """Verify environment sniffing patterns are detected."""
        code = "import os\nif os.getenv('TEST_ENV'):\n    return 1"
        patterns = detect_evasion_patterns(code)
        assert "os.getenv" in patterns

    def test_detect_evasion_patterns_clean(self):
        """Verify clean code has no evasion patterns."""
        code = "def add(a, b):\n    return a + b"
        patterns = detect_evasion_patterns(code)
        assert len(patterns) == 0

    def test_analyze_test_evasion_suspicious(self):
        """Verify suspicious code is flagged correctly."""
        code = "if 'pytest' in sys.argv:\n    return 1"
        metrics = {"is_hardcoded": True, "output_variance": 0.0}
        result = analyze_test_evasion(code, metrics)
        assert result["is_suspicious"] is True
        assert result["evasion_risk_score"] > 0.5
