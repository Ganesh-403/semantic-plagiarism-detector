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
tests/security/test_code_sandbox.py
-----------------------------------
Unit tests for the Dynamic Code Execution Sandboxing and Trace Analysis.
"""

import time

import pytest

from src.core.execution_trace_analyzer import (
    compare_behavioral_fingerprints,
    generate_behavioral_hash,
)
from src.security.code_sandbox import execute_code_sandbox


class TestCodeSandbox:
    """Test suite for sandbox execution and resource limits."""

    def test_execute_valid_code(self):
        """Verify valid Python code executes and captures stdout."""
        code = "print('Hello, Sandbox!')"
        result = execute_code_sandbox(code)
        assert result["return_code"] == 0
        assert "Hello, Sandbox!" in result["stdout"]

    def test_execute_syntax_error(self):
        """Verify syntax errors are captured in stderr."""
        code = "def foo(:"
        result = execute_code_sandbox(code)
        assert result["return_code"] != 0
        assert "SyntaxError" in result["stderr"]

    def test_execute_timeout(self):
        """Verify infinite loops are killed by the timeout."""
        code = "while True: pass"
        start = time.time()
        result = execute_code_sandbox(code, timeout=1.0)
        elapsed = time.time() - start

        assert result["return_code"] == -1
        assert "timed out" in result["stderr"].lower()
        assert elapsed < 3.0  # Should not hang

    def test_execute_with_test_cases(self):
        """Verify test cases are executed and results are captured."""
        code = "def solution(x): return x * 2"
        test_cases = [
            {"input": 2, "expected_output": 4},
            {"input": 5, "expected_output": 10},
            {"input": 3, "expected_output": 7},  # This will fail
        ]
        result = execute_code_sandbox(code, test_cases=test_cases)

        assert result["return_code"] == 0
        assert len(result["test_results"]) == 3

        passed = [t for t in result["test_results"] if t["status"] == "passed"]
        failed = [t for t in result["test_results"] if t["status"] == "failed"]

        assert len(passed) == 2
        assert len(failed) == 1


class TestExecutionTraceAnalyzer:
    """Test suite for behavioral hashing and fingerprint comparison."""

    def test_behavioral_hash_determinism(self):
        """Verify identical traces produce identical behavioral hashes."""
        trace = {
            "stdout": "Output 1\n",
            "stderr": "",
            "return_code": 0,
            "test_results": [{"test_id": 0, "status": "passed"}],
        }

        hash1 = generate_behavioral_hash(
            trace["stdout"],
            trace["stderr"],
            trace["test_results"],
            trace["return_code"],
        )
        hash2 = generate_behavioral_hash(
            trace["stdout"],
            trace["stderr"],
            trace["test_results"],
            trace["return_code"],
        )

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_behavioral_hash_whitespace_normalization(self):
        """Verify trailing whitespace in stdout doesn't change the hash."""
        trace1_stdout = "Output 1"
        trace2_stdout = "Output 1\n\n   "

        hash1 = generate_behavioral_hash(trace1_stdout, "", [], 0)
        hash2 = generate_behavioral_hash(trace2_stdout, "", [], 0)

        assert hash1 == hash2

    def test_compare_fingerprints_identical(self):
        """Verify identical traces yield a behavioral similarity of 1.0."""
        trace = {
            "stdout": "Result",
            "stderr": "",
            "return_code": 0,
            "test_results": [{"test_id": 0, "status": "passed"}],
        }

        result = compare_behavioral_fingerprints(trace, trace)
        assert result["behavioral_similarity"] == 1.0
        assert result["is_behavioral_clone"] is True

    def test_compare_fingerprints_different(self):
        """Verify different traces yield a lower behavioral similarity."""
        trace_a = {
            "stdout": "Result A",
            "stderr": "",
            "return_code": 0,
            "test_results": [{"test_id": 0, "status": "passed"}],
        }
        trace_b = {
            "stdout": "Result B",
            "stderr": "",
            "return_code": 0,
            "test_results": [{"test_id": 0, "status": "failed"}],
        }

        result = compare_behavioral_fingerprints(trace_a, trace_b)
        assert result["behavioral_similarity"] < 1.0
        assert result["is_behavioral_clone"] is False
