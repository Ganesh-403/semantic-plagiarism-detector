"""
src/core/execution_trace_analyzer.py
------------------------------------
Execution Trace Analyzer for Code Plagiarism Detection.

Captures standard I/O, exception traces, and generates behavioral hashes
from sandboxed code execution. Compares the behavioral fingerprint against
the source corpus to detect algorithmic plagiarism even when code is obfuscated.
"""

import hashlib
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def generate_behavioral_hash(
    stdout: str, stderr: str, test_results: list[dict[str, Any]], return_code: int
) -> str:
    """Generate a deterministic behavioral hash from execution traces.

    This hash represents the "behavioral fingerprint" of the code. Two programs
    that produce the exact same outputs and errors for the same inputs will
    have the same behavioral hash, even if their source code is completely different.

    Args:
        stdout: Standard output captured from the sandbox.
        stderr: Standard error captured from the sandbox.
        test_results: List of test case execution results.
        return_code: The process return code.

    Returns:
        A SHA-256 hex digest string representing the behavioral fingerprint.
    """
    # Normalize outputs to ensure determinism
    # Strip trailing whitespace/newlines which might vary by OS
    norm_stdout = stdout.strip()
    norm_stderr = stderr.strip()

    # Extract only the essential test outcomes (pass/fail/error) to ignore
    # minor differences in exception message formatting across Python versions
    normalized_tests = []
    for tr in test_results:
        normalized_tests.append(
            {
                "test_id": tr.get("test_id"),
                "status": tr.get("status"),  # 'passed', 'failed', 'error'
            }
        )

    # Sort test results by test_id to ensure order doesn't affect the hash
    normalized_tests.sort(key=lambda x: x.get("test_id", 0))

    # Build the fingerprint payload
    payload = {
        "stdout": norm_stdout,
        "stderr": norm_stderr,
        "return_code": return_code,
        "test_outcomes": normalized_tests,
    }

    # Serialize to canonical JSON string
    payload_str = json.dumps(payload, sort_keys=True)

    # Hash the payload
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def compare_behavioral_fingerprints(
    trace_a: dict[str, Any], trace_b: dict[str, Any]
) -> dict[str, Any]:
    """Compare two execution traces and compute behavioral similarity.

    Args:
        trace_a: Execution trace dictionary from code_sandbox.
        trace_b: Execution trace dictionary from code_sandbox.

    Returns:
        Dictionary containing the behavioral hashes and a similarity score.
    """
    hash_a = generate_behavioral_hash(
        trace_a.get("stdout", ""),
        trace_a.get("stderr", ""),
        trace_a.get("test_results", []),
        trace_a.get("return_code", -1),
    )

    hash_b = generate_behavioral_hash(
        trace_b.get("stdout", ""),
        trace_b.get("stderr", ""),
        trace_b.get("test_results", []),
        trace_b.get("return_code", -1),
    )

    # If the hashes match exactly, the behavioral similarity is 1.0
    # Otherwise, we compute a partial similarity based on test outcome overlap
    if hash_a == hash_b:
        similarity = 1.0
    else:
        # Compute Jaccard similarity of test outcomes
        tests_a = {
            tr["test_id"]: tr["status"] for tr in trace_a.get("test_results", [])
        }
        tests_b = {
            tr["test_id"]: tr["status"] for tr in trace_b.get("test_results", [])
        }

        all_ids = set(tests_a.keys()).union(set(tests_b.keys()))
        if not all_ids:
            similarity = 0.0
        else:
            matches = sum(1 for tid in all_ids if tests_a.get(tid) == tests_b.get(tid))
            similarity = matches / len(all_ids)

    return {
        "hash_a": hash_a,
        "hash_b": hash_b,
        "behavioral_similarity": round(similarity, 4),
        "is_behavioral_clone": hash_a == hash_b,
    }
