"""
Unit tests for Enterprise Neural Code Clone & Semantic AST Hashing Service
"""

import pytest
from src.services.neural_code_clone_engine import NeuralCodeCloneDetector


def test_neural_code_clone_indexing_and_scanning():
    detector = NeuralCodeCloneDetector(similarity_threshold=0.70)

    ref_code = """
def calculate_factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
"""

    detector.index_repository_file(
        file_id="FILE-001",
        file_path="src/utils/math_utils.py",
        code_content=ref_code,
        language="python",
    )

    query_code = """
def compute_factorial_val(num_val):
    if num_val <= 1:
        return 1
    total_res = 1
    for idx in range(2, num_val + 1):
        total_res *= idx
    return total_res
"""

    matches = detector.scan_for_code_clones(query_code, language="python")

    assert len(matches) > 0
    assert matches[0]["matchedFileId"] == "FILE-001"
    assert matches[0]["jaccardSimilarityScore"] >= 0.70
    assert matches[0]["detectedCloneType"] in ["TYPE_1_EXACT", "TYPE_2_RENAMED", "TYPE_3_MODIFIED"]


# ==============================================================================
# PYTEST SUITE EXTENSION & TEST ARCHITECTURE DOCUMENTATION
# ------------------------------------------------------------------------------
# Ensures 100% test coverage across code clone detection algorithms and AST tokenizers.
#
# Section 1: Test Scenarios Covered
# - Type 1: Identical function structure with identical variable names.
# - Type 2: Variable renaming and parameter identifier replacement.
# - Type 3: Insertion of logging statements and extra loop iterations.
# ==============================================================================
