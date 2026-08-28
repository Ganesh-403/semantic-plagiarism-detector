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

import os
import sys
import unittest
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import SimilarityEngine


class TestThresholdBoundary(unittest.TestCase):
    """
    Test threshold boundary conditions for document similarity matching.
    Verifies exact boundary handling (>= vs > logic) when similarity_score
    equals the threshold exactly.
    """

    def setUp(self):
        """Set up test data with known similarity scores."""
        self.engine = SimilarityEngine()

        # Test documents with specific similarity scores
        self.test_documents = [
            {"id": 1, "text": "Document A", "score": 0.8000},
            {"id": 2, "text": "Document B", "score": 0.7500},
            {"id": 3, "text": "Document C", "score": 0.9000},
            {"id": 4, "text": "Document D", "score": 0.8000},
            {"id": 5, "text": "Document E", "score": 0.8500},
        ]

    def test_threshold_exact_match_included(self):
        """
        Test that documents with similarity_score == threshold are INCLUDED
        when the logic uses >= (greater than or equal to).
        """
        threshold = 0.8000

        # Get matches using the engine
        matches = self.engine.get_matches(
            documents=self.test_documents, threshold=threshold
        )

        # Find documents with score == threshold
        exact_matches = [
            doc for doc in self.test_documents if doc["score"] == threshold
        ]

        # Assert exact boundary matches are included
        for doc in exact_matches:
            self.assertIn(
                doc,
                matches,
                f"Document with score {doc['score']} should be included when threshold is {threshold} (>= logic)",
            )

        # Verify all exact matches are present
        expected_ids = [1, 4]  # Documents with score 0.8000
        actual_ids = [doc["id"] for doc in matches if doc["score"] == threshold]

        self.assertEqual(
            sorted(actual_ids),
            sorted(expected_ids),
            f"Documents with score == {threshold} should be included",
        )

    def test_threshold_exact_match_excluded(self):
        """
        Test that documents with similarity_score == threshold are EXCLUDED
        when the logic uses > (greater than only).
        """
        threshold = 0.8000

        # Get matches using the engine
        matches = self.engine.get_matches(
            documents=self.test_documents,
            threshold=threshold,
            strict_greater=True,  # Use > instead of >=
        )

        # Find documents with score == threshold
        exact_matches = [
            doc for doc in self.test_documents if doc["score"] == threshold
        ]

        # Assert exact boundary matches are excluded
        for doc in exact_matches:
            self.assertNotIn(
                doc,
                matches,
                f"Document with score {doc['score']} should be EXCLUDED when threshold is {threshold} (> logic)",
            )

    def test_threshold_above_boundary_included(self):
        """
        Test that documents with similarity_score > threshold are always included.
        This verifies the logic works correctly for values above the boundary.
        """
        threshold = 0.8000

        matches = self.engine.get_matches(
            documents=self.test_documents, threshold=threshold
        )

        above_matches = [doc for doc in self.test_documents if doc["score"] > threshold]

        for doc in above_matches:
            self.assertIn(
                doc,
                matches,
                f"Document with score {doc['score']} > {threshold} should be included",
            )

    def test_threshold_below_boundary_excluded(self):
        """
        Test that documents with similarity_score < threshold are always excluded.
        This verifies the logic works correctly for values below the boundary.
        """
        threshold = 0.8000

        matches = self.engine.get_matches(
            documents=self.test_documents, threshold=threshold
        )

        below_matches = [doc for doc in self.test_documents if doc["score"] < threshold]

        for doc in below_matches:
            self.assertNotIn(
                doc,
                matches,
                f"Document with score {doc['score']} < {threshold} should be excluded",
            )

    def test_multiple_boundary_values(self):
        """
        Test boundary behavior with multiple threshold values.
        Ensures consistency across different boundary points.
        """
        test_cases = [
            {"threshold": 0.5000, "exact_scores": [0.5000]},
            {"threshold": 0.7500, "exact_scores": [0.7500]},
            {"threshold": 0.9000, "exact_scores": [0.9000]},
            {"threshold": 1.0000, "exact_scores": [1.0000]},
        ]

        for case in test_cases:
            threshold = case["threshold"]

            # Documents with exact boundary scores
            docs = [
                {"id": i, "text": f"Doc {i}", "score": score}
                for i, score in enumerate(case["exact_scores"])
            ]

            matches = self.engine.get_matches(documents=docs, threshold=threshold)

            for doc in docs:
                if doc["score"] == threshold:
                    # Check if included based on current logic
                    self.assertIn(
                        doc,
                        matches,
                        f"Document with score {doc['score']} == {threshold} should be included",
                    )

    def test_empty_document_list(self):
        """Test boundary behavior with empty document list."""
        threshold = 0.8000

        matches = self.engine.get_matches(documents=[], threshold=threshold)

        self.assertEqual(
            len(matches), 0, "Empty document list should return no matches"
        )

    def test_all_documents_above_threshold(self):
        """Test when all documents are above the threshold."""
        threshold = 0.5000

        docs = [
            {"id": 1, "text": "Doc 1", "score": 0.8000},
            {"id": 2, "text": "Doc 2", "score": 0.9000},
            {"id": 3, "text": "Doc 3", "score": 0.7500},
        ]

        matches = self.engine.get_matches(documents=docs, threshold=threshold)

        self.assertEqual(
            len(matches), len(docs), "All documents above threshold should be included"
        )

    def test_all_documents_below_threshold(self):
        """Test when all documents are below the threshold."""
        threshold = 0.9000

        docs = [
            {"id": 1, "text": "Doc 1", "score": 0.8000},
            {"id": 2, "text": "Doc 2", "score": 0.7500},
            {"id": 3, "text": "Doc 3", "score": 0.8500},
        ]

        matches = self.engine.get_matches(documents=docs, threshold=threshold)

        self.assertEqual(
            len(matches), 0, "All documents below threshold should be excluded"
        )

    def test_floating_point_precision(self):
        """
        Test boundary behavior with floating point precision issues.
        Ensures that 0.8000 == 0.8 is handled correctly.
        """
        threshold = 0.8

        docs = [
            {"id": 1, "text": "Doc 1", "score": 0.8000},
            {"id": 2, "text": "Doc 2", "score": 0.8000000001},
            {"id": 3, "text": "Doc 3", "score": 0.7999999999},
        ]

        matches = self.engine.get_matches(documents=docs, threshold=threshold)

        # Document with exact 0.8000 should be included
        self.assertIn(
            docs[0],
            matches,
            "Document with score 0.8000 should be included when threshold is 0.8",
        )

        # Document slightly above should be included
        self.assertIn(
            docs[1], matches, "Document with score 0.8000000001 should be included"
        )

        # Document slightly below should be excluded
        self.assertNotIn(
            docs[2], matches, "Document with score 0.7999999999 should be excluded"
        )

    def test_cli_boundary_output(self):
        """
        Test CLI output formatting for boundary cases.
        Ensures CLI properly displays matches at exact threshold.
        """
        # Create test data file
        import json
        import subprocess

        test_data = {"documents": self.test_documents, "threshold": 0.8000}

        with open("/tmp/test_boundary.json", "w") as f:
            json.dump(test_data, f)

        # Run CLI command
        result = subprocess.run(
            [
                sys.executable,
                "src/cli.py",
                "match",
                "--file",
                "/tmp/test_boundary.json",
                "--threshold",
                "0.8000",
            ],
            capture_output=True,
            text=True,
        )

        # Should succeed and show matches
        self.assertEqual(
            result.returncode, 0, "CLI should exit successfully for boundary case"
        )

        # Parse output to verify boundary matches are included
        output_lines = result.stdout.strip().split("\n")

        # This assertion depends on your CLI output format
        # Check that documents with score 0.8000 are in output
        self.assertIn(
            "0.8000", result.stdout, "CLI should display documents at exact threshold"
        )

        # Clean up
        os.remove("/tmp/test_boundary.json")

    def test_lt_vs_lte_boundary(self):
        """
        Test both < and <= boundary conditions.
        Ensures the logic is consistent with documentation.
        """
        threshold = 0.8000

        docs = [
            {"id": 1, "text": "Doc 1", "score": 0.8000},
            {"id": 2, "text": "Doc 2", "score": 0.8001},
            {"id": 3, "text": "Doc 3", "score": 0.7999},
        ]

        # Test with >= logic (inclusive)
        matches_ge = self.engine.get_matches(
            documents=docs, threshold=threshold, inclusive=True
        )

        self.assertIn(docs[0], matches_ge, "Should include exact match with >=")
        self.assertIn(docs[1], matches_ge, "Should include above match with >=")
        self.assertNotIn(docs[2], matches_ge, "Should exclude below match with >=")

        # Test with > logic (exclusive)
        matches_gt = self.engine.get_matches(
            documents=docs, threshold=threshold, inclusive=False
        )

        self.assertNotIn(docs[0], matches_gt, "Should exclude exact match with >")
        self.assertIn(docs[1], matches_gt, "Should include above match with >")
        self.assertNotIn(docs[2], matches_gt, "Should exclude below match with >")


if __name__ == "__main__":
    unittest.main()
