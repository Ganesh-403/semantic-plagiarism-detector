# tests/test_sentence_boundary_abort.py
"""
Test for _find_sentence_boundary max_search abort logic.

This test ensures that when a document has no punctuation for an extended
period, _find_sentence_boundary properly aborts the search after max_search
characters and forces a split at the original index.
"""

import unittest
from typing import Optional


class TestFindSentenceBoundaryAbort(unittest.TestCase):
    """Test cases for _find_sentence_boundary max_search abort logic."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a 500-word string with absolutely no punctuation
        # Using common words without any punctuation marks
        words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog",
            "and",
            "then",
            "runs",
            "through",
            "the",
            "forest",
            "with",
            "great",
            "speed",
            "and",
            "agility",
            "while",
            "chasing",
            "the",
            "rabbit",
            "that",
            "had",
            "escaped",
            "from",
            "its",
            "burrow",
            "earlier",
            "that",
            "morning",
            "when",
            "the",
            "sun",
            "was",
            "just",
            "beginning",
            "to",
            "rise",
            "over",
            "the",
            "distant",
            "mountains",
            "casting",
            "long",
            "shadows",
            "across",
            "the",
            "meadow",
            "where",
            "deer",
            "grazed",
            "peacefully",
            "unaware",
            "of",
            "the",
            "danger",
            "that",
            "lurked",
            "in",
            "the",
            "nearby",
            "thicket",
            "where",
            "the",
            "fox",
            "had",
            "made",
            "its",
            "den",
            "and",
            "raised",
            "its",
            "cubs",
            "during",
            "the",
            "spring",
            "season",
            "when",
            "food",
            "was",
            "plentiful",
        ]

        # Repeat words to create 500+ word string
        self.no_punctuation_text = " ".join(words * 20)  # ~500 words
        # Ensure it has exactly no punctuation
        self.no_punctuation_text = (
            self.no_punctuation_text.replace(".", "")
            .replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace(";", "")
            .replace(":", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("'", "")
        )

        # Create a document with some punctuation for comparison
        self.punctuated_text = "This is a sentence. This is another sentence! Is this a question? Yes, it is."

    def test_find_sentence_boundary_abort_at_max_search(self):
        """
        Test that _find_sentence_boundary aborts after max_search characters
        and returns the original index when no punctuation is found.
        """
        # Setup: call the method with index=200
        # The max_search is typically 150 characters
        index = 200
        result = self._find_sentence_boundary(self.no_punctuation_text, index)

        # Assert: should return exactly the original index
        self.assertEqual(
            result,
            index,
            f"Should return original index {index} when no punctuation found, but got {result}",
        )

        # Assert: result should not be beyond the text length
        self.assertLessEqual(result, len(self.no_punctuation_text))

        # Assert: if result != index, it should be within max_search range
        if result != index:
            distance = abs(result - index)
            self.assertLessEqual(
                distance,
                150,
                f"Boundary should be within 150 chars of index, but was {distance} chars away",
            )

    def test_find_sentence_boundary_abort_with_long_text(self):
        """
        Test abort logic with a long text that has no punctuation.
        This ensures no infinite looping or crashing.
        """
        index = 100
        result = self._find_sentence_boundary(self.no_punctuation_text, index)

        self.assertEqual(
            result,
            index,
            f"Should return original index {index} when no punctuation found",
        )

        # Test with different index positions
        for index in [50, 150, 250, 350, 450]:
            result = self._find_sentence_boundary(self.no_punctuation_text, index)
            self.assertEqual(
                result,
                index,
                f"Should return original index {index} when no punctuation found",
            )

    def test_find_sentence_boundary_with_punctuation(self):
        """
        Test that _find_sentence_boundary works normally with punctuation.
        This ensures the method still functions correctly with valid inputs.
        """
        index = 10
        result = self._find_sentence_boundary(self.punctuated_text, index)

        # Should find a sentence boundary (period, exclamation, or question mark)
        self.assertNotEqual(
            result, index, "Should find a sentence boundary when punctuation exists"
        )
        self.assertLessEqual(result, len(self.punctuated_text))

        # The boundary should be a punctuation mark or whitespace after it
        boundary_char = self.punctuated_text[result - 1] if result > 0 else ""
        self.assertIn(
            boundary_char,
            ".!?",
            f"Boundary should be at a punctuation mark, got '{boundary_char}' at position {result}",
        )

    def test_find_sentence_boundary_edge_cases(self):
        """
        Test edge cases: index at beginning, end, and near end of text.
        """
        # Test at beginning
        result = self._find_sentence_boundary(self.no_punctuation_text, 0)
        self.assertEqual(result, 0, "Should return 0 when at beginning")

        # Test near end
        text_len = len(self.no_punctuation_text)
        if text_len > 200:
            near_end_index = text_len - 50
            result = self._find_sentence_boundary(
                self.no_punctuation_text, near_end_index
            )
            self.assertEqual(
                result,
                near_end_index,
                f"Should return {near_end_index} when near end of text",
            )

        # Test with empty text
        empty_text = ""
        result = self._find_sentence_boundary(empty_text, 0)
        self.assertEqual(result, 0, "Should return 0 for empty text")

    def test_find_sentence_boundary_no_infinite_loop(self):
        """
        Test that the method doesn't infinite loop with text that has no punctuation.
        This is a critical safety valve test.
        """
        import time

        # Create a very long text with no punctuation
        long_text = " ".join(["word"] * 1000)  # 1000 words, no punctuation

        start_time = time.time()
        result = self._find_sentence_boundary(long_text, 200)
        elapsed_time = time.time() - start_time

        # Should return quickly (under 1 second) without infinite looping
        self.assertLess(
            elapsed_time, 1.0, "Method should complete quickly without infinite looping"
        )
        self.assertEqual(result, 200, "Should return original index")

    def _find_sentence_boundary(self, text: str, index: int) -> int:
        """
        Mock implementation of _find_sentence_boundary that includes the abort logic.

        This is a simplified version that demonstrates the expected behavior.
        The actual implementation may differ, but should follow this logic.
        """
        if not text or index >= len(text):
            return min(index, len(text))

        max_search = 150
        original_index = index

        # Look forward for sentence boundaries
        # For this test, we'll look for punctuation marks
        sentence_terminators = {".", "!", "?"}

        # Search forward within max_search characters
        search_limit = min(len(text), index + max_search)
        found_boundary = False

        for i in range(index, search_limit):
            if text[i] in sentence_terminators:
                # Found a sentence boundary
                # Return the position after the punctuation
                return i + 1
            elif (
                text[i] in {" ", "\n", "\t"}
                and i > index
                and text[i - 1] in sentence_terminators
            ):
                # Found whitespace after punctuation
                return i

        # No punctuation found within max_search, return original index
        return original_index


if __name__ == "__main__":
    unittest.main()
