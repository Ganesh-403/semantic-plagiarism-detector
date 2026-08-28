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
tests/utils/test_text_stats_sentence_boundaries.py
--------------------------------------------------
Regression tests for sentence-boundary detection in ``count_sentences()``.

The abbreviation guard used to run ``str.replace()`` over the lowercased text,
which matches anywhere rather than at a word boundary. "co." therefore matched
inside "disco.", "no." inside "casino." and "art." inside "chart.", stripping a
real sentence-ending period every time. The count came out low, and because
``get_readability_metrics()`` divides words by sentences, both the Flesch
Reading Ease and the Flesch-Kincaid Grade Level moved with it.

These tests pin the boundary rules: abbreviations only count as whole words,
dotted acronyms and decimal points never end a sentence, and ordinary words
that happen to contain an abbreviation as a substring are left alone.
"""

import pytest

from src.utils.text_stats import (
    compute_text_stats,
    count_sentences,
    get_readability_metrics,
    get_sentence_count,
    get_text_stats,
)

# Words that contain a listed abbreviation as a substring. Each pair is two
# plain sentences, so the expected count is always 2.
WORDS_ENDING_IN_AN_ABBREVIATION = [
    ("I went to the disco. It was fun.", "co"),
    ("She lost it all at the casino. Then she walked home.", "no"),
    ("He studied the chart. It made sense.", "art"),
    ("They toured the villa. Then they left.", "la"),
    ("Look at the flamingo. It is pink.", "go"),
    ("He plays the piano. She sings.", "no"),
    ("We ate the taco. It was good.", "co"),
    ("The tempo. That was the problem.", "po"),
]


class TestAbbreviationsMatchWholeWordsOnly:
    """A period only survives masking when the abbreviation is a full word."""

    @pytest.mark.parametrize(
        "text,substring",
        WORDS_ENDING_IN_AN_ABBREVIATION,
        ids=[pair[1] for pair in WORDS_ENDING_IN_AN_ABBREVIATION],
    )
    def test_word_containing_abbreviation_still_ends_a_sentence(self, text, substring):
        """The word ends a sentence; it is not the abbreviation inside it."""
        assert count_sentences(text) == 2

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Dr. Smith arrived. He stayed.", 2),
            ("Mr. and Mrs. Smith arrived. They left.", 2),
            ("See fig. 3 for details. It is clear.", 2),
            ("Prof. Lee vs. Dr. Kim debated. It ended.", 2),
            ("Acme Inc. filed the report. It was late.", 2),
            ("Refer to vol. 2, pp. 14-18. The rest is appendix.", 2),
        ],
    )
    def test_genuine_abbreviations_are_still_ignored(self, text, expected):
        """The original behaviour for real abbreviations is unchanged."""
        assert count_sentences(text) == expected

    def test_abbreviation_matching_is_case_insensitive(self):
        """ "DR." and "dr." are both abbreviations."""
        assert count_sentences("DR. Smith arrived. He stayed.") == 2
        assert count_sentences("dr. smith arrived. he stayed.") == 2

    def test_abbreviation_at_end_of_text_is_not_a_sentence(self):
        """Text that trails off in an abbreviation still floors at one."""
        assert count_sentences("Published by Acme Inc.") == 1


class TestDottedAcronyms:
    """Acronyms written with interior periods are a single token."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The U.S. economy grew. Inflation fell.", 2),
            ("The E.U. agreed. Members signed.", 2),
            ("He joined N.A.V.O. last year. Then he left.", 2),
            ("The U.K. and the U.S. disagreed.", 1),
        ],
    )
    def test_acronym_periods_do_not_end_sentences(self, text, expected):
        assert count_sentences(text) == expected

    def test_acronym_rule_is_generic(self):
        """An acronym absent from the abbreviation list is still handled."""
        assert count_sentences("She works at the C.I.A. now. He does not.") == 2


class TestDecimalNumbers:
    """A period between digits is a decimal point."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Pi is 3.14 exactly", 1),
            ("Version 2.5 shipped. Users cheered.", 2),
            ("The score was 9.75 out of 10.0 overall", 1),
            ("Revenue rose 12.5 percent. Costs fell 3.2 percent.", 2),
        ],
    )
    def test_decimal_points_do_not_end_sentences(self, text, expected):
        assert count_sentences(text) == expected

    def test_trailing_period_after_a_number_still_counts(self):
        """A period straight after a number can still end a clause."""
        assert count_sentences("The total was 10. We went home.") == 2


class TestExistingBehaviourIsPreserved:
    """Guard rails carried over from the previous implementation."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", 0),
            ("   ", 0),
            ("\n\t ", 0),
            ("Hello.", 1),
            ("Hello. World.", 2),
            ("Hello! How are you? I'm fine.", 3),
            ("No punctuation", 1),
            ("The cat sat on the mat", 1),
        ],
    )
    def test_baseline_counts(self, text, expected):
        assert count_sentences(text) == expected

    def test_consecutive_marks_count_once(self):
        """A run of marks such as "?!" or "..." is one break, not several."""
        assert count_sentences("Wait... What?! Really.") == 3

    def test_alias_matches_the_primary_name(self):
        """get_sentence_count is documented as an alias."""
        text = "I went to the disco. It was fun."
        assert get_sentence_count(text) == count_sentences(text)


class TestDownstreamMetrics:
    """The sentence count feeds readability, so it must reach those callers."""

    def test_readability_uses_the_corrected_sentence_count(self):
        """Two short sentences should not be scored as one long one."""
        text = "I went to the disco. It was fun."

        ease, grade = get_readability_metrics(text)

        # Words per sentence is 7/2, not 7/1. A single-sentence reading pushes
        # the grade level up by roughly a full grade.
        _, single_sentence_grade = get_readability_metrics(
            "I went to the disco it was fun"
        )
        assert grade < single_sentence_grade

    def test_compute_text_stats_reports_the_corrected_count(self):
        stats = compute_text_stats("I went to the disco. It was fun.")
        assert stats["sentence_count"] == 2

    def test_get_text_stats_reports_the_corrected_count(self):
        stats = get_text_stats("She lost it all at the casino. Then she left.")
        assert stats["sentences"] == 2

    def test_paragraph_with_mixed_hazards(self):
        """One paragraph exercising every masking rule at once."""
        text = (
            "Dr. Alvarez left the disco at 2.30 and drove to the casino. "
            "The U.S. delegation, per vol. 4, arrived later. "
            "Was it worth it? Probably not!"
        )
        # Sentences: the casino one, the delegation one, the question, the
        # exclamation.
        assert count_sentences(text) == 4
