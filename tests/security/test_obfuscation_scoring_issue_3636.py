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
tests/security/test_obfuscation_scoring_issue_3636.py
------------------------------------------------------
Regression tests for Issue #3636.

``calculate_obfuscation_score()`` scored only two of the three signals the module
detects. ``control_char_count`` was populated by ``analyze_text_obfuscation()``,
serialised by ``to_dict()`` — and then dropped on the floor, so a document padded
with invisible control characters and nothing else scored ``0.0``.

The density term was also capped at ``0.5``, which put a 20% zero-width injection
rate on the wrong side of every ``> 0.5`` check. Extreme obfuscation and moderate
obfuscation were indistinguishable in exactly the range where the distinction
matters.

Separately, every character in ``ZERO_WIDTH_PATTERN`` is Unicode category Cf, so
``detect_control_chars()`` was counting the whole zero-width set a second time
and the three counts on a report could exceed the document's length.
"""

import pytest

from src.security.obfuscation_detector import (
    CONTROL_CHAR_WEIGHT,
    MAX_DENSITY_SCORE,
    ObfuscationReport,
    analyze_text_obfuscation,
    calculate_obfuscation_score,
    detect_control_chars,
    detect_homoglyphs,
    detect_zero_width_chars,
)

ZWSP = "\u200b"  # ZERO WIDTH SPACE
ZWNJ = "\u200c"  # ZERO WIDTH NON-JOINER
BOM = "\ufeff"  # ZERO WIDTH NO-BREAK SPACE
WORD_JOINER = "\u2060"
SOFT_HYPHEN = "\u00ad"
CYRILLIC_O = "\u043e"  # homoglyph for Latin 'o'

# Cc control characters with no legitimate place in extracted document text.
STX = "\x02"
BEL = "\x07"
ESC = "\x1b"


class TestControlCharactersAreScored:
    """The signal the detector collected and the score ignored."""

    def test_control_chars_alone_produce_a_non_zero_score(self):
        report = ObfuscationReport(total_characters=100, control_char_count=40)

        assert calculate_obfuscation_score(report) > 0.0

    def test_control_char_density_is_weighted(self):
        report = ObfuscationReport(total_characters=100, control_char_count=1)

        expected = round(0.01 * CONTROL_CHAR_WEIGHT, 4)
        assert calculate_obfuscation_score(report) == expected

    def test_more_control_chars_score_higher(self):
        low = ObfuscationReport(total_characters=1000, control_char_count=2)
        high = ObfuscationReport(total_characters=1000, control_char_count=20)

        assert calculate_obfuscation_score(high) > calculate_obfuscation_score(low)

    def test_saturated_control_char_document_scores_high(self):
        report = ObfuscationReport(total_characters=50, control_char_count=25)

        assert calculate_obfuscation_score(report) > 0.5

    def test_large_absolute_control_char_count_is_penalised(self):
        """A long document hides density; the raw count still has to register."""
        report = ObfuscationReport(total_characters=5000, control_char_count=40)

        assert calculate_obfuscation_score(report) >= 0.2

    def test_control_chars_flag_a_document_end_to_end(self):
        text = "".join(f"word{STX}" for _ in range(20))

        report = analyze_text_obfuscation(text)

        assert report.control_char_count == 20
        assert report.obfuscation_score > 0.0
        assert report.is_suspicious is True

    def test_mixed_control_characters_are_all_counted(self):
        text = f"alpha{STX}beta{BEL}gamma{ESC}delta"

        report = analyze_text_obfuscation(text)

        assert report.control_char_count == 3
        assert report.is_suspicious is True


class TestDensityCap:
    """Saturated documents must be able to clear a high threshold."""

    def test_twenty_percent_zero_width_injection_scores_above_half(self):
        """The exact case reported in the issue."""
        report = ObfuscationReport(total_characters=50, zero_width_count=10)

        assert calculate_obfuscation_score(report) > 0.5

    def test_density_term_is_capped_but_not_at_half(self):
        report = ObfuscationReport(total_characters=10, zero_width_count=10)

        # Density alone saturates; no absolute penalty applies at exactly 10.
        assert calculate_obfuscation_score(report) == MAX_DENSITY_SCORE
        assert MAX_DENSITY_SCORE > 0.5

    def test_cap_still_bounds_the_density_term(self):
        report = ObfuscationReport(total_characters=10, homoglyph_count=10)

        assert calculate_obfuscation_score(report) <= 1.0

    def test_score_increases_with_density_below_the_cap(self):
        scores = [
            calculate_obfuscation_score(
                ObfuscationReport(total_characters=1000, zero_width_count=n)
            )
            for n in (1, 3, 5, 8)
        ]

        assert scores == sorted(scores)
        assert len(set(scores)) == len(scores)


class TestNoDoubleCounting:
    """Zero-width characters are Cf; they must not be counted twice."""

    @pytest.mark.parametrize("char", [ZWSP, ZWNJ, BOM, SOFT_HYPHEN, WORD_JOINER])
    def test_zero_width_chars_are_not_also_control_chars(self, char):
        count, indices = detect_control_chars(f"a{char}b")

        assert count == 0
        assert indices == []

    def test_zero_width_chars_are_still_detected_as_zero_width(self):
        count, _ = detect_zero_width_chars(f"a{ZWSP}b{BOM}c")

        assert count == 2

    def test_counts_are_disjoint(self):
        text = f"word{ZWSP}word{STX}word{CYRILLIC_O}word"

        report = analyze_text_obfuscation(text)

        assert report.zero_width_count == 1
        assert report.control_char_count == 1
        assert report.homoglyph_count == 1

    def test_counts_never_exceed_the_document_length(self):
        text = f"{ZWSP}{ZWNJ}{BOM}{STX}{BEL}"

        report = analyze_text_obfuscation(text)
        total = (
            report.zero_width_count + report.homoglyph_count + report.control_char_count
        )

        assert total <= report.total_characters

    def test_flagged_indices_cover_every_signal(self):
        text = f"a{ZWSP}b{STX}c{CYRILLIC_O}d"

        report = analyze_text_obfuscation(text)

        assert report.flagged_indices == [1, 3, 5]


class TestExistingBehaviourPreserved:
    """The parts that were already right must stay right."""

    def test_clean_text_scores_zero(self):
        report = ObfuscationReport(total_characters=100)

        assert calculate_obfuscation_score(report) == 0.0

    def test_empty_text_scores_zero(self):
        assert calculate_obfuscation_score(ObfuscationReport(total_characters=0)) == 0.0

    def test_clean_prose_is_not_flagged(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5

        report = analyze_text_obfuscation(text)

        assert report.obfuscation_score == 0.0
        assert report.is_suspicious is False

    def test_ordinary_whitespace_is_not_a_control_character(self):
        count, _ = detect_control_chars("line one\nline two\ttabbed\r\n")

        assert count == 0

    def test_absolute_zero_width_penalty_still_applies(self):
        report = ObfuscationReport(total_characters=1000, zero_width_count=15)

        assert calculate_obfuscation_score(report) >= 0.3

    def test_absolute_homoglyph_penalty_still_applies(self):
        report = ObfuscationReport(total_characters=1000, homoglyph_count=6)

        assert calculate_obfuscation_score(report) >= 0.2

    def test_homoglyph_detection_is_unchanged(self):
        count, indices = detect_homoglyphs(f"hell{CYRILLIC_O} w{CYRILLIC_O}rld")

        assert count == 2
        assert indices == [4, 7]

    def test_empty_text_returns_an_empty_report(self):
        report = analyze_text_obfuscation("")

        assert report.total_characters == 0
        assert report.is_suspicious is False


class TestScoreBounds:
    """The score is documented as living in [0.0, 1.0]."""

    @pytest.mark.parametrize(
        "zw, hg, ctrl, total",
        [
            (0, 0, 0, 1),
            (1, 1, 1, 3),
            (100, 100, 100, 300),
            (500, 0, 0, 500),
            (0, 500, 0, 500),
            (0, 0, 500, 500),
            (50, 50, 50, 10_000),
        ],
    )
    def test_score_stays_in_range(self, zw, hg, ctrl, total):
        report = ObfuscationReport(
            total_characters=total,
            zero_width_count=zw,
            homoglyph_count=hg,
            control_char_count=ctrl,
        )

        score = calculate_obfuscation_score(report)

        assert 0.0 <= score <= 1.0

    def test_fully_obfuscated_document_reaches_the_ceiling(self):
        report = ObfuscationReport(
            total_characters=100,
            zero_width_count=40,
            homoglyph_count=30,
            control_char_count=30,
        )

        assert calculate_obfuscation_score(report) == 1.0

    def test_score_is_reported_to_four_decimals(self):
        report = ObfuscationReport(total_characters=7, control_char_count=1)

        score = calculate_obfuscation_score(report)

        assert score == round(score, 4)


class TestReportSerialisation:
    """to_dict() advertises control_char_count; the score now backs that up."""

    def test_control_char_count_is_serialised(self):
        report = analyze_text_obfuscation(f"a{STX}b{BEL}c")

        assert report.to_dict()["control_char_count"] == 2

    def test_serialised_score_matches_the_computed_score(self):
        report = analyze_text_obfuscation(f"a{STX}b{BEL}c")

        assert report.to_dict()["obfuscation_score"] == report.obfuscation_score
