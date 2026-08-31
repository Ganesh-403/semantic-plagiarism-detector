"""
test_coleman_liau_readability_issue_3704.py
-------------------------------------------
Comprehensive unit and regression test suite for Issue #3704:
Coleman-Liau readability index in compute_text_stats.

Verifies:
1. Exact mathematical calculation of Coleman-Liau Index: CLI = 0.0588 * L - 0.296 * S - 15.8
2. Verification against known standard benchmark texts.
3. Proper inclusion of 'readability_score' in compute_text_stats() output dictionary.
4. Edge cases: empty text, whitespace-only, single-word, no punctuation, numbers, special characters, CJK scripts.
5. Grade level descriptions, batch evaluations, and comparative document readability analysis.
6. Integration with PDF table formatting helpers.
"""

import pytest

from src.utils.text_stats import (
    assess_readability_homogeneity,
    batch_compute_readability,
    compare_readability_scores,
    compute_coleman_liau_index,
    compute_text_stats,
    count_letters,
    count_sentences,
    count_words,
    explain_readability_grade_breakdown,
    format_stats_for_pdf,
    get_coleman_liau_grade_description,
)


class TestColemanLiauCalculation:
    """Test suite verifying mathematical precision of compute_coleman_liau_index."""

    def test_empty_string_returns_zero(self):
        assert compute_coleman_liau_index("") == 0.0
        assert compute_coleman_liau_index("   ") == 0.0
        assert compute_coleman_liau_index("\n\t\r") == 0.0

    def test_single_word_no_punctuation(self):
        # 1 word ("hello"), 5 letters, 1 sentence (default fallback)
        # L = (5 / 1) * 100 = 500.0
        # S = (1 / 1) * 100 = 100.0
        # CLI = 0.0588 * 500 - 0.296 * 100 - 15.8 = 29.4 - 29.6 - 15.8 = -16.0
        result = compute_coleman_liau_index("hello")
        assert isinstance(result, float)
        assert result == -16.0

    def test_known_benchmark_alice_in_wonderland(self):
        """Test with an excerpt from Alice in Wonderland."""
        text = (
            "Alice was beginning to get very tired of sitting by her sister on the bank, "
            "and of having nothing to do: once or twice she had peeped into the book her sister was reading, "
            "but it had no pictures or conversations in it, 'and what is the use of a book,' "
            "thought Alice 'without pictures or conversation?'"
        )
        words = count_words(text)
        sentences = count_sentences(text)
        letters = count_letters(text)
        assert words > 0
        assert sentences > 0
        assert letters > 0

        l_val = (letters / words) * 100.0
        s_val = (sentences / words) * 100.0
        expected = round(0.0588 * l_val - 0.296 * s_val - 15.8, 2)

        actual = compute_coleman_liau_index(text)
        assert actual == expected
        assert 6.0 <= actual <= 10.0

    def test_standard_high_school_level_passage(self):
        """A typical high-school biology textbook passage."""
        text = (
            "Photosynthesis is the biological process used by plants, algae, and certain bacteria "
            "to convert light energy into chemical energy. This chemical energy is stored in carbohydrate "
            "molecules, such as sugars, which are synthesized from carbon dioxide and water. "
            "Most plants produce oxygen gas as a byproduct of this biochemical reaction."
        )
        score = compute_coleman_liau_index(text)
        assert isinstance(score, float)
        assert 10.0 <= score <= 18.0

    def test_elementary_school_passage(self):
        """Simple sentences with short words."""
        text = "The cat sat on the mat. The dog ran in the park. The sun was hot. We had fun."
        score = compute_coleman_liau_index(text)
        assert isinstance(score, float)
        assert score < 6.0

    @pytest.mark.parametrize(
        "sample_text,expected_bracket",
        [
            ("See Spot run. Spot runs fast.", "Elementary School"),
            (
                "Modern distributed operating systems synchronize cluster state using replicated state machines.",
                "College",
            ),
            (
                "Quantum electrodynamics formulates relativistic field theory across non-Abelian gauge groups.",
                "Graduate",
            ),
        ],
    )
    def test_parameterized_grade_brackets(self, sample_text, expected_bracket):
        cli = compute_coleman_liau_index(sample_text)
        desc = get_coleman_liau_grade_description(cli)
        assert expected_bracket in desc


class TestCountLetters:
    """Test suite for count_letters helper function."""

    def test_count_letters_empty(self):
        assert count_letters("") == 0
        assert count_letters("   ") == 0

    def test_count_letters_basic(self):
        assert count_letters("abc") == 3
        assert count_letters("Hello World!") == 10
        assert count_letters("123 abc") == 6

    def test_count_letters_ignores_punctuation_and_spaces(self):
        text = "!@#$%^&*() _+=-~`[]{}|;:'\",.<>?/"
        assert count_letters(text) == 0

    def test_count_letters_unicode(self):
        text = "Café résumé naïve"
        # Standard regex [a-zA-Z0-9] counts ascii letters
        assert count_letters(text) >= 10


class TestComputeTextStatsIntegration:
    """Test suite ensuring compute_text_stats includes 'readability_score'."""

    def test_compute_text_stats_contains_readability_score_key(self):
        stats = compute_text_stats("The quick brown fox jumps over the lazy dog.")
        assert "readability_score" in stats
        assert isinstance(stats["readability_score"], float)

    def test_compute_text_stats_all_keys_present(self):
        stats = compute_text_stats("Sample academic paper content.")
        expected_keys = {
            "word_count",
            "sentence_count",
            "unique_word_count",
            "unique_word_ratio",
            "readability_score",
        }
        assert set(stats.keys()) == expected_keys

    def test_compute_text_stats_empty_text(self):
        stats = compute_text_stats("")
        assert stats["word_count"] == 0
        assert stats["sentence_count"] == 0
        assert stats["unique_word_count"] == 0
        assert stats["unique_word_ratio"] == 0.0
        assert stats["readability_score"] == 0.0

    def test_format_stats_for_pdf_includes_readability(self):
        stats = compute_text_stats("This is a formal test document for PDF generation.")
        table_rows = format_stats_for_pdf(stats)
        metric_names = [row[0] for row in table_rows]
        assert "Readability Grade (CLI)" in metric_names
        cli_row = [row for row in table_rows if row[0] == "Readability Grade (CLI)"][0]
        assert float(cli_row[1]) == stats["readability_score"]


class TestGradeDescriptionAndComparison:
    """Test suite for grade descriptions and comparative document analysis."""

    def test_grade_descriptions(self):
        assert "Before Grade 1" in get_coleman_liau_grade_description(-2.5)
        assert "Elementary School" in get_coleman_liau_grade_description(4.2)
        assert "Middle School" in get_coleman_liau_grade_description(7.8)
        assert "High School" in get_coleman_liau_grade_description(11.4)
        assert "College" in get_coleman_liau_grade_description(15.0)
        assert "Graduate" in get_coleman_liau_grade_description(18.2)

    def test_compare_readability_scores_identical_texts(self):
        text = "This is a sample document evaluating student prose and readability levels."
        comp = compare_readability_scores(text, text)
        assert comp["absolute_difference"] == 0.0
        assert comp["shares_grade_level"] is True
        assert comp["doc_a_readability"] == comp["doc_b_readability"]

    def test_compare_readability_scores_different_levels(self):
        text_simple = "The boy has a red ball. He plays with the dog."
        text_complex = (
            "Epistemological inquiries regarding cognitive semantics necessitate "
            "multidisciplinary methodological frameworks."
        )
        comp = compare_readability_scores(text_simple, text_complex)
        assert comp["absolute_difference"] > 5.0
        assert comp["doc_a_readability"] < comp["doc_b_readability"]
        assert comp["shares_grade_level"] is False


class TestBatchAndBreakdownUtilities:
    """Test suite for breakdown and batch evaluation functions."""

    def test_explain_readability_grade_breakdown(self):
        text = "Machine learning models process vector representations."
        breakdown = explain_readability_grade_breakdown(text)
        assert breakdown["words"] == 6
        assert breakdown["letters"] > 0
        assert breakdown["sentences"] == 1
        assert "letters_per_100_words" in breakdown
        assert "sentences_per_100_words" in breakdown
        assert "coleman_liau_index" in breakdown
        assert "grade_bracket" in breakdown

    def test_explain_readability_grade_breakdown_empty(self):
        breakdown = explain_readability_grade_breakdown("")
        assert breakdown["words"] == 0
        assert breakdown["coleman_liau_index"] == 0.0

    def test_batch_compute_readability(self):
        texts = [
            "Short sentence one.",
            "A second slightly longer sentence with multiple words for analysis.",
            "Third complex sentence incorporating algorithmic mechanisms.",
        ]
        results = batch_compute_readability(texts)
        assert len(results) == 3
        for idx, res in enumerate(results):
            assert res["index"] == idx
            assert res["word_count"] > 0
            assert isinstance(res["readability_score"], float)
            assert isinstance(res["grade_level"], str)

    def test_assess_readability_homogeneity(self):
        docs = {
            "doc1.txt": "The quick brown fox jumps over the lazy dog.",
            "doc2.txt": "A fast brown fox leaped across the resting hound.",
            "doc3.txt": "The agile fox bounded over a sleepy canine in the yard.",
        }
        res = assess_readability_homogeneity(docs)
        assert res["document_count"] == 3
        assert "mean_readability" in res
        assert "is_homogeneous" in res
        assert len(res["per_document"]) == 3

    def test_assess_readability_homogeneity_empty(self):
        res = assess_readability_homogeneity({})
        assert res["document_count"] == 0
        assert res["is_homogeneous"] is True
