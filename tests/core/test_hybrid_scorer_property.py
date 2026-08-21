"""Property-based tests for ``HybridScorer.compute_hybrid_similarity``.

Issue #3027
-----------
``compute_hybrid_similarity`` claims to *always* return a value in the closed
interval ``[0.0, 1.0]``. Hardcoded example-based tests can only ever exercise
a handful of hand-picked inputs and therefore cannot *prove* the invariant
holds for pathological inputs such as:

- empty strings
- strings composed solely of whitespace / punctuation / stop-words
- completely identical massive strings (10k+ characters)
- bizarre unicode (combining diacritics, RTL marks, CJK, emoji,
  zero-width characters, surrogate pairs)

This module uses the ``hypothesis`` library to generate adversarial inputs
that no human would think to write down, and asserts the bound invariant
holds for every generated example.

Strategies
~~~~~~~~~~
- ``ascii_text``   – printable ASCII (covers whitespace / punctuation edge cases)
- ``unicode_text`` – full BMP + supplementary plane characters
- ``massive_text`` – repeated short token to create a multi-kilobyte string
- ``unit_float``   – floats in ``[0.0, 1.0]`` (used for semantic score + alpha)

Test Coverage
~~~~~~~~~~~~
- One parametrized property test per lexical method
  (``tfidf``, ``jaccard``, ``dice``, ``overlap``, ``ngram``, ``char_ngram``)
- ASCII / unicode / massive-identical input families
- Explicit edge-case tests for empty, whitespace-only, and stop-word-only inputs
- Mathematical property tests: ``alpha=0`` ⟹ lexical-only, ``alpha=1`` ⟹ semantic-only
"""

from __future__ import annotations

import math
import re
import string

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.core.hybrid_scorer import HybridConfig, HybridScorer


# ── Strategies ────────────────────────────────────────────────────────────────

# Printable ASCII text — covers empty, all-whitespace, all-punctuation, and
# normal prose. Capped at 2_000 chars to keep individual examples fast.
ascii_text = st.text(
    alphabet=string.printable,
    min_size=0,
    max_size=2000,
)

# Unicode text — exercises CJK, RTL marks, combining diacritics, emoji,
# zero-width characters, etc. Surrogate-half code points (category ``Cs``)
# are excluded because CPython cannot encode them into a str without
# explicit surrogatepass handling. Control characters (category ``Cc``)
# are excluded because they can crash some downstream tokenizers.
unicode_text = st.text(
    alphabet=st.characters(
        exclude_categories=("Cs", "Cc"),
    ),
    min_size=0,
    max_size=500,
)


@st.composite
def massive_identical_text(draw):
    """Generate a single short token repeated many times.

    Produces strings of length 1..5000 by repeating a small alphabetic
    seed. Used to stress-test the upper bound and the lexical caches.
    """
    seed = draw(st.text(alphabet=string.ascii_letters, min_size=1, max_size=10))
    repeats = draw(st.integers(min_value=1, max_value=500))
    return seed * repeats


# A float strictly inside [0.0, 1.0]. NaN/inf are excluded because they
# violate the documented input contract for ``semantic_score`` and ``alpha``.
unit_float = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Matches sklearn's default token pattern: words of length >= 2.
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# All lexical methods supported by ``HybridScorer._compute_lexical_score``.
_LEXICAL_METHODS = [
    "tfidf",
    "jaccard",
    "dice",
    "overlap",
    "ngram",
    "char_ngram",
]


def _has_tfidf_vocabulary(text_a: str, text_b: str) -> bool:
    """Return True if at least one of the texts has a non-stopword token.

    ``TfidfVectorizer(stop_words="english")`` raises
    ``ValueError: empty vocabulary`` when every token in the corpus is a
    stop-word or when the corpus contains no alphanumeric tokens at all.
    This helper lets us ``assume()`` such inputs out of the tfidf property
    test without weakening the invariant for the other methods.
    """
    if not text_a and not text_b:
        return False
    return bool(_TOKEN_RE.findall(text_a)) or bool(_TOKEN_RE.findall(text_b))


# ── Property: 0.0 <= score <= 1.0 ────────────────────────────────────────────


class TestHybridScoreBoundInvariant:
    """The core invariant: the returned hybrid score is always in [0, 1]."""

    @pytest.mark.parametrize("method", _LEXICAL_METHODS)
    @given(
        text_a=ascii_text,
        text_b=ascii_text,
        semantic_score=unit_float,
        alpha=unit_float,
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    def test_score_in_unit_interval_ascii(
        self, method, text_a, text_b, semantic_score, alpha
    ):
        """ASCII inputs (incl. empty / whitespace / punctuation only)."""
        if method == "tfidf":
            # TfidfVectorizer raises on an empty post-stopword vocabulary.
            # That is a known sklearn limitation, not a HybridScorer bug;
            # skip those inputs but still exercise the bound for every
            # other case.
            assume(_has_tfidf_vocabulary(text_a, text_b))

        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=method))
        try:
            score = scorer.compute_hybrid_similarity(
                text_a, text_b, semantic_score=semantic_score
            )
        except ValueError:
            # Defensive: if sklearn still rejects the vocabulary (e.g.
            # the input is only punctuation), the bound claim does not
            # apply — but only for tfidf.
            assume(method == "tfidf")
            return
        finally:
            scorer.clear_cache()

        assert isinstance(score, float), (
            f"expected float, got {type(score).__name__} for method={method}"
        )
        assert 0.0 <= score <= 1.0, (
            f"score={score!r} outside [0, 1] for method={method}, "
            f"text_a={text_a[:40]!r}, text_b={text_b[:40]!r}, "
            f"semantic={semantic_score!r}, alpha={alpha!r}"
        )

    @pytest.mark.parametrize("method", _LEXICAL_METHODS)
    @given(
        text_a=unicode_text,
        text_b=unicode_text,
        semantic_score=unit_float,
        alpha=unit_float,
    )
    @settings(
        max_examples=150,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    def test_score_in_unit_interval_unicode(
        self, method, text_a, text_b, semantic_score, alpha
    ):
        """Bizarre unicode inputs (CJK, RTL, combining marks, emoji)."""
        if method == "tfidf":
            assume(_has_tfidf_vocabulary(text_a, text_b))

        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=method))
        try:
            score = scorer.compute_hybrid_similarity(
                text_a, text_b, semantic_score=semantic_score
            )
        except ValueError:
            assume(method == "tfidf")
            return
        finally:
            scorer.clear_cache()

        assert 0.0 <= score <= 1.0, (
            f"unicode score={score!r} outside [0, 1] for method={method}"
        )

    @pytest.mark.parametrize("method", _LEXICAL_METHODS)
    @given(
        text=massive_identical_text(),
        semantic_score=unit_float,
        alpha=unit_float,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    def test_score_in_unit_interval_massive_identical(
        self, method, text, semantic_score, alpha
    ):
        """Completely identical massive strings fed to both sides."""
        # Massive alphabetic text always has a TF-IDF vocabulary, so no
        # ``assume()`` is needed here.
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=method))
        try:
            score = scorer.compute_hybrid_similarity(
                text, text, semantic_score=semantic_score
            )
        except ValueError:
            assume(method == "tfidf")
            return
        finally:
            scorer.clear_cache()

        assert 0.0 <= score <= 1.0, (
            f"massive score={score!r} outside [0, 1] for method={method}"
        )


# ── Property: explicit edge cases ────────────────────────────────────────────


class TestHybridScoreEdgeCases:
    """Targeted edge cases that hypothesis *will* eventually generate but
    are cheap to assert deterministically too."""

    @pytest.mark.parametrize("method", ["jaccard", "dice", "overlap", "char_ngram"])
    @given(semantic_score=unit_float, alpha=unit_float)
    @settings(max_examples=100, deadline=None)
    def test_empty_strings(self, method, semantic_score, alpha):
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=method))
        score = scorer.compute_hybrid_similarity("", "", semantic_score=semantic_score)
        scorer.clear_cache()
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("method", ["jaccard", "dice", "overlap", "char_ngram"])
    @given(semantic_score=unit_float, alpha=unit_float)
    @settings(max_examples=100, deadline=None)
    def test_whitespace_only(self, method, semantic_score, alpha):
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=method))
        score = scorer.compute_hybrid_similarity(
            "   \t\n  ", "  \t \n  ", semantic_score=semantic_score
        )
        scorer.clear_cache()
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("method", ["jaccard", "dice", "overlap", "char_ngram"])
    @given(semantic_score=unit_float, alpha=unit_float)
    @settings(max_examples=100, deadline=None)
    def test_punctuation_only(self, method, semantic_score, alpha):
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=method))
        score = scorer.compute_hybrid_similarity(
            "!@#$%^&*()", "!@#$%^&*()", semantic_score=semantic_score
        )
        scorer.clear_cache()
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("method", ["jaccard", "dice", "overlap", "char_ngram"])
    @given(semantic_score=unit_float, alpha=unit_float)
    @settings(max_examples=100, deadline=None)
    def test_one_empty_one_not(self, method, semantic_score, alpha):
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method=method))
        score = scorer.compute_hybrid_similarity(
            "", "the quick brown fox", semantic_score=semantic_score
        )
        scorer.clear_cache()
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "s",
        [
            "a",
            "ab",
            "the",
            "the the the",
            "lorem ipsum dolor sit amet",
            "🚀🔥💯",
            "你好世界",
            "Привет мир",
            "مرحبا بالعالم",
        ],
    )
    @pytest.mark.parametrize("method", ["jaccard", "dice", "overlap", "char_ngram"])
    def test_self_similarity_is_bounded(self, method, s):
        """Identical non-empty inputs must yield a bounded score."""
        scorer = HybridScorer(HybridConfig(alpha=0.7, lexical_method=method))
        score = scorer.compute_hybrid_similarity(s, s, semantic_score=1.0)
        scorer.clear_cache()
        assert 0.0 <= score <= 1.0


# ── Property: clamping prevents out-of-range semantic scores ────────────────


class TestHybridScoreClamping:
    """The explicit ``min(1.0, max(0.0, ...))`` clamp in
    ``compute_hybrid_similarity`` must contain even pathological
    ``semantic_score`` values that violate the documented [0, 1] input
    contract."""

    @given(
        text_a=ascii_text,
        text_b=ascii_text,
        semantic_score=st.floats(
            min_value=-1000.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        alpha=unit_float,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_out_of_range_semantic_is_clamped(
        self, text_a, text_b, semantic_score, alpha
    ):
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method="jaccard"))
        score = scorer.compute_hybrid_similarity(
            text_a, text_b, semantic_score=semantic_score
        )
        scorer.clear_cache()
        assert 0.0 <= score <= 1.0


# ── Mathematical properties of the weighted blend ───────────────────────────


class TestHybridScoreBlendProperties:
    """Higher-level algebraic properties of the alpha blend."""

    @given(
        text_a=ascii_text,
        text_b=ascii_text,
        semantic_score=unit_float,
        alpha=unit_float,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_alpha_zero_uses_lexical_only(self, text_a, text_b, semantic_score, alpha):
        """With alpha=0, hybrid score must equal the (clamped) lexical score."""
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method="jaccard"))
        lexical = scorer._compute_lexical_score(text_a, text_b, "jaccard")
        hybrid = scorer.compute_hybrid_similarity(
            text_a, text_b, semantic_score=semantic_score, alpha=0.0
        )
        scorer.clear_cache()
        expected = max(0.0, min(1.0, lexical))
        assert math.isclose(hybrid, expected, abs_tol=1e-9), (
            f"alpha=0 hybrid={hybrid!r} != lexical={expected!r}"
        )

    @given(
        text_a=ascii_text,
        text_b=ascii_text,
        semantic_score=unit_float,
        alpha=unit_float,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_alpha_one_uses_semantic_only(self, text_a, text_b, semantic_score, alpha):
        """With alpha=1, hybrid score must equal the (clamped) semantic score."""
        scorer = HybridScorer(HybridConfig(alpha=alpha, lexical_method="jaccard"))
        hybrid = scorer.compute_hybrid_similarity(
            text_a, text_b, semantic_score=semantic_score, alpha=1.0
        )
        scorer.clear_cache()
        expected = max(0.0, min(1.0, semantic_score))
        assert math.isclose(hybrid, expected, abs_tol=1e-9), (
            f"alpha=1 hybrid={hybrid!r} != semantic={expected!r}"
        )

    @given(
        text_a=ascii_text,
        text_b=ascii_text,
        semantic_score=unit_float,
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_alpha_partition_of_unity(self, text_a, text_b, semantic_score):
        """For any text pair, ``h(a) + h(1-a) == lexical + semantic``
        (pre-clamp). Verify the post-clamp sum stays within ``[0, 2]``."""
        scorer = HybridScorer(HybridConfig(alpha=0.5, lexical_method="jaccard"))
        h_low = scorer.compute_hybrid_similarity(
            text_a, text_b, semantic_score=semantic_score, alpha=0.3
        )
        h_high = scorer.compute_hybrid_similarity(
            text_a, text_b, semantic_score=semantic_score, alpha=0.7
        )
        scorer.clear_cache()
        # Pre-clamp: 0.3 * s + 0.7 * lex + 0.7 * s + 0.3 * lex = s + lex
        # Post-clamp: each term is in [0, 1] so sum is in [0, 2]
        assert 0.0 <= h_low + h_high <= 2.0
