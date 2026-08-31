"""
tests/app/test_writing_style_generator_issue_4095.py
-----------------------------------------------------
Regression tests for the malformed generator filter in
``app/pages/5_Writing_Style_Analyzer.py`` (issue #4095).

The passive-voice counter in ``_compute_style_profile()`` was written as:

    past_participle_count = sum(1 for i, w in enumerate(words)
                               if w.lower() in be_forms and i + 1 < len(words)
                               and words[i + 1][0].isupper() if words[i + 1] else False)

A generator's filter clause takes a plain expression, not a conditional. The
trailing ``if words[i + 1]`` parsed as a *second* filter and ``else`` was then
unexpected, so the file did not compile and the entire Writing Style Analyzer
page was unreachable.

The ``if words[i + 1] else False`` was reaching for a real guard, though:
``words[i + 1][0]`` raises ``IndexError`` on an empty token, and the tokenizer
in this module can emit those. In a filter clause that guard belongs as an
ordinary ``and`` term placed *before* the subscript, where short-circuiting
does the work. The tests below pin both halves — that the file parses, and
that the guard actually guards.

Because the page runs ``st.set_page_config()`` at import and calls
``render_writing_style_analyzer()`` at the bottom, the ``page_module`` fixture
loads only the module's imports and function definitions.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PAGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "pages"
    / "5_Writing_Style_Analyzer.py"
)

STUBBED_MODULES = ("streamlit",)

# Top-level statements worth executing. Everything else in this page is
# rendering: st.set_page_config() at the top, and a
# `if __name__ == "__main__" or True: render_writing_style_analyzer()` block at
# the bottom that would draw the whole view.
LOADABLE_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)


@pytest.fixture(scope="module")
def page_source():
    return PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page_tree(page_source):
    """Parsing at all is the primary regression assert."""
    return ast.parse(page_source, filename=PAGE_PATH.name)


@pytest.fixture(scope="module")
def page_module(page_tree):
    """Execute the page's imports and defs, but not its rendering statements."""
    saved = {name: sys.modules.get(name) for name in STUBBED_MODULES}
    for name in STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    try:
        body = [node for node in page_tree.body if isinstance(node, LOADABLE_NODES)]
        namespace = {"__name__": "writing_style_isolated"}
        exec(  # noqa: S102 - deliberately loading a page module without running it
            compile(ast.Module(body=body, type_ignores=[]), PAGE_PATH.name, "exec"),
            namespace,
        )
        yield namespace
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def profile(page_module):
    """``_compute_style_profile`` bound for convenience."""
    return page_module["_compute_style_profile"]


# ── the page parses ────────────────────────────────────────────────────────────


def test_page_compiles(page_source):
    """The module must compile. Before the fix: SyntaxError at line 101."""
    compile(page_source, PAGE_PATH.name, "exec")


def test_no_generator_filter_holds_a_conditional_expression(page_tree):
    """No comprehension filter anywhere may be an ``IfExp``.

    This is the shape that broke, generalised: an ``if A else B`` sitting in a
    filter clause is always a mistake, because the filter is already the
    condition. Walking the tree stops the same slip reappearing elsewhere in
    the file rather than pinning only the one line that happened to break.
    """
    offenders = []
    for node in ast.walk(page_tree):
        if not isinstance(
            node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)
        ):
            continue
        for generator in node.generators:
            for condition in generator.ifs:
                if isinstance(condition, ast.IfExp):
                    offenders.append(node.lineno)

    assert not offenders, f"conditional expression used as a filter at {offenders}"


def test_compute_style_profile_is_callable(page_module):
    assert callable(page_module["_compute_style_profile"])


# ── the guard the broken clause was reaching for ───────────────────────────────


def test_empty_tokens_do_not_raise(profile):
    """An empty token after a be-form must not raise ``IndexError``.

    ``words[i + 1][0]`` on ``""`` is the crash the discarded ``if words[i + 1]``
    was trying to prevent. The fix keeps that guard as an ``and`` term ordered
    before the subscript so short-circuiting skips it.
    """
    result = profile("The result was  Measured by the team.", author="T")
    assert isinstance(result["passive_indicators"], int)


def test_be_form_as_the_final_token_does_not_raise(profile):
    """A be-form at the very end has no ``i + 1``; the bounds check holds."""
    result = profile("Nothing here was", author="T")
    assert result["passive_indicators"] == 0


def test_counter_is_never_negative(profile):
    for text in ("", "   ", "One.", "It is Done.", "was was was"):
        assert profile(text, author="T")["passive_indicators"] >= 0


# ── the counter still counts what it counted ───────────────────────────────────


def test_counts_a_be_form_followed_by_a_capitalised_word(profile):
    """The original predicate is preserved exactly: be-form, then a capital."""
    result = profile("The paper was Reviewed carefully.", author="T")
    assert result["passive_indicators"] == 1


def test_counts_each_occurrence(profile):
    result = profile("It was Written. They were Graded. He is Known.", author="T")
    assert result["passive_indicators"] == 3


def test_does_not_count_a_be_form_followed_by_a_lowercase_word(profile):
    result = profile("The paper was reviewed carefully.", author="T")
    assert result["passive_indicators"] == 0


def test_does_not_count_a_capitalised_word_without_a_be_form(profile):
    result = profile("The paper Reviewed carefully.", author="T")
    assert result["passive_indicators"] == 0


def test_be_form_matching_is_case_insensitive(profile):
    """``w.lower() in be_forms`` must survive; a sentence-initial ``Was`` counts."""
    result = profile("Was Reviewed by the panel.", author="T")
    assert result["passive_indicators"] == 1


def test_am_is_not_a_be_form(profile):
    """``be_forms`` deliberately omits ``am``; that list is what the fix kept.

    The unused ``passive_words`` list beside it did include ``am``, which is
    why it was easy to mistake for the live one. It has been removed.
    """
    result = profile("I am Certain about this.", author="T")
    assert result["passive_indicators"] == 0


def test_unused_passive_words_list_is_gone(page_source):
    """``passive_words`` was assigned and never read; ``be_forms`` is the live one."""
    assert "passive_words" not in page_source


# ── the rest of the profile still builds ───────────────────────────────────────


def test_profile_reports_the_passive_indicator_key(profile):
    assert "passive_indicators" in profile("Some sample text here.", author="T")


def test_profile_handles_empty_input(profile):
    """An empty document must produce a profile, not a division error."""
    result = profile("", author="T")
    assert result["word_count"] == 0
    assert result["passive_indicators"] == 0


def test_profile_keeps_its_other_metrics(profile):
    """The neighbouring metrics were untouched and must still be produced."""
    result = profile(
        "The study was Completed. Results were Published quickly. "
        "Researchers carefully reviewed everything.",
        author="Ada",
    )
    for key in (
        "author",
        "word_count",
        "sentence_count",
        "vocab_richness",
        "fk_grade",
        "adverb_density",
        "complex_word_pct",
    ):
        assert key in result
    assert result["author"] == "Ada"
    assert result["word_count"] > 0
