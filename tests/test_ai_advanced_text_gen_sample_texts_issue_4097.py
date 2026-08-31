"""
tests/test_ai_advanced_text_gen_sample_texts_issue_4097.py
-----------------------------------------------------------
Regression tests for the unterminated string literals in
``AI_ADVANCED_Text_Gen.py`` (issue #4097).

``_generate_sample_ai_texts()`` and ``_generate_sample_human_texts()`` each
return a list of five demonstration strings. Every one of them was written as a
single-quoted literal wrapped across several physical lines:

    "Artificial intelligence represents a paradigm shift in computational capabilities,
    offering unprecedented opportunities for automation and cognitive augmentation
    across multiple sectors of society.",

Python has no line continuation inside an ordinary string literal, so the first
one was unterminated and the whole ~1,300-line module failed to compile. Nothing
in it could be imported.

Ten literals were affected across the two methods. They are now written as
implicit concatenation, which is the shape the original layout was reaching
for: the prose reads as one continuous sentence with single spaces and no
embedded newlines.

The import at the top of this file is itself the regression test. The explicit
assertions below state the intent so the reason is not lost, and pin the
properties that make these strings usable as detector input -- because a
"fixed" version that merely compiles while leaving newlines and column padding
embedded in the prose would be no more useful than the broken one.
"""

import re
from pathlib import Path

import pytest

import AI_ADVANCED_Text_Gen
from AI_ADVANCED_Text_Gen import AITextDetectionSystem

MODULE_PATH = Path(AI_ADVANCED_Text_Gen.__file__)


@pytest.fixture(scope="module")
def module_source():
    return MODULE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ai_texts():
    """The methods do not touch ``self``, so they run unbound.

    ``AITextDetectionSystem.__init__`` builds classifiers and skill registries;
    none of that is needed to read a literal list.
    """
    return AITextDetectionSystem._generate_sample_ai_texts(None)


@pytest.fixture(scope="module")
def human_texts():
    return AITextDetectionSystem._generate_sample_human_texts(None)


@pytest.fixture(scope="module")
def all_texts(ai_texts, human_texts):
    return list(ai_texts) + list(human_texts)


# ── the module compiles and imports ────────────────────────────────────────────


def test_module_compiles(module_source):
    """Before the fix: SyntaxError, unterminated string literal at line 625."""
    compile(module_source, MODULE_PATH.name, "exec")


def test_module_imports():
    """The import at the top of this file already proves it; assert it anyway."""
    assert AI_ADVANCED_Text_Gen is not None
    assert hasattr(AI_ADVANCED_Text_Gen, "AITextDetectionSystem")


def test_sample_text_methods_exist():
    for name in ("_generate_sample_ai_texts", "_generate_sample_human_texts"):
        assert callable(getattr(AITextDetectionSystem, name))


# ── both lists are intact ──────────────────────────────────────────────────────


def test_ai_sample_list_has_five_entries(ai_texts):
    """All five survived the repair; none were merged or dropped."""
    assert len(ai_texts) == 5


def test_human_sample_list_has_five_entries(human_texts):
    assert len(human_texts) == 5


def test_every_entry_is_a_nonempty_string(all_texts):
    for text in all_texts:
        assert isinstance(text, str)
        assert text.strip()


def test_entries_are_distinct(all_texts):
    """Implicit concatenation is easy to get wrong by dropping a comma.

    A missing comma between two adjacent literals silently glues them into one
    entry, which would show up here as a short list or a duplicate.
    """
    assert len(set(all_texts)) == len(all_texts)


# ── the prose is usable, not merely syntactically valid ────────────────────────


def test_no_entry_contains_a_newline(all_texts):
    """The strings are single sentences, not wrapped blocks.

    A triple-quoted "fix" would compile while baking the source indentation
    into the prose. These are fed to a tokenizer and a vectorizer, so embedded
    newlines and column padding would corrupt the demonstration input.
    """
    for text in all_texts:
        assert "\n" not in text


def test_no_entry_contains_a_run_of_spaces(all_texts):
    """Concatenation boundaries must not double up or swallow a space."""
    for text in all_texts:
        assert "  " not in text, repr(text)


def test_no_entry_has_leading_or_trailing_whitespace(all_texts):
    for text in all_texts:
        assert text == text.strip()


def test_every_entry_ends_with_terminal_punctuation(all_texts):
    """Each sample is a complete sentence, so the closing quote is in the right place."""
    for text in all_texts:
        assert text.rstrip()[-1] in ".!?", repr(text[-40:])


def test_every_entry_is_long_enough_to_be_a_sample(all_texts):
    """A boundary lost mid-repair would leave a truncated fragment behind."""
    for text in all_texts:
        assert len(text) > 80, repr(text)
        assert len(text.split()) >= 12, repr(text)


# ── the content the samples were written to demonstrate ────────────────────────


@pytest.mark.parametrize(
    "phrase",
    [
        "paradigm shift in computational capabilities",
        "machine learning algorithms into healthcare systems",
        "solar photovoltaic systems achieving record efficiency",
        "Climate change mitigation strategies",
        "Quantum computing promises to revolutionize computational chemistry",
    ],
)
def test_ai_samples_keep_their_subject_matter(ai_texts, phrase):
    """The repair joined lines; it must not have reworded anything.

    Each phrase here spans a point where the original literal was broken across
    physical lines, so a lost or duplicated word at a join would fail.
    """
    joined = " ".join(ai_texts)
    assert phrase in joined


@pytest.mark.parametrize(
    "phrase",
    [
        "I think AI is really changing how we do things",
        "solar panels installed at our school",
        "something special about human doctors and nurses",
        "reduce my carbon footprint",
        "I'm not sure about all this AI stuff",
    ],
)
def test_human_samples_keep_their_subject_matter(human_texts, phrase):
    joined = " ".join(human_texts)
    assert phrase in joined


def test_human_samples_keep_their_contractions(human_texts):
    """Apostrophes inside double-quoted literals must survive untouched."""
    joined = " ".join(human_texts)
    for contraction in ("It's", "I'm", "We've", "there's", "it'll"):
        assert contraction in joined


def test_the_two_sample_sets_do_not_overlap(ai_texts, human_texts):
    """The detector contrasts these two sets; they must stay disjoint."""
    assert not set(ai_texts) & set(human_texts)


# ── no unterminated literals left anywhere in the file ─────────────────────────


def test_no_source_line_ends_mid_literal(module_source):
    """A trailing space after an unclosed quote was the tell in the original."""
    pattern = re.compile(r'^\s*"[^"]*[^",]\s+$')
    offenders = [
        i + 1 for i, line in enumerate(module_source.split("\n")) if pattern.match(line)
    ]
    assert not offenders, f"lines opening a quote without closing it: {offenders}"
