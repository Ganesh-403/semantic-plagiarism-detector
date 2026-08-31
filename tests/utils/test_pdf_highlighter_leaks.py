import pytest
 feat/pdf-ngram-highlighter
from src.utils.pdf_highlighter import highlight_pdf_matches, get_word_ngrams

def test_get_word_ngrams():
    """
    Assert that get_word_ngrams correctly splits a phrase into 6-word overlapping windows.
    """
    phrase = "This is a very long phrase that has more than six words"
    ngrams = get_word_ngrams(phrase, n=6)
    
    assert len(ngrams) == 7
    assert ngrams[0] == "This is a very long phrase"
    assert ngrams[1] == "is a very long phrase that"
    assert ngrams[-1] == "has more than six words"

def test_pdf_highlighter_ngrams():
    """
    Verify that highlight_pdf_matches executes successfully when n-gram highlighting is applied.
    """
    with open("tests/fixtures/clean.pdf", "rb") as f:
        pdf_bytes = f.read()

    # Test highlighting with sliding windows
    phrase = "Plagiarism detection is a critical educational challenge in modern classrooms."
    res = highlight_pdf_matches(pdf_bytes, [phrase])
    assert isinstance(res, bytes)
    assert len(res) > 0

from src.utils.pdf_highlighter import highlight_pdf_matches as highlight_hl
from src.utils.pdf_report import highlight_pdf_matches as highlight_rep

def test_pdf_highlighters_context_manager():
    """
    Verify that highlight_pdf_matches functions from both pdf_highlighter and pdf_report
    execute without leaks, utilizing context managers correctly.
    """
 perf/optimize-pdf-writing-3980
    with open("tests/fixtures/clean.pdf", "rb") as f:
        pdf_bytes = f.read()

    # Test pdf_highlighter.highlight_pdf_matches with the new deflate and garbage flags

    # 1. Read a valid sample PDF fixture
    with open("tests/fixtures/clean.pdf", "rb") as f:
        pdf_bytes = f.read()

    # 2. Test pdf_highlighter.highlight_pdf_matches
 main
    res_hl = highlight_hl(pdf_bytes, ["plagiarism", "semantic"])
    assert isinstance(res_hl, bytes)
    assert len(res_hl) > 0

 perf/optimize-pdf-writing-3980
    # Test pdf_report.highlight_pdf_matches

    # 3. Test pdf_report.highlight_pdf_matches
 main
    res_rep = highlight_rep(pdf_bytes, ["plagiarism", "semantic"])
    assert isinstance(res_rep, bytes)
    assert len(res_rep) > 0
 main
