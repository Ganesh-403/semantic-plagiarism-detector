"""
tests/core/test_template_fingerprint.py
---------------------------------------
Unit tests for Document Formatting Entropy and Template Fingerprinting.
"""

import pytest
from src.core.formatting_entropy_extractor import (
    extract_latex_macros,
    compute_formatting_entropy,
)
from src.core.template_fingerprinter import (
    generate_template_fingerprint,
    compare_template_fingerprints,
)


class TestFormattingEntropyExtractor:
    def test_extract_latex_macros(self):
        source = r"""\documentclass{article}
\usepackage{amsmath}
\newcommand{\mycmd}{test}
\begin{document}
Hello
\end{document}"""
        macros = extract_latex_macros(source)
        assert "class:article" in macros
        assert "pkg:amsmath" in macros
        assert "macro:mycmd" in macros
        assert "env:document" in macros

    def test_compute_formatting_entropy_uniform(self):
        styles = ["style:A"] * 10
        entropy = compute_formatting_entropy(styles)
        assert entropy == 0.0

    def test_compute_formatting_entropy_diverse(self):
        styles = ["style:A", "style:B", "style:C", "style:D"]
        entropy = compute_formatting_entropy(styles)
        assert entropy > 0.0


class TestTemplateFingerprinter:
    def test_generate_template_fingerprint(self):
        styles = ["style:A", "style:B"]
        fp = generate_template_fingerprint(styles, "docx")
        assert "template_hash" in fp
        assert fp["style_count"] == 2

    def test_compare_template_fingerprints_exact(self):
        styles = ["style:A", "style:B"]
        fp_a = generate_template_fingerprint(styles)
        fp_b = generate_template_fingerprint(styles)
        result = compare_template_fingerprints(fp_a, fp_b)
        assert result["is_exact_match"] is True
        assert result["is_template_plagiarism"] is True

    def test_compare_template_fingerprints_different(self):
        fp_a = generate_template_fingerprint(["style:A"])
        fp_b = generate_template_fingerprint(["style:X", "style:Y", "style:Z"])
        result = compare_template_fingerprints(fp_a, fp_b)
        assert result["is_exact_match"] is False
        assert result["is_template_plagiarism"] is False
