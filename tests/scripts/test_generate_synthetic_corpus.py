from __future__ import annotations

"""
test_generate_synthetic_corpus.py
---------------------------------
Unit tests for the synthetic corpus generator script.

Validates:
- Text mutation functions (synonyms, reordering, deletion, noise)
- Corpus generation logic and file output
- Manifest creation and metadata accuracy
- CLI argument parsing and validation
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch
import random

import pytest

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_synthetic_corpus as gen_corpus


# ─── Mutation Function Tests ──────────────────────────────────────────────────


def test_replace_synonyms_basic():
    """Verify synonym replacement occurs for known words."""
    text = "The critical component is essential."
    random.seed(42)
    mutated = gen_corpus.replace_synonyms(text, mutation_ratio=1.0)

    # "critical" and "essential" should be replaced
    assert "critical" not in mutated.lower() or "component" not in mutated.lower()
    assert len(mutated) > 0


def test_replace_synonyms_preserves_capitalization():
    """Verify original capitalization is preserved after synonym replacement."""
    text = "Critical systems are vital."
    random.seed(42)
    mutated = gen_corpus.replace_synonyms(text, mutation_ratio=1.0)

    # First word should remain capitalized
    assert mutated[0].isupper()


def test_replace_synonyms_zero_ratio():
    """With 0.0 ratio, text should remain unchanged."""
    text = "The critical component is essential."
    mutated = gen_corpus.replace_synonyms(text, mutation_ratio=0.0)
    assert mutated == text


def test_reorder_words_basic():
    """Verify word reordering occurs within sentences."""
    text = "The quick brown fox jumps over the lazy dog."
    random.seed(42)
    mutated = gen_corpus.reorder_words(text, mutation_ratio=1.0)

    # Text should be different but contain same words (mostly)
    assert len(mutated) > 0
    assert mutated != text


def test_reorder_words_preserves_punctuation():
    """Verify sentences still end with proper punctuation after reordering."""
    text = "First clause, second clause, third clause."
    random.seed(42)
    mutated = gen_corpus.reorder_words(text, mutation_ratio=1.0)

    assert mutated.endswith(".")


def test_delete_sentences_basic():
    """Verify sentences are deleted based on ratio."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    random.seed(42)
    mutated = gen_corpus.delete_sentences(text, mutation_ratio=0.5)

    # Should have fewer sentences than original
    orig_count = len(gen_corpus._tokenize_sentences(text))
    mut_count = len(gen_corpus._tokenize_sentences(mutated))
    assert mut_count < orig_count


def test_delete_sentences_keeps_at_least_one():
    """Verify at least one sentence remains even with 1.0 deletion ratio."""
    text = "Only one sentence here."
    mutated = gen_corpus.delete_sentences(text, mutation_ratio=1.0)
    assert len(mutated) > 0


def test_insert_noise_basic():
    """Verify filler phrases are inserted into text."""
    text = (
        "This is a relatively long sentence that should accept noise insertion easily."
    )
    random.seed(42)
    mutated = gen_corpus.insert_noise(text, mutation_ratio=1.0)

    # Mutated text should be longer due to inserted filler
    assert len(mutated) > len(text)


def test_apply_mutations_combines_strategies():
    """Verify apply_mutations runs all mutation strategies."""
    text = "The critical component allows educators to detect paraphrased content using traditional tools."
    random.seed(42)
    mutated = gen_corpus.apply_mutations(text, intensity=0.8)

    assert mutated != text
    assert len(mutated) > 0


# ─── Corpus Generation Tests ──────────────────────────────────────────────────


def test_generate_corpus_creates_files(tmp_path):
    """Verify correct number of files are created in output directory."""
    manifest = gen_corpus.generate_corpus(
        num_docs=10,
        plagiarism_ratio=0.3,
        output_dir=tmp_path,
    )

    # 10 docs total
    txt_files = list(tmp_path.glob("*.txt"))
    assert len(txt_files) == 10

    # Manifest should have 10 entries
    assert len(manifest) == 10


def test_generate_corpus_manifest_structure(tmp_path):
    """Verify manifest.json has correct structure and metadata."""
    gen_corpus.generate_corpus(
        num_docs=5,
        plagiarism_ratio=0.4,
        output_dir=tmp_path,
    )

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 5

    for entry in data:
        assert "filename" in entry
        assert "type" in entry
        assert entry["type"] in ["original", "plagiarized"]
        assert entry["filename"].endswith(".txt")


def test_generate_corpus_plagiarism_ratio(tmp_path):
    """Verify the actual ratio of plagiarized docs matches requested ratio."""
    random.seed(42)
    manifest = gen_corpus.generate_corpus(
        num_docs=100,
        plagiarism_ratio=0.3,
        output_dir=tmp_path,
    )

    plag_count = sum(1 for m in manifest if m["type"] == "plagiarized")
    # Allow small variance due to integer rounding
    assert 25 <= plag_count <= 35


def test_generate_corpus_original_has_no_source(tmp_path):
    """Verify original documents have null source in manifest."""
    manifest = gen_corpus.generate_corpus(
        num_docs=10,
        plagiarism_ratio=0.0,  # All original
        output_dir=tmp_path,
    )

    for entry in manifest:
        assert entry["type"] == "original"
        assert entry["source"] is None


def test_generate_corpus_plagiarized_has_source(tmp_path):
    """Verify plagiarized documents reference a valid source file."""
    manifest = gen_corpus.generate_corpus(
        num_docs=10,
        plagiarism_ratio=1.0,  # All plagiarized
        output_dir=tmp_path,
    )



    for entry in manifest:
        assert entry["type"] == "plagiarized"
        assert entry["source"] is not None
        # Source should be one of the generated files (or base)
        # Since all are plagiarized, source might be base_0.txt if no originals
        # But normally it should be in filenames


# ─── CLI Argument Parsing Tests ───────────────────────────────────────────────


def test_parse_arguments_defaults():
    """Verify default CLI argument values."""
    with patch("sys.argv", ["generate_synthetic_corpus.py"]):
        args = gen_corpus.parse_arguments()

    assert args.num_docs == 10
    assert args.plagiarism_ratio == 0.3
    assert args.output_dir == "data/synthetic"
    assert args.seed is None


def test_parse_arguments_custom():
    """Verify custom CLI argument values are parsed correctly."""
    test_args = [
        "generate_synthetic_corpus.py",
        "--num-docs",
        "50",
        "--plagiarism-ratio",
        "0.5",
        "--output-dir",
        "custom/dir",
        "--seed",
        "123",
    ]

    with patch("sys.argv", test_args):
        args = gen_corpus.parse_arguments()

    assert args.num_docs == 50
    assert args.plagiarism_ratio == 0.5
    assert args.output_dir == "custom/dir"
    assert args.seed == 123


def test_parse_arguments_invalid_num_docs():
    """Verify invalid num_docs raises SystemExit."""
    test_args = ["generate_synthetic_corpus.py", "--num-docs", "0"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            gen_corpus.parse_arguments()


def test_parse_arguments_invalid_ratio():
    """Verify invalid plagiarism ratio raises SystemExit."""
    test_args = ["generate_synthetic_corpus.py", "--plagiarism-ratio", "1.5"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            gen_corpus.parse_arguments()
