from pathlib import Path


SOURCE = Path("src/core/cross_lingual.py")
TESTS = Path("tests/core/test_cross_lingual.py")


def test_translation_memory_uses_sha256_sentence_hashing():
    source = SOURCE.read_text(encoding="utf-8")

    assert "class TranslationMemoryCache:" in source
    assert "hashlib.sha256(" in source
    assert 'sentence: str' in source


def test_cache_lookup_occurs_before_translator_call():
    source = SOURCE.read_text(encoding="utf-8")

    cache_lookup = source.index("cached_translation = cache.get(")
    translator_call = source.index(
        "translated_text = translator_fn(",
        cache_lookup,
    )
    assert cache_lookup < translator_call


def test_identical_sentence_cache_hit_test_exists():
    source = TESTS.read_text(encoding="utf-8")

    assert (
        "test_translation_memory_cache_hits_for_identical_sentence"
        in source
    )
    assert "assert len(calls) == 1" in source
