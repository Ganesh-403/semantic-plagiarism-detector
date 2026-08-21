import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "utils" / "similarity_cache.py"
)
_SPEC = importlib.util.spec_from_file_location("similarity_cache", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

build_similarity_cache_key = _MODULE.build_similarity_cache_key


def test_lexical_cache_key_uses_lexical_suffix():
    assert build_similarity_cache_key("session-123", use_hybrid=False) == (
        "session-123:analysis_results_lexical"
    )


def test_hybrid_cache_key_uses_hybrid_v1_suffix():
    assert build_similarity_cache_key("session-123", use_hybrid=True) == (
        "session-123:analysis_results_hybrid_v1"
    )


def test_lexical_and_hybrid_cache_keys_are_distinct():
    lexical = build_similarity_cache_key("session-123", use_hybrid=False)
    hybrid = build_similarity_cache_key("session-123", use_hybrid=True)
    assert lexical != hybrid
