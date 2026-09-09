from pathlib import Path
from typing import Union

from src.utils.redis_cache import normalize_cache_key_path


def build_similarity_cache_key(
    session_id: Union[str, Path], *, use_hybrid: bool
) -> str:
    """Build a mode-specific analysis cache key.

    Lexical and Hybrid scoring produce different result spaces. Keeping the
    mode in the key prevents a result generated in one mode from being reused
    after the UI switches to the other mode.
    Normalize Windows and POSIX spellings identically on every host OS.
    """
    sid_str = normalize_cache_key_path(session_id)

    suffix = "hybrid_v1" if use_hybrid else "lexical"
    return f"{sid_str}:analysis_results_{suffix}"

