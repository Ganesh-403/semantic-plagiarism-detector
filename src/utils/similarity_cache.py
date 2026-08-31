from pathlib import Path
from typing import Union


def build_similarity_cache_key(
    session_id: Union[str, Path], *, use_hybrid: bool
) -> str:
    """Build a mode-specific analysis cache key.

    Lexical and Hybrid scoring produce different result spaces. Keeping the
    mode in the key prevents a result generated in one mode from being reused
    after the UI switches to the other mode.
    Converts pathing logic using pathlib.Path(p).as_posix() explicitly for cross-platform support (Issue #2939, #3028).
    """
    if isinstance(session_id, Path):
        sid_str = session_id.as_posix()
    elif session_id is not None:
        sid_str = Path(str(session_id)).as_posix()
    else:
        sid_str = ""

    suffix = "hybrid_v1" if use_hybrid else "lexical"
    return f"{sid_str}:analysis_results_{suffix}"

