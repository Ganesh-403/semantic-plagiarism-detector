"""
test_normalize_cache_key_paths_issue_2939.py
---------------------------------------------
Unit test suite for Issue #2939:
Validates that Redis cache key pathing logic normalizes Windows backslashes (\)
to POSIX forward slashes (/) using pathlib.Path(p).as_posix() for cross-platform compatibility.
"""

from pathlib import Path
from src.utils.redis_cache import (
    CacheNamespace,
    normalize_cache_key_path,
    store_large_data,
    get_large_data,
)
from src.utils.similarity_cache import build_similarity_cache_key


def test_normalize_cache_key_path_converts_backslashes():
    """Verify normalize_cache_key_path converts Windows backslashes to POSIX slashes."""
    win_path = r"C:\var\backups\spd\db.sqlite"
    normalized = normalize_cache_key_path(win_path)
    assert "\\" not in normalized
    assert normalized == "C:/var/backups/spd/db.sqlite"


def test_normalize_cache_key_path_handles_path_objects(tmp_path):
    """Verify normalize_cache_key_path handles pathlib.Path objects."""
    path_obj = tmp_path / "sub" / "file.txt"
    normalized = normalize_cache_key_path(path_obj)
    assert "\\" not in normalized
    assert normalized == path_obj.as_posix()


from src.version import APP_VERSION


def test_cache_namespace_build_key_normalizes_paths():
    """Verify CacheNamespace.build_key normalizes path components into POSIX format."""
    win_path = r"uploads\user123\doc.pdf"
    key = CacheNamespace.UPLOADS.build_key(win_path)
    assert "\\" not in key
    assert key == f"spd:v1:uploads:{APP_VERSION}:uploads/user123/doc.pdf"


def test_build_similarity_cache_key_normalizes_path_session_id():
    """Verify build_similarity_cache_key converts Windows path session_ids using as_posix()."""
    win_session_id = r"session_folder\user_session_456"
    key = build_similarity_cache_key(win_session_id, use_hybrid=True)
    assert "\\" not in key
    assert key == "session_folder/user_session_456:analysis_results_hybrid_v1"


def test_store_and_get_large_data_with_path_keys():
    """Verify store_large_data and get_large_data handle Windows path keys."""
    win_key = r"large_cache\matrices\sim_matrix.bin"
    payload = {"data": [1, 2, 3]}

    store_large_data(win_key, payload)
    retrieved = get_large_data(win_key)
    assert retrieved == payload
