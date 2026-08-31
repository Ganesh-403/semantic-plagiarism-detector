"""
test_posix_path_cache_keys_issue_3028.py
-----------------------------------------
Unit tests for Issue #3028:
Ensures paths are explicitly converted to POSIX format using pathlib.Path(path).as_posix()
whenever creating cache keys based on file paths to guarantee cross-platform Redis key compatibility.
"""

import os
from pathlib import Path
from src.utils.redis_cache import (
    CacheNamespace,
    normalize_cache_key_path,
    store_large_data,
    get_large_data,
    clear_large_data,
    clear_all_large_data,
)
from src.utils.similarity_cache import build_similarity_cache_key


def test_normalize_cache_key_path_converts_windows_backslashes_to_posix():
    """Verify Windows paths with backslashes produce normalized POSIX keys."""
    win_path = r"uploads\user_123\reports\document.pdf"
    posix_expected = "uploads/user_123/reports/document.pdf"
    assert normalize_cache_key_path(win_path) == posix_expected
    assert "\\" not in normalize_cache_key_path(win_path)


def test_normalize_cache_key_path_handles_path_objects_and_drive_letters():
    """Verify pathlib.Path and Windows drive paths are properly converted."""
    drive_path = r"C:\Users\developer\data\corpus.index"
    assert normalize_cache_key_path(drive_path) == "C:/Users/developer/data/corpus.index"

    path_obj = Path("folder") / "subfolder" / "file.txt"
    assert normalize_cache_key_path(path_obj) == path_obj.as_posix()
    assert "\\" not in normalize_cache_key_path(path_obj)


from src.version import APP_VERSION


def test_cache_namespace_build_key_produces_consistent_posix_keys():
    """Verify CacheNamespace.build_key produces identical keys regardless of OS path separator."""
    win_path = r"documents\incident_001\scan.docx"
    linux_path = "documents/incident_001/scan.docx"

    key_from_win = CacheNamespace.ANALYSIS.build_key(win_path)
    key_from_linux = CacheNamespace.ANALYSIS.build_key(linux_path)

    assert key_from_win == key_from_linux
    assert key_from_win == f"spd:v1:analysis:{APP_VERSION}:documents/incident_001/scan.docx"
    assert "\\" not in key_from_win


def test_build_similarity_cache_key_cross_platform_parity():
    """Verify build_similarity_cache_key produces identical keys for Windows and Linux path inputs."""
    win_session = r"sessions\user_42\analysis_run"
    linux_session = "sessions/user_42/analysis_run"

    key_win_hybrid = build_similarity_cache_key(win_session, use_hybrid=True)
    key_linux_hybrid = build_similarity_cache_key(linux_session, use_hybrid=True)
    assert key_win_hybrid == key_linux_hybrid
    assert key_win_hybrid == "sessions/user_42/analysis_run:analysis_results_hybrid_v1"

    key_win_lexical = build_similarity_cache_key(win_session, use_hybrid=False)
    key_linux_lexical = build_similarity_cache_key(linux_session, use_hybrid=False)
    assert key_win_lexical == key_linux_lexical
    assert key_win_lexical == "sessions/user_42/analysis_run:analysis_results_lexical"


def test_large_data_cross_platform_store_and_retrieve():
    """Verify data stored with a Windows path key can be retrieved with a Linux path key."""
    win_key = r"large_matrices\session_99\matrix.pkl"
    linux_key = "large_matrices/session_99/matrix.pkl"
    payload = {"scores": [0.95, 0.82, 0.45], "status": "completed"}

    store_large_data(win_key, payload)

    # Retrieval via Linux-style POSIX path must succeed
    retrieved = get_large_data(linux_key)
    assert retrieved == payload

    # Retrieval via Windows-style path must also succeed
    retrieved_win = get_large_data(win_key)
    assert retrieved_win == payload

    # Clean up
    clear_large_data(win_key)
    assert get_large_data(linux_key) is None


def test_clear_all_large_data_with_windows_session_path():
    """Verify clear_all_large_data clears keys when given Windows session path."""
    win_session = r"sessions\tenant_1\user_99"
    item_key_win = r"sessions\tenant_1\user_99\doc_a"
    item_key_linux = "sessions/tenant_1/user_99/doc_a"

    store_large_data(item_key_win, {"data": 1})
    assert get_large_data(item_key_linux) == {"data": 1}

    clear_all_large_data(win_session)
    assert get_large_data(item_key_linux) is None
