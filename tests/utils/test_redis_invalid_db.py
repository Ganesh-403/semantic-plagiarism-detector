"""
tests/utils/test_redis_invalid_db.py
------------------------------------
Unit tests for handling invalid REDIS_DB configurations gracefully (Issue #2818).
"""

import importlib
import logging


def test_invalid_redis_db_falls_back_to_zero_and_logs_warning(caplog, monkeypatch):
    """
    Verify that if REDIS_DB is set to a non-integer string like 'db1',
    ValueError is caught, a warning is logged, and REDIS_DB defaults to 0.
    """
    monkeypatch.setenv("REDIS_DB", "db1")

    with caplog.at_level(logging.WARNING):
        import src.utils.redis_cache

        importlib.reload(src.utils.redis_cache)

    assert src.utils.redis_cache.REDIS_DB == 0
    assert any(
        "Invalid REDIS_DB configuration 'db1'. Defaulting to 0." in record.message
        for record in caplog.records
    )


def test_valid_redis_db_is_parsed_correctly(monkeypatch):
    """Verify that valid integer strings for REDIS_DB are parsed without issue."""
    monkeypatch.setenv("REDIS_DB", "3")
    import src.utils.redis_cache

    importlib.reload(src.utils.redis_cache)

    assert src.utils.redis_cache.REDIS_DB == 3
