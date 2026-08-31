# tests/core/test_core_init.py


def test_with_sqlite_retry_accessible_via_getattr():
    """Verify with_sqlite_retry is accessible via __getattr__ at runtime."""
    from src import core

    # Should not raise AttributeError
    assert hasattr(core, "with_sqlite_retry")

    # Should be callable
    retry_decorator = core.with_sqlite_retry
    assert callable(retry_decorator)


def test_with_sqlite_retry_in_all_list():
    """Verify with_sqlite_retry is listed in __all__ for proper re-export."""
    from src import core

    assert "with_sqlite_retry" in core.__all__


def test_invalid_attribute_raises_attribute_error():
    """Verify accessing non-existent attributes raises AttributeError."""
    import pytest

    from src import core

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = core.nonexistent_function_xyz
