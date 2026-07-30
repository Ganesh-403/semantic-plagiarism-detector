"""
test_rate_limiting.py
--------------------
Tests for rate limiting functionality (login and upload rate limits).
"""

import pytest

from src.db.auth import check_login_rate_limit
from src.db.auth import clear_login_attempts as auth_clear_login_attempts
from src.db.auth import record_failed_login
from src.utils.redis_cache import (CacheKeyPrefix, clear_login_attempts,
                                   get_cache, get_login_attempts,
                                   get_upload_count, increment_login_attempts,
                                   increment_upload_count,
                                   is_login_locked_out, is_upload_rate_limited)


@pytest.fixture(autouse=True)
def require_redis():
    """Skip tests if Redis is not available."""
    cache = get_cache()
    if not cache.is_available():
        pytest.skip("Redis not available - skipping rate limiting tests")


def test_login_rate_limiting():
    """Test that login rate limiting works correctly."""
    username = "testuser"

    # Clear any existing attempts
    clear_login_attempts(username)

    # Initially should not be locked out
    assert not is_login_locked_out(username)
    assert get_login_attempts(username) == 0

    # Add 4 failed attempts - should still be allowed
    for _ in range(4):
        increment_login_attempts(username)

    assert not is_login_locked_out(username)
    assert get_login_attempts(username) == 4

    # Add 5th attempt - should be locked out
    increment_login_attempts(username)
    assert is_login_locked_out(username)
    assert get_login_attempts(username) == 5

    # Clear attempts
    clear_login_attempts(username)
    assert not is_login_locked_out(username)
    assert get_login_attempts(username) == 0


def test_auth_login_rate_limiting():
    """Test auth module rate limiting functions."""
    username = "testuser2"

    # Clear any existing attempts
    auth_clear_login_attempts(username)

    # Initially should be allowed
    is_allowed, error_msg = check_login_rate_limit(username)
    assert is_allowed is True
    assert error_msg is None

    # Record 5 failed attempts
    for _ in range(5):
        record_failed_login(username)

    # Should now be locked out
    is_allowed, error_msg = check_login_rate_limit(username)
    assert is_allowed is False
    assert error_msg is not None
    assert "too many failed attempts" in error_msg

    # Clear on successful login
    auth_clear_login_attempts(username)
    is_allowed, error_msg = check_login_rate_limit(username)
    assert is_allowed is True


def test_upload_rate_limiting():
    """Test that upload rate limiting works correctly."""
    username = "testuser"

    # Clear any existing count by setting to 0
    from src.utils.redis_cache import get_cache

    cache = get_cache()
    cache.delete(f"{CacheKeyPrefix.LEGACY_UPLOADS_PREFIX.value}{username}")

    # Initially should not be rate limited
    assert not is_upload_rate_limited(username)
    assert get_upload_count(username) == 0

    # Add 99 uploads - should still be allowed
    for _ in range(99):
        increment_upload_count(username)

    assert not is_upload_rate_limited(username)
    assert get_upload_count(username) == 99

    # Add 100th upload - should be rate limited
    increment_upload_count(username)
    assert is_upload_rate_limited(username)
    assert get_upload_count(username) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
