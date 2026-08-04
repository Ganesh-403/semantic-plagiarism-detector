from unittest.mock import MagicMock

import pytest
import redis

from src.security.rate_limiter import RateLimiter, get_rate_limit_headers

@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    redis_mock = MagicMock(spec=redis.Redis)
    redis_mock.pipeline.return_value = MagicMock()
    return redis_mock

@pytest.fixture
def rate_limiter(mock_redis):
    """Create a RateLimiter instance with mock Redis."""
    return RateLimiter(
        redis_client=mock_redis,
        limit=5,
        window=60,
        block_duration=300,
        prefix="test_limit"
    )

def test_check_rate_limit_first_request(rate_limiter, mock_redis):
    """Test that the first request is allowed and returns correct headers."""
    mock_redis.get.return_value = None

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is True
    assert headers["X-RateLimit-Limit"] == "5"
    assert headers["X-RateLimit-Remaining"] == "4"
    assert "X-RateLimit-Reset" in headers
    assert headers["Retry-After"] == "60"

def test_check_rate_limit_within_limit(rate_limiter, mock_redis):
    """Test requests within the limit return correct remaining count."""
    mock_redis.get.return_value = "3"  # 3 requests already made

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is True
    assert headers["X-RateLimit-Limit"] == "5"
    assert headers["X-RateLimit-Remaining"] == "1"  # 5 - 3 - 1 = 1

def test_check_rate_limit_exceeded(rate_limiter, mock_redis):
    """Test that exceeding the limit returns False and Retry-After header."""
    mock_redis.get.return_value = "5"  # Limit reached
    mock_redis.exists.return_value = False

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is False
    assert headers["X-RateLimit-Limit"] == "5"
    assert headers["X-RateLimit-Remaining"] == "0"
    assert headers["Retry-After"] == "300"  # Block duration
    assert "X-RateLimit-Reset" in headers

    # Verify block key was set
    mock_redis.setex.assert_called_with("test_limit:blocked:192.168.1.1", 300, "1")

def test_check_rate_limit_already_blocked(rate_limiter, mock_redis):
    """Test that a blocked client is denied with Retry-After based on TTL."""
    mock_redis.exists.return_value = True
    mock_redis.ttl.return_value = 150  # 150 seconds remaining on block

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is False
    assert headers["X-RateLimit-Remaining"] == "0"
    assert headers["Retry-After"] == "150"
    assert "X-RateLimit-Reset" in headers

def test_check_rate_limit_retry_after_minimum(rate_limiter, mock_redis):
    """Test that Retry-After is at least 1 second when blocked."""
    mock_redis.exists.return_value = True
    mock_redis.ttl.return_value = 0  # Edge case: TTL is 0 or negative

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is False
    assert headers["Retry-After"] == "1"  # Should be max(1, 0)

def test_reset_limit(rate_limiter, mock_redis):
    """Test that resetting the limit deletes both count and block keys."""
    result = rate_limiter.reset_limit("192.168.1.1")

    assert result is True

    # Verify pipeline was called to delete both keys
    pipeline_mock = mock_redis.pipeline.return_value
    pipeline_mock.delete.assert_any_call("test_limit:192.168.1.1")
    pipeline_mock.delete.assert_any_call("test_limit:blocked:192.168.1.1")

def test_get_rate_limit_headers_convenience_function(mock_redis):
    """Test the convenience function returns the same as the class method."""
    mock_redis.get.return_value = None

    is_allowed, headers = get_rate_limit_headers(
        redis_client=mock_redis,
        identifier="10.0.0.1",
        limit=10,
        window=120
    )

    assert is_allowed is True
    assert headers["X-RateLimit-Limit"] == "10"
    assert headers["X-RateLimit-Remaining"] == "9"
    assert headers["Retry-After"] == "120"

def test_header_format_compliance(rate_limiter, mock_redis):
    """Test that all returned headers are strings as required by HTTP spec."""
    mock_redis.get.return_value = "2"

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    # All header values must be strings
    for key, value in headers.items():
        assert isinstance(value, str), f"Header {key} value must be string, got {type(value)}"

    # Verify specific header names match HTTP standards
    expected_headers = {"X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"}
    assert set(headers.keys()) == expected_headers
