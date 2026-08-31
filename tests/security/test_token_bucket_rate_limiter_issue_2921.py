"""
test_token_bucket_rate_limiter_issue_2921.py
---------------------------------------------
Unit test suite for Issue #2921:
Validates Token-Bucket rate limiting per API Bearer token in FastAPI middleware.
Ensures distinct credentials.credentials strings maintain independent token buckets
and return HTTP 429 Too Many Requests when request rate limits are exceeded.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from src.api.middleware import verify_bearer_token
from src.security.rate_limiter import (
    TokenBucketRateLimiter,
    get_token_bucket_limiter,
)


def test_token_bucket_consume_basic():
    """Verify TokenBucketRateLimiter consumes tokens up to capacity and denies requests when empty."""
    limiter = TokenBucketRateLimiter(capacity=3.0, refill_rate=0.0)
    token = "test-token-123"

    assert limiter.consume(token) is True
    assert limiter.consume(token) is True
    assert limiter.consume(token) is True
    assert limiter.consume(token) is False


def test_token_bucket_refill():
    """Verify TokenBucketRateLimiter refills tokens over time."""
    limiter = TokenBucketRateLimiter(capacity=2.0, refill_rate=10.0)
    token = "test-token-refill"

    assert limiter.consume(token) is True
    assert limiter.consume(token) is True
    assert limiter.consume(token) is False

    time.sleep(0.15)  # Should refill ~1.5 tokens
    assert limiter.consume(token) is True


def test_token_bucket_independent_identities():
    """Verify different tokens have independent bucket capacities."""
    limiter = TokenBucketRateLimiter(capacity=1.0, refill_rate=0.0)
    token_a = "token-alpha"
    token_b = "token-beta"

    assert limiter.consume(token_a) is True
    assert limiter.consume(token_a) is False

    assert limiter.consume(token_b) is True
    assert limiter.consume(token_b) is False


@pytest.mark.asyncio
async def test_verify_bearer_token_rate_limiting_exceeded():
    """Verify verify_bearer_token middleware raises HTTP 429 when per-token bucket is exhausted."""
    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url.path = "/api/v1/analysis"

    test_token = "unique-test-bearer-token-429"
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=test_token)

    limiter = TokenBucketRateLimiter(capacity=1.0, refill_rate=0.0)

    with (
        patch(
            "src.api.middleware.get_valid_tokens", return_value={test_token: ["read"]}
        ),
        patch("src.api.middleware.db_auth.is_token_revoked", return_value=False),
        patch("src.api.middleware.get_token_bucket_limiter", return_value=limiter),
    ):
        # First request consumes the single available token
        result = await verify_bearer_token(mock_request, credentials)
        assert result == test_token

        # Second request exceeds token capacity and raises 429
        with pytest.raises(HTTPException) as exc_info:
            await verify_bearer_token(mock_request, credentials)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Rate limit exceeded" in exc_info.value.detail
