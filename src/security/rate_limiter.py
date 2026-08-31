"""
rate_limiter.py
---------------
Security utility for API rate limiting with standard HTTP response headers.
Tracks request counts per IP/user and returns appropriate headers when limits are exceeded.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import redis

logger = logging.getLogger(__name__)

# Default rate limiting configuration
DEFAULT_LIMIT = 100  # Maximum requests
DEFAULT_WINDOW = 60  # Time window in seconds
DEFAULT_BLOCK_DURATION = 300  # Lock duration in seconds when limit is exceeded


class RateLimiter:
    """
    A Redis-backed rate limiter that tracks request counts and enforces limits.
    Returns standard HTTP rate-limiting headers for API responses.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        limit: int = DEFAULT_LIMIT,
        window: int = DEFAULT_WINDOW,
        block_duration: int = DEFAULT_BLOCK_DURATION,
        prefix: str = "rate_limit",
    ):
        """
        Initialize the rate limiter.

        Args:
            redis_client: Active Redis connection instance.
            limit: Maximum number of requests allowed in the time window.
            window: Time window in seconds.
            block_duration: Duration in seconds to block the client after exceeding the limit.
            prefix: Redis key prefix for rate limiting data.

        Raises:
            ValueError: If limit, window, or block_duration is not positive.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if window < 1:
            raise ValueError(f"window must be >= 1 second, got {window}")
        if block_duration < 1:
            raise ValueError(
                f"block_duration must be >= 1 second, got {block_duration}"
            )

        self.redis = redis_client
        self.limit = limit
        self.window = window
        self.block_duration = block_duration
        self.prefix = prefix

    def _get_key(self, identifier: str) -> str:
        """Generate the Redis key for a specific client identifier."""
        return f"{self.prefix}:{identifier}"

    def _get_block_key(self, identifier: str) -> str:
        """Generate the Redis key for a blocked client."""
        return f"{self.prefix}:blocked:{identifier}"

    def _build_headers(self, remaining: int, retry_after: int) -> dict[str, str]:
        """Assemble the standard rate-limit response headers.

        Args:
            remaining: Requests still available in the current window.
            retry_after: Seconds until the client may retry / the window resets.

        Returns:
            Dictionary of HTTP header names to string values.
        """
        retry_after = max(1, int(retry_after))

        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, int(remaining))),
            "X-RateLimit-Reset": str(int(time.time()) + retry_after),
            "Retry-After": str(retry_after),
        }

    def check_rate_limit(self, identifier: str) -> tuple[bool, dict[str, str]]:
        """
        Check if a request is allowed under the rate limit and return headers.

        The counter is incremented atomically and the returned count is the
        authoritative value, so two concurrent callers can never both observe
        the same pre-increment value and both decide they are under the limit.

        Args:
            identifier: Unique identifier for the client (e.g., IP address or user ID).

        Returns:
            Tuple[bool, Dict[str, str]]:
                - bool: True if the request is allowed, False if rate limited.
                - Dict[str, str]: Dictionary of HTTP headers to include in the response.
        """
        key = self._get_key(identifier)
        block_key = self._get_block_key(identifier)

        # Check if the client is currently blocked
        if self.redis.exists(block_key):
            ttl = self.redis.ttl(block_key)
            retry_after = max(1, int(ttl))

            logger.warning(
                f"Rate limit exceeded for {identifier}. Blocked for {retry_after}s."
            )
            return False, self._build_headers(remaining=0, retry_after=retry_after)

        # Increment first, then read the TTL, in a single round trip. INCR
        # creates the key at 1 when it does not exist, so there is no separate
        # "first request" branch that could double-count.
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        results = pipe.execute()

        current_count = int(results[0])
        ttl = int(results[1])

        # A brand new key (count == 1) has no expiry yet. A pre-existing key
        # reporting -1 lost its TTL somehow; re-arming it stops the counter
        # from living forever and locking the client out permanently.
        if current_count == 1 or ttl < 0:
            self.redis.expire(key, self.window)
            ttl = self.window

        # current_count is the number of requests served *including* this one,
        # so the request that lands exactly on `limit` is still allowed and the
        # next one is not. This serves exactly `limit` requests per window.
        if current_count > self.limit:
            self.redis.setex(block_key, self.block_duration, "1")

            logger.warning(
                f"Rate limit exceeded for {identifier}. Blocking for {self.block_duration}s."
            )
            return False, self._build_headers(
                remaining=0, retry_after=self.block_duration
            )

        remaining = self.limit - current_count
        return True, self._build_headers(remaining=remaining, retry_after=ttl)

    def reset_limit(self, identifier: str) -> bool:
        """
        Reset the rate limit for a specific client.

        Args:
            identifier: Unique identifier for the client.

        Returns:
            bool: True if reset was successful, False otherwise.
        """
        key = self._get_key(identifier)
        block_key = self._get_block_key(identifier)

        pipe = self.redis.pipeline()
        pipe.delete(key)
        pipe.delete(block_key)
        pipe.execute()

        logger.info(f"Rate limit reset for {identifier}")
        return True


def get_rate_limit_headers(
    redis_client: redis.Redis,
    identifier: str,
    limit: int = DEFAULT_LIMIT,
    window: int = DEFAULT_WINDOW,
) -> tuple[bool, dict[str, str]]:
    """
    Convenience function to check rate limits and get headers.

    Args:
        redis_client: Active Redis connection instance.
        identifier: Unique identifier for the client.
        limit: Maximum number of requests allowed.
        window: Time window in seconds.

    Returns:
        Tuple[bool, Dict[str, str]]: (is_allowed, headers_dict)
    """
    limiter = RateLimiter(redis_client, limit=limit, window=window)
    return limiter.check_rate_limit(identifier)


import os
import threading
from typing import Optional


class TokenBucketRateLimiter:
    """Token-Bucket rate limiter for per-token API rate limiting (Issue #2921).

    Tracks token buckets per API bearer token string (credentials.credentials)
    with configurable capacity and refill rate to prevent token abuse.
    """

    def __init__(
        self,
        capacity: float = 60.0,
        refill_rate: float = 1.0,
    ) -> None:
        """Initialize TokenBucketRateLimiter.

        Args:
            capacity: Maximum token capacity for a bucket.
            refill_rate: Tokens added per second.
        """
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._buckets: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def consume(self, identifier: str, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens for the given identifier (credentials.credentials).

        Args:
            identifier: Token string or client identifier.
            tokens: Number of tokens to consume (default 1.0).

        Returns:
            bool: True if request is allowed, False if bucket has insufficient tokens.
        """
        if not identifier:
            return True

        now = time.time()
        with self._lock:
            bucket = self._buckets.get(identifier)
            if bucket is None:
                bucket = {"tokens": self.capacity, "last_refill": now}
                self._buckets[identifier] = bucket

            elapsed = max(0.0, now - bucket["last_refill"])
            bucket["last_refill"] = now
            bucket["tokens"] = min(
                self.capacity,
                bucket["tokens"] + elapsed * self.refill_rate,
            )

            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return True
            return False

    def reset(self, identifier: Optional[str] = None) -> None:
        """Reset rate limit buckets (or single bucket if identifier is provided)."""
        with self._lock:
            if identifier:
                self._buckets.pop(identifier, None)
            else:
                self._buckets.clear()


_TOKEN_BUCKET_LIMITER: Optional[TokenBucketRateLimiter] = None
_LIMITER_LOCK = threading.Lock()


def get_token_bucket_limiter(
    capacity: float = 60.0,
    refill_rate: float = 1.0,
) -> TokenBucketRateLimiter:
    """Retrieve or initialize the global TokenBucketRateLimiter instance."""
    global _TOKEN_BUCKET_LIMITER
    if _TOKEN_BUCKET_LIMITER is None:
        with _LIMITER_LOCK:
            if _TOKEN_BUCKET_LIMITER is None:
                cap_env = os.getenv("TOKEN_BUCKET_CAPACITY")
                refill_env = os.getenv("TOKEN_BUCKET_REFILL_RATE")
                cap_val = float(cap_env) if cap_env else capacity
                refill_val = float(refill_env) if refill_env else refill_rate
                _TOKEN_BUCKET_LIMITER = TokenBucketRateLimiter(
                    capacity=cap_val,
                    refill_rate=refill_val,
                )
    return _TOKEN_BUCKET_LIMITER
