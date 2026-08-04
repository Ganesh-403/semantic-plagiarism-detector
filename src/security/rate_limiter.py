from __future__ import annotations

"""
rate_limiter.py
---------------
Security utility for API rate limiting with standard HTTP response headers.
Tracks request counts per IP/user and returns appropriate headers when limits are exceeded.
"""

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
        prefix: str = "rate_limit"
    ):
        """
        Initialize the rate limiter.

        Args:
            redis_client: Active Redis connection instance.
            limit: Maximum number of requests allowed in the time window.
            window: Time window in seconds.
            block_duration: Duration in seconds to block the client after exceeding the limit.
            prefix: Redis key prefix for rate limiting data.
        """
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

    def check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, str]]:
        """
        Check if a request is allowed under the rate limit and return headers.

        Args:
            identifier: Unique identifier for the client (e.g., IP address or user ID).

        Returns:
            Tuple[bool, Dict[str, str]]:
                - bool: True if the request is allowed, False if rate limited.
                - Dict[str, str]: Dictionary of HTTP headers to include in the response.
        """
        key = self._get_key(identifier)
        block_key = self._get_block_key(identifier)
        current_time = int(time.time())

        # Check if the client is currently blocked
        if self.redis.exists(block_key):
            ttl = self.redis.ttl(block_key)
            retry_after = max(1, ttl)

            headers = {
                "X-RateLimit-Limit": str(self.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(current_time + retry_after),
                "Retry-After": str(retry_after)
            }

            logger.warning(f"Rate limit exceeded for {identifier}. Blocked for {retry_after}s.")
            return False, headers

        # Get current request count
        current_count = self.redis.get(key)

        if current_count is None:
            # First request in the window
            pipe = self.redis.pipeline()
            pipe.setex(key, self.window, 1)
            pipe.execute()
            current_count = 1
        else:
            current_count = int(current_count)

        # Check if limit is exceeded
        if current_count >= self.limit:
            # Block the client
            self.redis.setex(block_key, self.block_duration, "1")

            headers = {
                "X-RateLimit-Limit": str(self.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(current_time + self.block_duration),
                "Retry-After": str(self.block_duration)
            }

            logger.warning(f"Rate limit exceeded for {identifier}. Blocking for {self.block_duration}s.")
            return False, headers

        # Increment the counter
        pipe = self.redis.pipeline()
        pipe.incr(key)
        # Ensure the key expires even if it was created without expiration
        pipe.expire(key, self.window)
        pipe.execute()

        remaining = max(0, self.limit - current_count - 1)
        reset_time = current_time + self.window

        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "Retry-After": str(self.window)  # Fallback retry-after
        }

        return True, headers

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
    window: int = DEFAULT_WINDOW
) -> Tuple[bool, Dict[str, str]]:
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
