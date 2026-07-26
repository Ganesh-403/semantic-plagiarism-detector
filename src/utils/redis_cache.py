"""
redis_cache.py
--------------
Redis connection and caching utilities for session state and FAISS results.
Supports scaling across multiple server nodes in Docker/Kubernetes environments.
"""

import json
import os
import pickle
from typing import Any, Optional

try:
    import redis
except ImportError:
    redis = None
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

RedisError = getattr(redis, "RedisError", Exception)
RedisConnectionError = getattr(redis, "ConnectionError", ConnectionError)
RedisTimeoutError = getattr(redis, "TimeoutError", TimeoutError)


# Redis connection configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

# TTL settings (in seconds)
SESSION_TTL = 15 * 60  # 15 minutes for session state
FAISS_INDEX_TTL = 24 * 60 * 60  # 24 hours for FAISS index cache
ANALYSIS_RESULTS_TTL = 2 * 60 * 60  # 2 hours for analysis results
LOGIN_LOCKOUT_TTL = 15 * 60  # 15 minutes for login lockout
UPLOAD_RATE_TTL = 60 * 60  # 1 hour for upload rate limiting


class RedisCache:
    """Redis cache manager for session state and computational results."""

    _instance: Optional["RedisCache"] = None
    _client: Optional[Any] = None

    def __new__(cls) -> "RedisCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self) -> None:
        """Establish Redis connection with fallback to in-memory if unavailable."""
        if redis is None:
            self._client = None
            return

        try:
            if REDIS_URL:
                self._client = redis.from_url(
                    REDIS_URL,
                    password=REDIS_PASSWORD,
                    decode_responses=False,
                    socket_connect_timeout=5,
                )
            else:
                self._client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=False,
                    socket_connect_timeout=5,
                )
            # Test connection
            self._client.ping()
            print(f"[RedisCache] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except (
            RedisConnectionError,
            RedisTimeoutError,
            ConnectionRefusedError,
        ) as e:
            print(f"[RedisCache] Redis connection failed: {e}. Running without cache.")
            logger.warning(
                f"[RedisCache] Redis connection failed: {e}. Running without cache."
            )
            self._client = None

    def is_available(self) -> bool:
        """Check if Redis is available."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def ping(self) -> tuple[bool, float | None]:
        """Ping Redis and measure round-trip latency.

        Returns:
            Tuple of (connected: bool, latency_ms: float | None).
            latency_ms is None if the connection is unavailable.
        """
        if self._client is None:
            return False, None
        try:
            import time

            start = time.monotonic()
            self._client.ping()
            elapsed = (time.monotonic() - start) * 1000
            return True, round(elapsed, 1)
        except Exception:
            return False, None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in Redis with optional TTL."""
        if not self.is_available():
            return False

        try:
            serialized = pickle.dumps(value)
            if ttl:
                self._client.setex(key, ttl, serialized)
            else:
                self._client.set(key, serialized)
            return True
        except (RedisError, pickle.PickleError, ConnectionRefusedError) as e:
            print(f"[RedisCache] Error setting key {key}: {e}")
            logger.error(f"[RedisCache] Error setting key {key}: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from Redis."""
        if not self.is_available():
            return None

        try:
            data = self._client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except (RedisError, pickle.PickleError, ConnectionRefusedError) as e:
            print(f"[RedisCache] Error getting key {key}: {e}")
            logger.error(f"[RedisCache] Error getting key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self.is_available():
            return False

        try:
            self._client.delete(key)
            return True
        except (RedisError, ConnectionRefusedError) as e:
            print(f"[RedisCache] Error deleting key {key}: {e}")
            logger.error(f"[RedisCache] Error deleting key {key}: {e}")
            return False

    def set_json(self, key: str, value: dict, ttl: Optional[int] = None) -> bool:
        """Store a JSON-serializable dict in Redis."""
        if not self.is_available():
            return False

        try:
            serialized = json.dumps(value)
            if ttl:
                self._client.setex(key, ttl, serialized)
            else:
                self._client.set(key, serialized)
            return True
        except (RedisError, json.JSONDecodeError, ConnectionRefusedError) as e:
            print(f"[RedisCache] Error setting JSON key {key}: {e}")
            logger.error(f"[RedisCache] Error setting JSON key {key}: {e}")
            return False

    def get_json(self, key: str) -> Optional[dict]:
        """Retrieve a JSON value from Redis."""
        if not self.is_available():
            return None

        try:
            data = self._client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except (RedisError, json.JSONDecodeError, ConnectionRefusedError) as e:
            print(f"[RedisCache] Error getting JSON key {key}: {e}")
            logger.error(f"[RedisCache] Error getting JSON key {key}: {e}")
            return None

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        if not self.is_available():
            return False

        try:
            return bool(self._client.exists(key))
        except (RedisError, ConnectionRefusedError) as e:
            print(f"[RedisCache] Error checking key {key}: {e}")
            logger.error(f"[RedisCache] Error checking key {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        if not self.is_available():
            return 0

        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        except (RedisError, ConnectionRefusedError) as e:
            print(f"[RedisCache] Error clearing pattern {pattern}: {e}")
            logger.error(f"[RedisCache] Error clearing pattern {pattern}: {e}")
            return 0


    def close(self) -> None:
        """Explicitly close the Redis connection."""
        if self._client is not None:
            try:
                self._client.close()
                self._client = None
            except Exception as e:
                print(f"[RedisCache] Error closing Redis connection: {e}")
                logger.error(f"[RedisCache] Error closing Redis connection: {e}")


# Global cache instance
_cache = RedisCache()


def get_cache() -> RedisCache:
    """Get the global Redis cache instance."""
    return _cache


def cache_session_state(session_id: str, key: str, value: Any) -> bool:
    """Cache session state data with TTL."""
    cache_key = f"session:{session_id}:{key}"
    return _cache.set(cache_key, value, SESSION_TTL)


def get_session_state(session_id: str, key: str) -> Optional[Any]:
    """Retrieve session state data from cache."""
    cache_key = f"session:{session_id}:{key}"
    return _cache.get(cache_key)


def clear_session(session_id: str) -> bool:
    """Clear all session data for a given session ID."""
    pattern = f"session:{session_id}:*"
    return _cache.clear_pattern(pattern) > 0


def cache_faiss_index(index_key: str, index_data: bytes) -> bool:
    """Cache FAISS index binary data."""
    cache_key = f"faiss:index:{index_key}"
    return _cache.set(cache_key, index_data, FAISS_INDEX_TTL)


def get_faiss_index(index_key: str) -> Optional[bytes]:
    """Retrieve FAISS index binary data from cache."""
    cache_key = f"faiss:index:{index_key}"
    return _cache.get(cache_key)


def cache_analysis_results(analysis_key: str, results: dict) -> bool:
    """Cache analysis results (embeddings, similarity matrices, etc.)."""
    cache_key = f"analysis:{analysis_key}"
    return _cache.set(cache_key, results, ANALYSIS_RESULTS_TTL)


def get_analysis_results(analysis_key: str) -> Optional[dict]:
    """Retrieve analysis results from cache."""
    cache_key = f"analysis:{analysis_key}"
    return _cache.get(cache_key)


def increment_login_attempts(identifier: str) -> int:
    """Increment failed login attempt counter for a username/IP."""
    cache_key = f"login_attempts:{identifier}"
    current = _cache.get(cache_key)
    if current is None:
        current = 0
    current += 1
    _cache.set(cache_key, current, LOGIN_LOCKOUT_TTL)
    return current


def get_login_attempts(identifier: str) -> int:
    """Get current failed login attempt count for a username/IP."""
    cache_key = f"login_attempts:{identifier}"
    current = _cache.get(cache_key)
    return current if current is not None else 0


def is_login_locked_out(identifier: str) -> bool:
    """Check if a username/IP is locked out due to too many failed attempts."""
    return get_login_attempts(identifier) >= 5


def clear_login_attempts(identifier: str) -> bool:
    """Clear failed login attempt counter after successful login."""
    cache_key = f"login_attempts:{identifier}"
    return _cache.delete(cache_key)


def increment_upload_count(username: str) -> int:
    """Increment upload counter for a user per hour."""
    cache_key = f"uploads:{username}"
    current = _cache.get(cache_key)
    if current is None:
        current = 0
    current += 1
    _cache.set(cache_key, current, UPLOAD_RATE_TTL)
    return current


def get_upload_count(username: str) -> int:
    """Get current upload count for a user in the current hour window."""
    cache_key = f"uploads:{username}"
    current = _cache.get(cache_key)
    return current if current is not None else 0


def is_upload_rate_limited(username: str) -> bool:
    """Check if a user has exceeded the upload rate limit (100 uploads/hour)."""
    return get_upload_count(username) >= 100


import atexit

def _cleanup_redis() -> None:
    """Close the global Redis connection when the process terminates."""
    if _cache:
        _cache.close()

atexit.register(_cleanup_redis)
