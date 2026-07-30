# filepath: redis_cache.py
"""
redis_cache.py
--------------
Redis connection and caching utilities for session state and FAISS results.
Supports scaling across multiple server nodes in Docker/Kubernetes environments.
Now includes highly optimized payload compression using zlib for massive similarity matrices.
"""

import atexit
import json
import logging
import os
import pickle
import threading
import time
import zlib
from enum import Enum
from typing import Any, Optional


class CacheKeyPrefix(str, Enum):
    LOGIN_ATTEMPTS = "login_attempts:"
    UPLOAD_COUNT = "upload_count:"
    SIMILARITY_RESULT = "similarity:"
    DOCUMENT_CACHE = "doc:"
    LEGACY_UPLOADS_PREFIX = "upload_count:"

try:
    import redis
except ImportError:
    redis = None

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

_RedisErr = getattr(redis, "RedisError", Exception)
RedisError = _RedisErr if isinstance(_RedisErr, type) and issubclass(_RedisErr, BaseException) else Exception

_ConnErr = getattr(redis, "ConnectionError", ConnectionError)
RedisConnectionError = _ConnErr if isinstance(_ConnErr, type) and issubclass(_ConnErr, BaseException) else ConnectionError

_TimeoutErr = getattr(redis, "TimeoutError", TimeoutError)
RedisTimeoutError = _TimeoutErr if isinstance(_TimeoutErr, type) and issubclass(_TimeoutErr, BaseException) else TimeoutError



# Redis connection configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
REDIS_TIMEOUT_SECONDS = float(os.getenv("REDIS_TIMEOUT_SECONDS", "2.0"))

# TTL settings (in seconds)
SESSION_TTL = 15 * 60  # 15 minutes for session state
FAISS_INDEX_TTL = 24 * 60 * 60  # 24 hours for FAISS index cache
ANALYSIS_RESULTS_TTL = 2 * 60 * 60  # 2 hours for analysis results
LOGIN_LOCKOUT_TTL = 15 * 60  # 15 minutes for login lockout
UPLOAD_RATE_TTL = 60 * 60  # 1 hour for upload rate limiting
DEFAULT_TTL = 24 * 60 * 60  # 24 hours fallback for keys without explicit TTL


# ============================================================================
# COMPRESSION UTILITIES
# ============================================================================

class PayloadCompressor:
    """
    Handles robust compression and decompression of serialized cache payloads.
    Uses zlib (standard library) to drastically reduce memory usage of large matrices.
    """
    
    # Threshold above which data is compressed (e.g., 512KB)
    COMPRESSION_THRESHOLD_BYTES = 512 * 1024
    
    # Magic header bytes to distinguish compressed vs uncompressed payloads in Redis
    MAGIC_HEADER = b"ZLIB_COMPRESSED_V1::"
    
    @classmethod
    def compress(cls, data: bytes) -> bytes:
        """
        Compresses bytes if they exceed the threshold. Appends magic header.
        
        Args:
            data (bytes): Raw serialized bytes.
            
        Returns:
            bytes: Compressed bytes with header, or original bytes if too small.
        """
        if len(data) < cls.COMPRESSION_THRESHOLD_BYTES:
            return data
            
        try:
            start_time = time.perf_counter()
            compressed_data = zlib.compress(data, level=zlib.Z_BEST_SPEED)
            compression_ratio = len(data) / max(1, len(compressed_data))
            
            logger.debug(
                f"[CacheCompression] Compressed payload from {len(data)}B to {len(compressed_data)}B. "
                f"Ratio: {compression_ratio:.2f}x. Time: {(time.perf_counter()-start_time)*1000:.2f}ms"
            )
            
            return cls.MAGIC_HEADER + compressed_data
        except zlib.error as e:
            logger.error(f"[CacheCompression] zlib compression failed: {e}. Falling back to uncompressed.")
            return data

    @classmethod
    def decompress(cls, data: bytes) -> bytes:
        """
        Decompresses bytes if they contain the magic header.
        
        Args:
            data (bytes): Stored bytes retrieved from cache.
            
        Returns:
            bytes: Decompressed raw bytes.
        """
        if not isinstance(data, bytes):
            return data
            
        if data.startswith(cls.MAGIC_HEADER):
            try:
                start_time = time.perf_counter()
                payload = data[len(cls.MAGIC_HEADER):]
                decompressed_data = zlib.decompress(payload)
                
                logger.debug(
                    f"[CacheCompression] Decompressed payload. "
                    f"Time: {(time.perf_counter()-start_time)*1000:.2f}ms"
                )
                return decompressed_data
            except zlib.error as e:
                logger.error(f"[CacheCompression] zlib decompression failed: {e}. Corrupted payload?")
                raise e
        
        return data


# ============================================================================
# REDIS NAMESPACES
# ============================================================================

class CacheNamespace(str, Enum):
    SESSION = "spd:v1:session"
    FAISS = "spd:v1:faiss"
    ANALYSIS = "spd:v1:analysis"
    LOGIN_ATTEMPTS = "spd:v1:login_attempts"
    UPLOADS = "spd:v1:uploads"

    def build_key(self, *parts: str) -> str:
        """Construct a standardized cache key with namespace prefix."""
        return ":".join([self.value] + list(parts))


# ============================================================================
# MAIN REDIS CACHE MANAGER
# ============================================================================

class RedisCache:
    """Redis cache manager for session state and computational results."""

    _instance: Optional["RedisCache"] = None
    _client: Optional[Any] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "RedisCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._fallback_cache = {}
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_fallback_cache") or self._fallback_cache is None:
            self._fallback_cache = {}
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._connect()

    @property
    def fallback_cache(self) -> dict:
        """Lazily initialize fallback cache dictionary if not present."""
        if not hasattr(self, "_fallback_cache") or self._fallback_cache is None:
            with self._lock:
                if not hasattr(self, "_fallback_cache") or self._fallback_cache is None:
                    self._fallback_cache = {}
        return self._fallback_cache

    def _fallback_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        expire_at = time.time() + ttl if ttl is not None else None
        with self._lock:
            self.fallback_cache[key] = (value, expire_at)
        return True

    def _fallback_get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self.fallback_cache:
                return None
            value, expire_at = self.fallback_cache[key]
            if expire_at is not None and time.time() > expire_at:
                del self.fallback_cache[key]
                return None
            return value

    def _fallback_delete(self, key: str) -> bool:
        with self._lock:
            if key in self.fallback_cache:
                del self.fallback_cache[key]
                return True
        return False

    def _fallback_exists(self, key: str) -> bool:
        return self._fallback_get(key) is not None

    def _fallback_set_json(self, key: str, value: dict, ttl: Optional[int] = None) -> bool:
        serialized = json.dumps(value)
        return self._fallback_set(key, json.loads(serialized), ttl)

    def _fallback_get_json(self, key: str) -> Optional[dict]:
        val = self._fallback_get(key)
        if isinstance(val, dict):
            return val
        return None

    def _fallback_clear_pattern(self, pattern: str) -> int:
        import fnmatch
        with self._lock:
            keys_to_delete = [
                key for key in list(self.fallback_cache.keys())
                if fnmatch.fnmatch(key, pattern)
            ]
            count = 0
            for key in keys_to_delete:
                if key in self.fallback_cache:
                    del self.fallback_cache[key]
                    count += 1
            return count

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
                    socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
                )
            else:
                self._client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=False,
                    socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
                )
            self._client.ping()
            print(f"[RedisCache] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except (
            RedisConnectionError,
            RedisTimeoutError,
            ConnectionRefusedError,
        ) as e:
            print(f"[RedisCache] Redis connection failed: {e}. Running without cache.")
            logger.warning(f"[RedisCache] Redis connection failed: {e}. Running without cache.")
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

    def ping(self) -> tuple[bool, Optional[float]]:
        if self._client is None:
            return False, None
        try:
            start = time.monotonic()
            self._client.ping()
            elapsed = (time.monotonic() - start) * 1000
            return True, round(elapsed, 1)
        except Exception:
            return False, None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in Redis with optional TTL and automatic compression."""
        if self.is_available():
            try:
                # 1. Serialize
                serialized = pickle.dumps(value)
                # 2. Compress large payloads
                processed_bytes = PayloadCompressor.compress(serialized)
                
                # 3. Store
                if ttl:
                    self._client.setex(key, ttl, processed_bytes)
                else:
                    self._client.set(key, processed_bytes)
                return True
            except (
                RedisError,
                RedisConnectionError,
                RedisTimeoutError,
                ConnectionRefusedError,
                ConnectionResetError,
                pickle.PickleError,
            ) as e:
                print(f"[RedisCache] Error setting key {key}: {e}. Falling back to in-memory.")
                logger.error(f"[RedisCache] Error setting key {key}: {e}. Falling back to in-memory.")

        return self._fallback_set(key, value, ttl)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from Redis with automatic decompression."""
        if self.is_available():
            try:
                data = self._client.get(key)
                if data is not None:
                    # 1. Decompress if magic header is present
                    raw_bytes = PayloadCompressor.decompress(data)
                    # 2. Deserialize
                    return pickle.loads(raw_bytes)
            except (
                RedisError,
                RedisConnectionError,
                RedisTimeoutError,
                ConnectionRefusedError,
                ConnectionResetError,
                pickle.PickleError,
                zlib.error,
            ) as e:
                print(f"[RedisCache] Error getting key {key}: {e}. Falling back to in-memory.")
                logger.error(f"[RedisCache] Error getting key {key}: {e}. Falling back to in-memory.")

        return self._fallback_get(key)

    def delete(self, key: str) -> bool:
        redis_deleted = False
        if self.is_available():
            try:
                redis_deleted = bool(self._client.delete(key))
            except Exception as e:
                logger.error(f"[RedisCache] Error deleting key {key}: {e}")

        fallback_deleted = self._fallback_delete(key)
        return redis_deleted or fallback_deleted

    def set_json(self, key: str, value: dict, ttl: Optional[int] = None) -> bool:
        """Store a JSON-serializable dict in Redis with automatic compression."""
        if self.is_available():
            try:
                serialized = json.dumps(value).encode('utf-8')
                processed_bytes = PayloadCompressor.compress(serialized)
                
                if ttl:
                    self._client.setex(key, ttl, processed_bytes)
                else:
                    self._client.set(key, processed_bytes)
                return True
            except Exception as e:
                logger.error(f"[RedisCache] Error setting JSON key {key}: {e}")

        return self._fallback_set_json(key, value, ttl)

    def get_json(self, key: str) -> Optional[dict]:
        """Retrieve a JSON value from Redis with automatic decompression."""
        if self.is_available():
            try:
                data = self._client.get(key)
                if data is not None:
                    raw_bytes = PayloadCompressor.decompress(data)
                    return json.loads(raw_bytes.decode('utf-8'))
            except Exception as e:
                logger.error(f"[RedisCache] Error getting JSON key {key}: {e}")

        return self._fallback_get_json(key)

    def exists(self, key: str) -> bool:
        if self.is_available():
            try:
                if bool(self._client.exists(key)):
                    return True
            except Exception as e:
                logger.error(f"[RedisCache] Error checking key {key}: {e}")

        return self._fallback_exists(key)

    def clear_pattern(self, pattern: str) -> int:
        redis_count = 0
        if self.is_available():
            try:
                keys = self._client.keys(pattern)
                if keys and not isinstance(keys, (list, set, tuple)):
                    keys = None
                if keys:
                    res = self._client.delete(*keys)
                    redis_count = int(res) if isinstance(res, (int, float)) else 0
            except (
                RedisError,
                RedisConnectionError,
                RedisTimeoutError,
                ConnectionRefusedError,
                ConnectionResetError,
                Exception,
            ) as e:
                print(f"[RedisCache] Error clearing pattern {pattern}: {e}. Falling back to in-memory.")
                logger.error(f"[RedisCache] Error clearing pattern {pattern}: {e}. Falling back to in-memory.")

        fallback_count = self._fallback_clear_pattern(pattern)
        return (int(redis_count) if isinstance(redis_count, (int, float)) else 0) + fallback_count


    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                    self._client = None
                except Exception as e:
                    logger.error(f"[RedisCache] Error closing Redis connection: {e}")


# Global cache instance
_cache = RedisCache()


# ============================================================================
# MODULE LEVEL PUBLIC API
# ============================================================================

def get_cache(key: Optional[str] = None):
    if key is not None:
        return _cache.get(key)
    return _cache

def set_cache(key: str, value: Any, expire: Optional[int] = None) -> bool:
    return _cache.set(key, value, ttl=expire)

def delete_cache(key: str) -> bool:
    return _cache.delete(key)

def cache_session_state(session_id: str, key: str, value: Any) -> bool:
    cache_key = CacheNamespace.SESSION.build_key(session_id, key)
    return _cache.set(cache_key, value, SESSION_TTL)

def get_session_state(session_id: str, key: str) -> Optional[Any]:
    cache_key = CacheNamespace.SESSION.build_key(session_id, key)
    return _cache.get(cache_key)

def clear_session(session_id: str) -> bool:
    pattern = CacheNamespace.SESSION.build_key(session_id, "*")
    return _cache.clear_pattern(pattern) > 0

def cache_faiss_index(index_key: str, index_data: bytes) -> bool:
    cache_key = CacheNamespace.FAISS.build_key("index", index_key)
    return _cache.set(cache_key, index_data, FAISS_INDEX_TTL)

def get_faiss_index(index_key: str) -> Optional[bytes]:
    cache_key = CacheNamespace.FAISS.build_key("index", index_key)
    return _cache.get(cache_key)

def cache_analysis_results(analysis_key: str, results: dict) -> bool:
    cache_key = CacheNamespace.ANALYSIS.build_key(analysis_key)
    return _cache.set(cache_key, results, ANALYSIS_RESULTS_TTL)

def get_analysis_results(analysis_key: str) -> Optional[dict]:
    cache_key = CacheNamespace.ANALYSIS.build_key(analysis_key)
    return _cache.get(cache_key)

def increment_login_attempts(identifier: str) -> int:
    cache_key = CacheNamespace.LOGIN_ATTEMPTS.build_key(identifier)
    current = _cache.get(cache_key)
    if current is None:
        current = 0
    current += 1
    _cache.set(cache_key, current, LOGIN_LOCKOUT_TTL)
    return current

def get_login_attempts(identifier: str) -> int:
    cache_key = CacheNamespace.LOGIN_ATTEMPTS.build_key(identifier)
    current = _cache.get(cache_key)
    return current if current is not None else 0

def is_login_locked_out(identifier: str) -> bool:
    return get_login_attempts(identifier) >= 5

def clear_login_attempts(identifier: str) -> bool:
    cache_key = CacheNamespace.LOGIN_ATTEMPTS.build_key(identifier)
    return _cache.delete(cache_key)

def increment_upload_count(username: str) -> int:
    cache_key = CacheNamespace.UPLOADS.build_key(username)
    current = _cache.get(cache_key)
    if current is None:
        current = 0
    current += 1
    _cache.set(cache_key, current, UPLOAD_RATE_TTL)
    return current

def get_upload_count(username: str) -> int:
    cache_key = CacheNamespace.UPLOADS.build_key(username)
    current = _cache.get(cache_key)
    return current if current is not None else 0

def is_upload_rate_limited(username: str) -> bool:
    return get_upload_count(username) >= 100

def _cleanup_redis() -> None:
    if _cache:
        _cache.close()

atexit.register(_cleanup_redis)
