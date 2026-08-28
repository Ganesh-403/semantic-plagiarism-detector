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
import urllib.parse
import zlib
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

# CacheKeyPrefix has been consolidated into CacheNamespace below

try:
    import redis
except ImportError:
    redis = None


try:
    from src.core.app_config import REDIS_CACHE_TTL
except ImportError:
    REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))

try:
    from src.core.app_config import REDIS_PORT
except ImportError:
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
try:
    from src.version import APP_VERSION
except ImportError:
    APP_VERSION = "1.0.0"

logger = logging.getLogger(__name__)


class DummyRedisError(Exception):
    pass


class DummyRedisConnectionError(DummyRedisError):
    pass


class DummyRedisTimeoutError(DummyRedisError):
    pass


_RedisErr = getattr(redis, "RedisError", DummyRedisError)
RedisError = (
    _RedisErr
    if isinstance(_RedisErr, type) and issubclass(_RedisErr, BaseException)
    else DummyRedisError
)

_ConnErr = getattr(redis, "ConnectionError", DummyRedisConnectionError)
RedisConnectionError = (
    _ConnErr
    if isinstance(_ConnErr, type) and issubclass(_ConnErr, BaseException)
    else DummyRedisConnectionError
)

_TimeoutErr = getattr(redis, "TimeoutError", DummyRedisTimeoutError)
RedisTimeoutError = (
    _TimeoutErr
    if isinstance(_TimeoutErr, type) and issubclass(_TimeoutErr, BaseException)
    else DummyRedisTimeoutError
)


# Redis connection configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
try:
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
except ValueError:
    logger.warning(
        f"Invalid REDIS_DB configuration '{os.getenv('REDIS_DB')}'. Defaulting to 0."
    )
    REDIS_DB = 0
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

if REDIS_PASSWORD:
    encoded_password = urllib.parse.quote_plus(REDIS_PASSWORD)
    REDIS_URL = os.getenv(
        "REDIS_URL",
        f"redis://:{encoded_password}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    )
else:
    REDIS_URL = os.getenv(
        "REDIS_URL",
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    )
REDIS_TIMEOUT_SECONDS = float(os.getenv("REDIS_TIMEOUT_SECONDS", "2.0"))

# TTL settings (in seconds) - Configurable via environment variables (Issue #2323)
# Defaults are preserved for backward compatibility when env vars are not set
SESSION_TTL = int(os.getenv("SESSION_TTL", str(15 * 60)))  # 15 minutes for session state
FAISS_INDEX_TTL = int(os.getenv("FAISS_INDEX_TTL", str(24 * 60 * 60)))  # 24 hours for FAISS index cache
ANALYSIS_RESULTS_TTL = int(os.getenv("ANALYSIS_RESULTS_TTL", str(2 * 60 * 60)))  # 2 hours for analysis results
LOGIN_LOCKOUT_TTL = int(os.getenv("LOGIN_LOCKOUT_TTL", str(15 * 60)))  # 15 minutes for login lockout
UPLOAD_RATE_TTL = int(os.getenv("UPLOAD_RATE_TTL", str(60 * 60)))  # 1 hour for upload rate limiting
BADGE_TTL = int(os.getenv("BADGE_TTL", str(24 * 60 * 60)))  # 24 hours for badge buffer cache
SCAN_JOBS_TTL = int(os.getenv("SCAN_JOBS_TTL", str(24 * 60 * 60)))  # 24 hours for scan jobs
DEFAULT_TTL = int(os.getenv("DEFAULT_TTL", str(24 * 60 * 60)))  # 24 hours fallback for keys without explicit TTL


# ============================================================================
# COMPRESSION UTILITIES
# ============================================================================


class PayloadCompressor:
    """
    Handles robust compression and decompression of serialized cache payloads.
    Uses zlib to drastically reduce memory usage of large matrices.
    """

    COMPRESSION_THRESHOLD_BYTES: int = 64 * 1024
    _raw_threshold = os.getenv("REDIS_COMPRESSION_THRESHOLD", "").strip()
    if _raw_threshold:
        try:
            COMPRESSION_THRESHOLD_BYTES = int(_raw_threshold)
        except ValueError:
            pass

    COMPRESSION_LEVEL: int = zlib.Z_BEST_SPEED
    _raw_level = os.getenv("REDIS_COMPRESSION_LEVEL", "").strip()
    if _raw_level:
        try:
            COMPRESSION_LEVEL = int(_raw_level)
        except ValueError:
            _consts = {
                "Z_BEST_SPEED": zlib.Z_BEST_SPEED,
                "Z_BEST_COMPRESSION": zlib.Z_BEST_COMPRESSION,
                "Z_DEFAULT_COMPRESSION": zlib.Z_DEFAULT_COMPRESSION,
                "Z_NO_COMPRESSION": zlib.Z_NO_COMPRESSION,
            }
            COMPRESSION_LEVEL = _consts.get(_raw_level.upper(), zlib.Z_BEST_SPEED)

    MAGIC_HEADER = b"ZLIB_COMPRESSED_V1::"

    @classmethod
    def get_threshold(cls) -> int:
        return cls.COMPRESSION_THRESHOLD_BYTES

    @classmethod
    def compress(cls, data: bytes) -> bytes:
        if len(data) < cls.get_threshold():
            return data

        try:
            start_time = time.perf_counter()
            compressed_data = zlib.compress(data, level=cls.COMPRESSION_LEVEL)
            compression_ratio = len(data) / max(1, len(compressed_data))

            logger.debug(
                f"[CacheCompression] Compressed payload from {len(data)}B to {len(compressed_data)}B. "
                f"Ratio: {compression_ratio:.2f}x. Time: {(time.perf_counter()-start_time)*1000:.2f}ms"
            )

            return cls.MAGIC_HEADER + compressed_data
        except zlib.error as e:
            logger.error(
                f"[CacheCompression] zlib compression failed: {e}. Falling back to uncompressed."
            )
            return data

    @classmethod
    def decompress(cls, data: bytes) -> bytes | None:
        if not isinstance(data, bytes):
            return data

        if data.startswith(cls.MAGIC_HEADER):
            try:
                start_time = time.perf_counter()
                payload = data[len(cls.MAGIC_HEADER) :]
                decompressed_data = zlib.decompress(payload)

                logger.debug(
                    f"[CacheCompression] Decompressed payload. "
                    f"Time: {(time.perf_counter()-start_time)*1000:.2f}ms"
                )
                return decompressed_data
                
            except zlib.error as e:
                logger.critical(
                    "[CacheCompression] CRITICAL: zlib decompression failed due to "
                    "corrupted payload. Treating as cache miss. Error: %s",
                    e,
                    exc_info=True
                )
                return None
                
            except Exception as e:
                logger.critical(
                    "[CacheCompression] CRITICAL: Unexpected error during decompression: %s",
                    e,
                    exc_info=True
                )
                return None

        return data


# ============================================================================
# REDIS NAMESPACES
# ============================================================================


def normalize_cache_key_path(p: Any) -> str:
    """Normalize path strings for cross-platform Redis cache keys (Issue #2939, #3028).

    Uses pathlib.Path(p).as_posix() explicitly whenever creating cache keys based on file paths
    to convert backslashes (\) on Windows to POSIX forward slashes (/) for cross-platform
    cache key compatibility.
    """
    if p is None:
        return ""
    if isinstance(p, Path):
        return p.as_posix()
    p_str = str(p)
    if not p_str:
        return ""
    return Path(p_str).as_posix()


class CacheNamespace(str, Enum):
    SESSION = "spd:v1:session"
    FAISS = "spd:v1:faiss"
    ANALYSIS = "spd:v1:analysis"
    LOGIN_ATTEMPTS = "spd:v1:login_attempts"
    UPLOADS = "spd:v1:uploads"
    BADGES = "spd:v1:badges"
    SCAN_JOBS = "spd:v1:scan_jobs"
    CLUSTERING_JOBS = "spd:v1:clustering_jobs"

    def build_key(self, *parts: Any) -> str:
        """Build a normalized Redis cache key appending APP_VERSION and using pathlib.Path(p).as_posix() for path components."""
        normalized_parts = [normalize_cache_key_path(p) for p in parts]
        key_parts = [self.value, APP_VERSION] + [p for p in normalized_parts if p]
        return ":".join(key_parts)


CacheKeyPrefix = CacheNamespace


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
                    instance = super().__new__(cls)
                    instance._fallback_cache = {}
                    instance._hits = 0
                    instance._misses = 0
                    instance._client = None
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_fallback_cache") or self._fallback_cache is None:
            self._fallback_cache = {}
        if not getattr(self, "_initialized", False):
            with self._lock:
                if not getattr(self, "_initialized", False):
                    self._connect()
                    self._initialized = True

    @classmethod
    def get_instance(cls) -> "RedisCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def fallback_cache(self) -> dict:
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

    def _fallback_set_json(
        self, key: str, value: dict, ttl: Optional[int] = None
    ) -> bool:
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
                key
                for key in list(self.fallback_cache.keys())
                if fnmatch.fnmatch(key, pattern)
            ]
            count = 0
            for key in keys_to_delete:
                if key in self.fallback_cache:
                    del self.fallback_cache[key]
                    count += 1
            return count

    def _connect(self) -> None:
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
            logger.info(f"[RedisCache] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except (
            RedisConnectionError,
            RedisTimeoutError,
            ConnectionRefusedError,
        ) as e:
            logger.warning(
                f"[RedisCache] Redis connection failed: {e}. Running without cache."
            )
            self._client = None

    def is_available(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def scan_keys(self, match: str) -> list[str]:
        if not self.is_available():
            return []
        try:
            raw_keys = list(self._client.scan_iter(match=match))
            return [
                k.decode("utf-8") if isinstance(k, bytes) else k for k in raw_keys
            ]
        except Exception as e:
            logger.error(f"[RedisCache] Error scanning keys for pattern {match}: {e}")
            return []

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

    def get_stats(self) -> dict[str, Any]:
        total_requests = self._hits + self._misses
        hit_ratio = (self._hits / total_requests) if total_requests > 0 else 0.0

        total_items = 0
        if self._client is not None and self.is_available():
            try:
                total_items = self._client.dbsize()
            except Exception:
                total_items = len(self.fallback_cache)
        else:
            total_items = len(self.fallback_cache)

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": hit_ratio,
            "total_items": total_items,
        }

    def get_hit_rate(self) -> float:
        with self._lock:
            hits, misses = self._hits, self._misses
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100

    def _inc_hits(self) -> None:
        with self._lock:
            self._hits += 1
        try:
            from src.core.metrics import cache_hits_total
            cache_hits_total.labels(cache_type="redis").inc()
        except Exception:
            pass

    def _inc_misses(self) -> None:
        with self._lock:
            self._misses += 1
        try:
            from src.core.metrics import cache_misses_total
            cache_misses_total.labels(cache_type="redis").inc()
        except Exception:
            pass

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if self.is_available():
            try:
                serialized = pickle.dumps(value)
                processed_bytes = PayloadCompressor.compress(serialized)

                if ttl:
                    self._client.setex(key, ttl, processed_bytes)
                else:
                    self._client.set(key, processed_bytes)
                return True
            except Exception as e:
                logger.error(
                    f"[RedisCache] Error setting key {key}: {e}. Falling back to in-memory."
                )

        return self._fallback_set(key, value, ttl)

    def get(self, key: str) -> Optional[Any]:
        if self.is_available():
            try:
                data = self._client.get(key)
                if data is not None:
                    decompressed = PayloadCompressor.decompress(data)
                    
                    if decompressed is None:
                        logger.warning(
                            "[RedisCache] Corrupted payload detected for key '%s'. Deleting.",
                            key
                        )
                        try:
                            self._client.delete(key)
                        except Exception:
                            pass
                    else:
                        self._inc_hits()
                        return pickle.loads(decompressed)
                        
            except Exception as e:
                logger.error(f"[RedisCache] Error getting key {key}: {e}. Falling back.")

        val = self._fallback_get(key)
        if val is not None:
            self._inc_hits()
            return val

        self._inc_misses()
        return None

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
        if self.is_available():
            try:
                serialized = json.dumps(value).encode("utf-8")
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
        if self.is_available():
            try:
                data = self._client.get(key)
                if data is not None:
                    decompressed = PayloadCompressor.decompress(data)
                    
                    if decompressed is None:
                        logger.warning(
                            "[RedisCache] Corrupted JSON payload for key '%s'. Deleting.",
                            key
                        )
                        try:
                            self._client.delete(key)
                        except Exception:
                            pass
                    else:
                        self._inc_hits()
                        return json.loads(decompressed.decode('utf-8'))
                        
            except Exception as e:
                logger.error(f"[RedisCache] Error getting JSON key {key}: {e}.")

        val = self._fallback_get_json(key)
        if val is not None:
            self._inc_hits()
            return val

        self._inc_misses()
        return None

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
                if hasattr(self._client, "scan_iter"):
                    keys = list(self._client.scan_iter(match=pattern, count=1000))
                else:
                    keys = self._client.keys(pattern)

                if keys:
                    pipeline = self._client.pipeline()
                    chunk_size = 1000
                    for i in range(0, len(keys), chunk_size):
                        chunk = keys[i : i + chunk_size]
                        pipeline.delete(*chunk)
                    results = pipeline.execute()
                    redis_count = sum(
                        r for r in results if isinstance(r, (int, float))
                    )
            except Exception as e:
                logger.error(
                    f"[RedisCache] Error clearing pattern {pattern}: {e}. Falling back to in-memory."
                )

        fallback_count = self._fallback_clear_pattern(pattern)
        return (
            int(redis_count) if isinstance(redis_count, (int, float)) else 0
        ) + fallback_count

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


def cache_badge(
    badge_type: str,
    identifier: str,
    date: str,
    data: bytes,
    ttl: Optional[int] = None,
) -> bool:
    """Cache generated badge bytes (PNG/PDF) in Redis for 24 hours (Issue #2941)."""
    cache_key = CacheNamespace.BADGES.build_key(badge_type.lower(), identifier, date)
    return _cache.set(cache_key, data, ttl or BADGE_TTL)


def get_cached_badge(
    badge_type: str,
    identifier: str,
    date: str,
) -> Optional[bytes]:
    """Retrieve cached badge bytes (PNG/PDF) from Redis (Issue #2941)."""
    cache_key = CacheNamespace.BADGES.build_key(badge_type.lower(), identifier, date)
    return _cache.get(cache_key)


def cache_scan_job(
    job_id: str,
    data: dict,
    ttl: Optional[int] = None,
) -> bool:
    """Store scan job status and results in Redis under spd:v1:scan_jobs:{job_id} with 24-hour TTL (Issue #3222)."""
    cache_key = CacheNamespace.SCAN_JOBS.build_key(job_id)
    return _cache.set_json(cache_key, data, ttl or SCAN_JOBS_TTL)


def get_scan_job(job_id: str) -> Optional[dict]:
    """Retrieve scan job status and results from Redis (Issue #3222)."""
    cache_key = CacheNamespace.SCAN_JOBS.build_key(job_id)
    return _cache.get_json(cache_key)


def delete_scan_job(job_id: str) -> bool:
    """Delete scan job from Redis (Issue #3222)."""
    cache_key = CacheNamespace.SCAN_JOBS.build_key(job_id)
    return _cache.delete(cache_key)


def _cleanup_redis() -> None:
    if _cache:
        _cache.close()


atexit.register(_cleanup_redis)


def store_large_data(key: str | Path, data: Any, ttl: int = 1800) -> None:
    """Store large data in Redis with compression and normalized POSIX key paths."""
    key_str = normalize_cache_key_path(key)
    try:
        cache = get_cache()
        compressed = zlib.compress(pickle.dumps(data))
        
        if cache.is_available():
            cache._client.setex(f"spd:v1:large:{key_str}", ttl, compressed)
        else:
            cache.fallback_cache[f"spd:v1:large:{key_str}"] = {
                "data": compressed,
                "expiry": time.time() + ttl
            }
        logger.debug(f"Stored large data for key: {key_str} ({len(compressed)} bytes compressed)")
    except Exception as e:
        logger.error(f"Failed to store large data for key {key_str}: {e}")


def get_large_data(key: str | Path) -> Optional[Any]:
    """Retrieve large data from Redis with decompression and normalized POSIX key paths."""
    key_str = normalize_cache_key_path(key)
    try:
        cache = get_cache()
        data = None
        
        if cache.is_available():
            data = cache._client.get(f"spd:v1:large:{key_str}")
        else:
            entry = cache.fallback_cache.get(f"spd:v1:large:{key_str}")
            if entry and entry.get("expiry", 0) > time.time():
                data = entry["data"]
            elif entry:
                del cache.fallback_cache[f"spd:v1:large:{key_str}"]
        
        if data:
            return pickle.loads(zlib.decompress(data))
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve large data for key {key}: {e}")
        return None


def clear_large_data(key: str | Path) -> None:
    """Clear large data from cache using normalized POSIX key paths (Issue #3028)."""
    key_str = normalize_cache_key_path(key)
    try:
        cache = get_cache()
        if cache.is_available():
            cache._client.delete(f"spd:v1:large:{key_str}")
        else:
            cache.fallback_cache.pop(f"spd:v1:large:{key_str}", None)
        logger.debug(f"Cleared large data for key: {key_str}")
    except Exception as e:
        logger.error(f"Failed to clear large data for key {key_str}: {e}")


def clear_all_large_data(session_id: str | Path) -> None:
    """Clear all large data for a session using pipelined deletion and normalized POSIX path (Issue #3028)."""
    sid_str = normalize_cache_key_path(session_id)
    try:
        cache = get_cache()
        patterns = [f"spd:v1:large:{sid_str}:*", f"spd:v1:large:{sid_str}/*"]

        if cache.is_available():
            keys = []
            for pattern in patterns:
                if hasattr(cache._client, "scan_iter"):
                    keys.extend(list(cache._client.scan_iter(match=pattern, count=1000)))
                else:
                    keys.extend(cache._client.keys(pattern))
            keys = list(set(keys))
            if keys:
                pipeline = cache._client.pipeline()
                chunk_size = 1000
                for i in range(0, len(keys), chunk_size):
                    chunk = keys[i : i + chunk_size]
                    pipeline.delete(*chunk)
                pipeline.execute()
        else:
            prefixes = (f"spd:v1:large:{sid_str}:", f"spd:v1:large:{sid_str}/")
            keys_to_remove = [
                k for k in cache.fallback_cache.keys()
                if any(k.startswith(p) for p in prefixes)
            ]
            for key in keys_to_remove:
                del cache.fallback_cache[key]
        logger.debug(f"Cleared all large data for session: {sid_str}")
    except Exception as e:
        logger.error(f"Failed to clear all large data for session {sid_str}: {e}")
