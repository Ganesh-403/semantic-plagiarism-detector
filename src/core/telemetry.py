import logging

from src.db.auth import (get_user_count, get_user_preferences,
                         update_user_preferences)
from src.db.corpus_db import get_all_documents
from src.utils.redis_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

class TelemetryService:
    """
    A background metrics and telemetry aggregator service.
    Caches expensive aggregate queries (e.g., active user counts, document counts)
    in Redis to prevent dashboard loading from slamming the primary database.
    """
    
    CACHE_KEY_USER_COUNT = "telemetry:active_user_count"
    CACHE_KEY_DOC_COUNT = "telemetry:total_document_count"
    CACHE_TTL_SECONDS = 300  # 5 minutes
    
    @classmethod
    def get_active_user_count(cls) -> int:
        """
        Retrieves the total system user count. Uses Redis caching for performance.
        Falls back to direct DB lookup on cache miss.
        """
        try:
            # 1. Attempt Cache Hit
            cached_val = get_cache(cls.CACHE_KEY_USER_COUNT)
            if cached_val is not None:
                return int(cached_val)
        except Exception as e:
            logger.warning(f"Telemetry cache miss/error: {e}")
            
        # 2. Database Lookup
        try:
            count = get_user_count()
        except Exception as e:
            logger.error(f"Failed to aggregate user count: {e}")
            return 0
            
        # 3. Populate Cache
        try:
            set_cache(cls.CACHE_KEY_USER_COUNT, str(count), expire=cls.CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to populate telemetry cache: {e}")
            
        return count

    @classmethod
    def get_document_count(cls) -> int:
        """
        Retrieves the total system document count. Uses Redis caching for performance.
        Falls back to direct DB lookup on cache miss.
        """
        try:
            # 1. Attempt Cache Hit
            cached_val = get_cache(cls.CACHE_KEY_DOC_COUNT)
            if cached_val is not None:
                return int(cached_val)
        except Exception as e:
            logger.warning(f"Telemetry doc cache miss/error: {e}")
            
        # 2. Database Lookup
        try:
            count = len(get_all_documents())
        except Exception as e:
            logger.error(f"Failed to aggregate document count: {e}")
            return 0
            
        # 3. Populate Cache
        try:
            set_cache(cls.CACHE_KEY_DOC_COUNT, str(count), expire=cls.CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to populate doc telemetry cache: {e}")
            
        return count

    @classmethod
    def clear_telemetry_data(cls) -> None:
        """Flush all cached telemetry metrics from Redis.

        This clears the telemetry cache keys so the next read will
        perform a fresh lookup from the database. Safe to call even
        when Redis is unavailable.
        """
        from src.utils.redis_cache import get_cache

        cache = get_cache()
        for key in (cls.CACHE_KEY_USER_COUNT, cls.CACHE_KEY_DOC_COUNT):
            try:
                cache.delete(key)
            except Exception as e:
                logger.warning("Failed to clear telemetry cache key '%s': %s", key, e)

        logger.info("Telemetry cache cleared successfully.")

    @classmethod
    def force_refresh_metrics(cls) -> None:
        """
        Forces a recalculation of all telemetry metrics and updates the cache.
        Intended to be called by background cron jobs or upon manual admin request.
        """
        try:
            u_count = get_user_count()
            set_cache(cls.CACHE_KEY_USER_COUNT, str(u_count), expire=cls.CACHE_TTL_SECONDS)
            
            d_count = len(get_all_documents())
            set_cache(cls.CACHE_KEY_DOC_COUNT, str(d_count), expire=cls.CACHE_TTL_SECONDS)
            
            logger.info("Telemetry metrics force-refreshed successfully.")
        except Exception as e:
            logger.error(f"Force refresh of telemetry failed: {e}")

    @classmethod
    def is_sharing_enabled(cls, username: str) -> bool:
        """
        Returns whether the given user has opted in to sharing anonymous
        usage data. Defaults to True (opted in) if no preference has been
        saved yet.
        """
        try:
            prefs = get_user_preferences(username)
            return bool(prefs.get("telemetry_opt_in", True))
        except Exception as e:
            logger.warning(f"Failed to read telemetry preference for '{username}': {e}")
            return True

    @classmethod
    def set_sharing_preference(cls, username: str, enabled: bool) -> None:
        """
        Updates whether the given user has opted in to sharing anonymous
        usage data. Stored alongside the user's other preferences (e.g. theme,
        threshold) so it persists across sessions.
        """
        try:
            prefs = get_user_preferences(username)
            prefs["telemetry_opt_in"] = bool(enabled)
            update_user_preferences(username, prefs)
        except Exception as e:
            logger.warning(f"Failed to save telemetry preference for '{username}': {e}")