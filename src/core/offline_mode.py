# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Offline Mode for Privacy-Sensitive Deployments.

Provides functionality to run the plagiarism detection system without
external dependencies like Redis, external APIs, and online model downloads.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field  # noqa: F401
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401

logger = logging.getLogger(__name__)

# ============================================================================
# OFFLINE MODE CONFIGURATION
# ============================================================================


@dataclass
class OfflineConfig:
    """Configuration for offline mode."""

    enabled: bool = False
    use_local_cache: bool = True
    cache_dir: str = ".cache/offline"
    model_cache_dir: str = ".cache/models"
    preload_models: bool = True
    disable_telemetry: bool = True
    disable_external_apis: bool = True
    use_fallback_embedding: bool = True
    max_cache_size_mb: int = 500
    auto_cleanup: bool = True
    cleanup_interval_hours: int = 24

    @classmethod
    def from_env(cls) -> "OfflineConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Check if offline mode is enabled
        config.enabled = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        config.use_local_cache = (
            os.getenv("OFFLINE_USE_LOCAL_CACHE", "true").lower() == "true"
        )
        config.cache_dir = os.getenv("OFFLINE_CACHE_DIR", ".cache/offline")
        config.model_cache_dir = os.getenv("OFFLINE_MODEL_CACHE_DIR", ".cache/models")
        config.preload_models = (
            os.getenv("OFFLINE_PRELOAD_MODELS", "true").lower() == "true"
        )
        config.disable_telemetry = (
            os.getenv("OFFLINE_DISABLE_TELEMETRY", "true").lower() == "true"
        )
        config.disable_external_apis = (
            os.getenv("OFFLINE_DISABLE_EXTERNAL_APIS", "true").lower() == "true"
        )
        config.use_fallback_embedding = (
            os.getenv("OFFLINE_USE_FALLBACK_EMBEDDING", "true").lower() == "true"
        )

        try:
            config.max_cache_size_mb = int(
                os.getenv("OFFLINE_MAX_CACHE_SIZE_MB", "500")
            )
        except ValueError:
            config.max_cache_size_mb = 500

        config.auto_cleanup = (
            os.getenv("OFFLINE_AUTO_CLEANUP", "true").lower() == "true"
        )

        try:
            config.cleanup_interval_hours = int(
                os.getenv("OFFLINE_CLEANUP_INTERVAL_HOURS", "24")
            )
        except ValueError:
            config.cleanup_interval_hours = 24

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "use_local_cache": self.use_local_cache,
            "cache_dir": self.cache_dir,
            "model_cache_dir": self.model_cache_dir,
            "preload_models": self.preload_models,
            "disable_telemetry": self.disable_telemetry,
            "disable_external_apis": self.disable_external_apis,
            "use_fallback_embedding": self.use_fallback_embedding,
            "max_cache_size_mb": self.max_cache_size_mb,
            "auto_cleanup": self.auto_cleanup,
            "cleanup_interval_hours": self.cleanup_interval_hours,
        }


# ============================================================================
# LOCAL FILE CACHE
# ============================================================================


class LocalFileCache:
    """File-based cache for offline mode."""

    def __init__(self, cache_dir: str = ".cache/offline", max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        # Sanitize key for filesystem
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        return self.cache_dir / f"{safe_key}.cache"

    def _get_metadata_path(self) -> Path:
        """Get metadata file path."""
        return self.cache_dir / "metadata.json"

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata."""
        meta_path = self._get_metadata_path()
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save cache metadata."""
        try:
            with open(self._get_metadata_path(), "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        # Check if expired
        metadata = self._load_metadata()
        if key in metadata:
            expiry = metadata[key].get("expiry", 0)
            if expiry > 0 and time.time() > expiry:
                # Expired, remove it
                self.delete(key)
                return None

        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read cache for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        """Set value in cache with TTL."""
        cache_path = self._get_cache_path(key)
        metadata = self._load_metadata()

        # Check size before writing
        self._check_size_limit()

        try:
            with open(cache_path, "w") as f:
                json.dump(value, f)

            metadata[key] = {
                "created": time.time(),
                "expiry": time.time() + ttl_seconds,
                "size": cache_path.stat().st_size,
            }
            self._save_metadata(metadata)
        except Exception as e:
            logger.error(f"Failed to write cache for {key}: {e}")

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                cache_path.unlink()
                metadata = self._load_metadata()
                if key in metadata:
                    del metadata[key]
                    self._save_metadata(metadata)
            except Exception as e:
                logger.error(f"Failed to delete cache for {key}: {e}")

    def clear(self) -> None:
        """Clear all cache."""
        try:
            for file in self.cache_dir.glob("*.cache"):
                file.unlink()
            metadata_path = self._get_metadata_path()
            if metadata_path.exists():
                metadata_path.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def _check_size_limit(self) -> None:
        """Check and enforce size limit."""
        if self.max_size_mb <= 0:
            return

        total_size = 0
        files = list(self.cache_dir.glob("*.cache"))

        for file in files:
            total_size += file.stat().st_size

        # If over limit, remove oldest files
        if total_size > self.max_size_mb * 1024 * 1024:
            files.sort(key=lambda f: f.stat().st_mtime)
            while files and total_size > self.max_size_mb * 1024 * 1024:
                file = files.pop(0)
                total_size -= file.stat().st_size
                file.unlink()
                logger.info(f"Evicted cache file: {file.name}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in files)
        metadata = self._load_metadata()

        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "cache_dir": str(self.cache_dir),
            "metadata_entries": len(metadata),
            "usage_percent": (
                (total_size / (self.max_size_mb * 1024 * 1024)) * 100
                if self.max_size_mb > 0
                else 0
            ),
        }


# ============================================================================
# OFFLINE MODEL MANAGER
# ============================================================================


class OfflineModelManager:
    """Manager for offline model loading."""

    def __init__(self, config: OfflineConfig):
        self.config = config
        self.model_cache_dir = Path(config.model_cache_dir)
        self._models: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._ensure_model_dir()

    def _ensure_model_dir(self) -> None:
        """Ensure model directory exists."""
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get local path for a model."""
        safe_name = "".join(c for c in model_name if c.isalnum() or c in "./_-")
        return self.model_cache_dir / safe_name

    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is cached locally."""
        model_path = self.get_model_path(model_name)
        return model_path.exists()

    def get_model(
        self, model_name: str, load_func: Optional[callable] = None
    ) -> Optional[Any]:
        """Get model from cache or load it."""
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]

            if not self.is_model_cached(model_name) and load_func:
                # Download and cache model
                try:
                    model = load_func()
                    self._models[model_name] = model
                    return model
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    return None

            # Try to load from cache
            try:
                model_path = self.get_model_path(model_name)
                if model_path.exists():
                    # Load model from path
                    # This is a placeholder - actual loading depends on model type
                    self._models[model_name] = model_path
                    return model_path
            except Exception as e:
                logger.error(f"Failed to load model from cache: {e}")

            return None


# ============================================================================
# OFFLINE MODE MANAGER
# ============================================================================


class OfflineModeManager:
    """Main offline mode manager."""

    def __init__(self):
        self.config = OfflineConfig.from_env()
        self.cache = (
            LocalFileCache(
                cache_dir=self.config.cache_dir,
                max_size_mb=self.config.max_cache_size_mb,
            )
            if self.config.enabled
            else None
        )
        self.model_manager = (
            OfflineModelManager(self.config) if self.config.enabled else None
        )
        self._initialized = False

    def is_offline(self) -> bool:
        """Check if offline mode is enabled."""
        return self.config.enabled

    def initialize(self) -> None:
        """Initialize offline mode."""
        if not self.config.enabled:
            logger.info("Offline mode is disabled")
            return

        if self._initialized:
            return

        logger.info("Initializing offline mode...")

        # Create cache directories
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_cache_dir).mkdir(parents=True, exist_ok=True)

        # Preload models if configured
        if self.config.preload_models:
            self._preload_models()

        self._initialized = True
        logger.info("Offline mode initialized")

    def _preload_models(self) -> None:
        """Preload commonly used models."""
        logger.info("Preloading models for offline mode...")
        # This is a placeholder - actual preloading depends on the models
        # Models to preload:
        # - SentenceTransformer model
        # - Cross-Encoder model if used
        # - Any other required models

    def get_cache(self) -> Optional[LocalFileCache]:
        """Get local cache instance."""
        return self.cache

    def get_model_manager(self) -> Optional[OfflineModelManager]:
        """Get model manager instance."""
        return self.model_manager

    def get_status(self) -> dict[str, Any]:
        """Get offline mode status."""
        if not self.config.enabled:
            return {"enabled": False, "status": "disabled"}

        status = {
            "enabled": True,
            "config": self.config.to_dict(),
            "initialized": self._initialized,
            "cache_stats": self.cache.get_stats() if self.cache else {},
        }

        return status


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_offline_manager: Optional[OfflineModeManager] = None
_manager_lock = threading.Lock()


def get_offline_manager() -> OfflineModeManager:
    """Get global offline mode manager."""
    global _offline_manager
    with _manager_lock:
        if _offline_manager is None:
            _offline_manager = OfflineModeManager()
        return _offline_manager


def is_offline_mode() -> bool:
    """Check if offline mode is enabled."""
    manager = get_offline_manager()
    return manager.is_offline()


def get_offline_cache() -> Optional[LocalFileCache]:
    """Get offline cache instance."""
    manager = get_offline_manager()
    return manager.get_cache()


def initialize_offline_mode() -> None:
    """Initialize offline mode."""
    manager = get_offline_manager()
    manager.initialize()
