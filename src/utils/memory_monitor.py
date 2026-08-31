"""Memory monitoring utility for detecting leaks."""

import logging
import os
from typing import Any, Callable, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


def get_memory_usage() -> dict[str, Any]:
    """Get current memory usage of the process."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        return {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(interval=0.1),
        }
    except Exception as e:
        logger.error("Failed to get memory usage: %s", e)
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0, "cpu_percent": 0}


def log_memory_usage(tag: str = "") -> dict[str, Any]:
    """Log current memory usage with a tag."""
    usage = get_memory_usage()
    logger.info(
        "[Memory] %s - RSS: %.1fMB, VMS: %.1fMB, Process: %.1f%%",
        tag,
        usage["rss_mb"],
        usage["vms_mb"],
        usage["percent"],
    )
    return usage
