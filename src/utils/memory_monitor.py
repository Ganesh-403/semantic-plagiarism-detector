"""Memory monitoring utility for detecting leaks."""

import logging
import math
import os
from typing import Any, Callable, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


def check_memory_threshold(
    threshold_percent: float = 85.0,
    on_exceeded: Optional[Callable[[], None]] = None,
) -> bool:
    """Report process memory above a percentage and optionally run a callback."""
    if not math.isfinite(threshold_percent) or not 0 <= threshold_percent <= 100:
        raise ValueError("threshold_percent must be finite and between 0 and 100")
    usage = get_memory_usage()
    exceeded = usage["percent"] > threshold_percent
    if exceeded:
        logger.warning("Process memory %.1f%% exceeds %.1f%%", usage["percent"], threshold_percent)
        if on_exceeded is not None:
            try:
                on_exceeded()
            except Exception:
                logger.exception("Memory threshold callback failed")
    return exceeded


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
