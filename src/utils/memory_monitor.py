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

"""Memory monitoring utility for detecting leaks."""

import logging
import os
from typing import Any, Dict

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


def check_memory_threshold(
    threshold_percent: float = 85.0,
    on_exceeded: Optional[Callable[[], None]] = None,
) -> bool:
    """Check whether current process memory usage exceeds a threshold.

    Intended to be polled periodically (e.g. from a background worker loop)
    so the caller can react to memory pressure — for example, pausing heavy
    background jobs or clearing caches — before the process is killed by an
    OOM handler.

    Args:
        threshold_percent: Percentage of total system memory (as reported by
            ``psutil.Process.memory_percent()``) above which usage is
            considered to have exceeded the threshold. Defaults to 85.0.
        on_exceeded: Optional zero-argument callable invoked when memory
            usage exceeds ``threshold_percent``. Not called when usage is
            at or below the threshold. If the callback itself raises, the
            exception is logged and swallowed rather than propagated, so a
            failing callback cannot crash the caller's monitoring loop.

    Returns:
        True if current memory usage exceeds ``threshold_percent``, else
        False. Returned regardless of whether ``on_exceeded`` was provided.
    """
    usage = get_memory_usage()
    current_percent = usage.get("percent", 0)
    exceeded = current_percent > threshold_percent

    if exceeded:
        logger.warning(
            "Memory usage %.1f%% exceeds threshold %.1f%% (RSS: %.1fMB)",
            current_percent,
            threshold_percent,
            usage.get("rss_mb", 0),
        )
        if on_exceeded is not None:
            try:
                on_exceeded()
            except Exception as e:
                logger.error("on_exceeded callback raised an exception: %s", e)

    return exceeded
