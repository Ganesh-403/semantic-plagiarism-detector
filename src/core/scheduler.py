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
src/core/scheduler.py
----------------------
Lightweight asyncio-based scheduler for the continuous plagiarism-rescan
job (see ``src.core.processing.rescan_recent_documents``).

Why a plain asyncio loop instead of APScheduler?
-------------------------------------------------
The project's dependency set intentionally stays small (no APScheduler is
currently pinned in requirements*.txt, and this environment has no network
access to add one). A single `asyncio.Task` that sleeps for the configured
interval and re-runs the rescan is sufficient for the "one job, one
interval" requirement here, needs no extra dependency, and is trivial to
drive from tests via a fake clock/stop-event instead of real sleeps.

Usage (see ``src/asgi_app.py`` for the actual wiring):

    scheduler = get_scheduler()
    await scheduler.start()   # on app startup
    ...
    await scheduler.stop()    # on app shutdown
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.core.app_config import get_rescan_interval_minutes
from src.core.config import PLAGIARISM_THRESHOLD

logger = logging.getLogger(__name__)


class RescanScheduler:
    """Runs ``rescan_recent_documents`` on a fixed interval in the background.

    The scheduler is disabled by default (``RESCAN_INTERVAL_MINUTES`` unset
    or ``0``); ``start()`` becomes a no-op in that case so the job never
    silently runs against a deployment that hasn't opted in.
    """

    def __init__(
        self,
        interval_minutes: int | None = None,
        grace_period_minutes: int | None = None,
        threshold: float = PLAGIARISM_THRESHOLD,
    ) -> None:
        self.interval_minutes = (
            interval_minutes
            if interval_minutes is not None
            else get_rescan_interval_minutes()
        )
        self.grace_period_minutes = grace_period_minutes
        self.threshold = threshold

        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def enabled(self) -> bool:
        """Whether a positive interval has been configured."""
        return self.interval_minutes > 0

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the background loop. No-op if disabled or already running."""
        if not self.enabled:
            logger.info(
                "Scheduled plagiarism rescan is disabled "
                "(set RESCAN_INTERVAL_MINUTES to enable)."
            )
            return
        if self.is_running:
            return

        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run_loop(), name="plagiarism-rescan-scheduler"
        )
        logger.info(
            "Started scheduled plagiarism rescan job (interval=%d minutes).",
            self.interval_minutes,
        )

    async def stop(self) -> None:
        """Stop the background loop gracefully, if running."""
        if self._task is None:
            return

        if self._stop_event is not None:
            self._stop_event.set()

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._stop_event = None
        logger.info("Stopped scheduled plagiarism rescan job.")

    async def _run_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                # A single failed rescan pass must never kill the loop —
                # log and retry on the next tick.
                logger.exception("Scheduled plagiarism rescan pass failed.")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_minutes * 60,
                )
            except asyncio.TimeoutError:
                pass  # Normal case: interval elapsed, loop again.

    async def run_once(self):
        """Run a single rescan pass off the event loop thread.

        ``rescan_recent_documents`` does blocking SQLite/FAISS I/O, so it
        is dispatched via ``asyncio.to_thread`` rather than awaited
        directly, keeping the ASGI event loop responsive to concurrent
        requests (e.g. manual scans) while the rescan runs.
        """
        from src.core.processing import rescan_recent_documents

        return await asyncio.to_thread(
            rescan_recent_documents,
            self.grace_period_minutes,
            self.threshold,
        )


_scheduler_singleton: RescanScheduler | None = None


def get_scheduler() -> RescanScheduler:
    """Return the process-wide :class:`RescanScheduler` singleton.

    Created lazily so importing this module never has side effects, and so
    tests can construct their own ``RescanScheduler`` instances without
    touching global state.
    """
    global _scheduler_singleton
    if _scheduler_singleton is None:
        _scheduler_singleton = RescanScheduler()
    return _scheduler_singleton


async def start_scheduler() -> None:
    """Convenience entrypoint for ASGI lifespan startup."""
    await get_scheduler().start()


async def stop_scheduler() -> None:
    """Convenience entrypoint for ASGI lifespan shutdown."""
    scheduler = get_scheduler()
    await scheduler.stop()
