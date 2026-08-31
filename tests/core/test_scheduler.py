"""
tests/core/test_scheduler.py
------------------------------
Unit tests for src.core.scheduler.RescanScheduler — the asyncio-based
background loop that periodically calls
``src.core.processing.rescan_recent_documents`` (see
src/asgi_app.py lifespan wiring).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from src.core.scheduler import RescanScheduler, get_scheduler


def test_scheduler_disabled_by_default(monkeypatch):
    """RESCAN_INTERVAL_MINUTES unset => disabled, matching the issue's
    'default disabled' requirement."""
    monkeypatch.delenv("RESCAN_INTERVAL_MINUTES", raising=False)
    scheduler = RescanScheduler()
    assert scheduler.enabled is False


def test_scheduler_enabled_with_positive_interval():
    scheduler = RescanScheduler(interval_minutes=15)
    assert scheduler.enabled is True


def test_scheduler_start_is_noop_when_disabled():
    async def _run():
        scheduler = RescanScheduler(interval_minutes=0)
        await scheduler.start()
        assert scheduler.is_running is False
        await scheduler.stop()  # must not raise even though never started

    asyncio.run(_run())


def test_scheduler_start_and_stop_when_enabled():
    async def _run():
        scheduler = RescanScheduler(interval_minutes=60)
        with patch.object(
            RescanScheduler, "run_once", new_callable=AsyncMock
        ) as mock_run_once:
            await scheduler.start()
            assert scheduler.is_running is True

            # Give the loop's first iteration a chance to execute.
            await asyncio.sleep(0.05)
            mock_run_once.assert_called_once()

            await scheduler.stop()
            assert scheduler.is_running is False

    asyncio.run(_run())


def test_scheduler_ticks_again_after_interval_elapses():
    """Verifies the loop re-runs on each interval tick (mocked/short interval,
    standing in for the 'mocked clock' requested in the issue)."""

    async def _run():
        # A tiny interval (in "minutes") so the test doesn't take real minutes;
        # the scheduler only cares about elapsed wall-clock seconds here.
        scheduler = RescanScheduler(interval_minutes=0.002)  # ~0.12s
        with patch.object(
            RescanScheduler, "run_once", new_callable=AsyncMock
        ) as mock_run_once:
            await scheduler.start()
            await asyncio.sleep(0.35)
            await scheduler.stop()

        assert mock_run_once.call_count >= 2

    asyncio.run(_run())


def test_scheduler_survives_a_failed_pass():
    """A single failed rescan pass must not kill the background loop."""

    async def _run():
        scheduler = RescanScheduler(interval_minutes=0.002)
        call_count = {"n": 0}

        async def _flaky_run_once():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated rescan failure")

        with patch.object(RescanScheduler, "run_once", _flaky_run_once):
            await scheduler.start()
            await asyncio.sleep(0.35)
            await scheduler.stop()

        assert call_count["n"] >= 2  # loop kept going after the failure

    asyncio.run(_run())


def test_scheduler_double_start_is_idempotent():
    async def _run():
        scheduler = RescanScheduler(interval_minutes=60)
        with patch.object(RescanScheduler, "run_once", new_callable=AsyncMock):
            await scheduler.start()
            first_task = scheduler._task
            await scheduler.start()  # should not replace the running task
            assert scheduler._task is first_task
            await scheduler.stop()

    asyncio.run(_run())


def test_get_scheduler_returns_singleton():
    first = get_scheduler()
    second = get_scheduler()
    assert first is second


def test_scheduler_run_once_dispatches_to_thread():
    """run_once must call rescan_recent_documents (off-thread) with the
    scheduler's configured grace period and threshold."""

    async def _run():
        scheduler = RescanScheduler(
            interval_minutes=60, grace_period_minutes=45, threshold=0.8
        )
        with patch("src.core.processing.rescan_recent_documents") as mock_rescan:
            mock_rescan.return_value = "sentinel-result"
            result = await scheduler.run_once()

        mock_rescan.assert_called_once_with(45, 0.8)
        assert result == "sentinel-result"

    asyncio.run(_run())
