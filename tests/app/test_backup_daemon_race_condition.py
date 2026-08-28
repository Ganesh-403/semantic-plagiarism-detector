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
tests/app/test_backup_daemon_race_condition.py
----------------------------------------------
Regression and functional tests for Issue #3334 regarding backup daemon initialization.
"""

import time
from unittest.mock import Mock, patch

import pytest


def test_backup_daemon_waits_for_timeout_after_startup():
    """Verify that backup daemon skips execution if time since startup is less than timeout."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    # Mock last_activity to be extremely old (e.g., 2 hours ago), which WOULD trigger it
    # immediately if the daemon didn't protect itself against startup race conditions.
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": time.time() - 7200,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        # Force time.time() to advance by only 5 seconds (less than the 1800s timeout)
        current_time = time.time()
        with patch("time.time", side_effect=[current_time, current_time + 5]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            # Since daemon_start_time = current_time, and now = current_time + 5
            # now - daemon_start_time (5) < timeout (1800), so it skips completely
            mock_snapshot.assert_not_called()
            # Should not have even attempted to read last_activity
            mock_cache.get.assert_called_once_with("spd:v1:global:last_backup_time")


def test_backup_daemon_triggers_after_timeout_after_startup():
    """Verify that backup daemon triggers normally once the timeout period has fully elapsed."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": time.time() - 7200,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot", return_value=b"snap"
    ) as mock_snapshot, patch("src.db.database_backup.cleanup_old_backups"), patch(
        "src.core.app_config.get_backup_dir"
    ), patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        # Force time to advance by 1801 seconds, which exceeds the timeout!
        with patch(
            "time.time",
            side_effect=[current_time, current_time + 1801, current_time + 1801],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            # It should trigger the backup
            mock_snapshot.assert_called_once()


def test_backup_daemon_missing_last_activity_on_startup():
    """Verify that missing last_activity on startup initializes properly without triggering backup."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    # Missing last_activity
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        # Fast forward 2000s so it doesn't skip due to startup wait
        with patch(
            "time.time",
            side_effect=[current_time, current_time + 2000, current_time + 2000],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            # Because idle = 0, it is LESS than timeout (1800), so backup should NOT trigger
            mock_cache.set.assert_called_with(
                "spd:v1:global:last_activity", current_time + 2000
            )
            mock_snapshot.assert_not_called()


def test_backup_daemon_malformed_last_activity_on_startup():
    """Verify that a malformed (non-float) last_activity is cleanly recovered."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": "invalid_string_not_a_float",
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        with patch(
            "time.time",
            side_effect=[current_time, current_time + 2000, current_time + 2000],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_cache.set.assert_called_with(
                "spd:v1:global:last_activity", current_time + 2000
            )
            mock_snapshot.assert_not_called()


def test_backup_daemon_startup_with_active_sessions():
    """Verify backup daemon avoids execution if active sessions exist even if timeout passed."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": time.time() - 7200,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=3), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        with patch(
            "time.time",
            side_effect=[current_time, current_time + 2000, current_time + 2000],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_not_called()


def test_backup_daemon_exactly_at_timeout_boundary():
    """Verify behavior when precisely at the timeout boundary."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": time.time() - 1800,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot", return_value=b"snap"
    ) as mock_snapshot, patch("src.db.database_backup.cleanup_old_backups"), patch(
        "src.core.app_config.get_backup_dir"
    ), patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        with patch(
            "time.time",
            side_effect=[current_time, current_time + 1800, current_time + 1800],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_called_once()


def test_backup_daemon_repeated_iterations():
    """Verify daemon skips first iteration correctly but executes on subsequent ones if conditions match."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()

    current_time = time.time()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": current_time - 7200,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot", return_value=b"snap"
    ) as mock_snapshot, patch("src.db.database_backup.cleanup_old_backups"), patch(
        "src.core.app_config.get_backup_dir"
    ), patch("time.sleep", side_effect=[None, InterruptedError("Stop loop")]):
        # Mock time to start, then advance 1000s, then advance 2000s
        with patch(
            "time.time",
            side_effect=[
                current_time,  # daemon_start_time
                current_time + 1000,  # First iteration now()
                current_time + 2000,  # Second iteration now()
                current_time + 2000,  # Second iteration now() (for idle check)
            ],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_called_once()


def test_backup_daemon_transition_from_active_to_zero_sessions():
    """Verify backup triggers only when sessions drop to 0 and timeout has passed."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    current_time = time.time()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": current_time - 7200,
    }.get(k, None)

    # First iteration: 3 active sessions. Second iteration: 0 active sessions.
    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", side_effect=[3, 0]), patch(
        "src.db.database_backup.create_corpus_database_snapshot", return_value=b"snap"
    ) as mock_snapshot, patch("src.db.database_backup.cleanup_old_backups"), patch(
        "src.core.app_config.get_backup_dir"
    ), patch("time.sleep", side_effect=[None, InterruptedError("Stop loop")]):
        with patch(
            "time.time",
            side_effect=[
                current_time,  # daemon_start_time
                current_time
                + 2000,  # First iteration now() (fails on active_sessions == 3)
                current_time + 4000,  # Second iteration now() (succeeds)
                current_time + 4000,  # Second iteration now() (for idle check)
            ],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_called_once()


def test_backup_daemon_startup_phase_skips_backup_but_updates_activity():
    """Verify that during the startup phase, it updates last_activity but does NOT backup."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": None,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        with patch("time.time", side_effect=[current_time, current_time + 100]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_cache.set.assert_called_with(
                "spd:v1:global:last_activity", current_time + 100
            )
            mock_snapshot.assert_not_called()


def test_backup_daemon_active_sessions_count_fails():
    """Verify that if get_active_sessions_count() returns -1, backup is skipped."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": time.time() - 7200,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=-1), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        with patch("time.time", side_effect=[current_time, current_time + 2000]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_not_called()


def test_backup_daemon_no_backup_if_activity_recent():
    """Verify backup doesn't trigger if last_activity was very recent."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    current_time = time.time()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": current_time + 1950,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        with patch("time.time", side_effect=[current_time, current_time + 2000]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_not_called()


def test_backup_daemon_last_activity_not_greater_than_last_backup_time():
    """Verify backup doesn't trigger if last_activity <= last_backup_time."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    current_time = time.time()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": str(current_time + 3000),
        "spd:v1:global:last_activity": current_time - 7200,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        with patch("time.time", side_effect=[current_time, current_time + 2000]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_not_called()


def test_backup_daemon_recovers_from_cache_error_gracefully():
    """Verify daemon loop continues despite cache exceptions."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = Exception("Redis connection lost")

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch(
        "time.sleep", side_effect=[None, InterruptedError("Stop loop")]
    ) as mock_sleep:
        try:
            _run_backup_daemon()
        except InterruptedError:
            pass

        assert mock_sleep.call_count == 2


def test_backup_daemon_mocked_time_progression():
    """Simulate multiple daemon loops with time progression to verify state transitions."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    current_time = time.time()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": None,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("src.db.database_backup.cleanup_old_backups"), patch(
        "src.core.app_config.get_backup_dir"
    ), patch(
        "time.sleep", side_effect=[None, None, None, InterruptedError("Stop loop")]
    ):
        # 1st loop: starts daemon (startup_phase)
        # 2nd loop: still in startup_phase
        # 3rd loop: past startup_phase, idle high enough
        with patch(
            "time.time",
            side_effect=[
                current_time,  # daemon_start_time
                current_time + 100,  # Loop 1
                current_time + 1000,  # Loop 2
                current_time + 2000,  # Loop 3
                current_time + 2000,  # Loop 3 idle calc
            ],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            assert mock_snapshot.call_count == 1


def test_backup_daemon_zero_sessions_but_idle_time_not_reached():
    """Verify that even with 0 active sessions, if idle < timeout, no backup occurs."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    current_time = time.time()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": current_time - 1000,  # idle is 1000 < 1800
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        with patch("time.time", side_effect=[current_time, current_time + 2000]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_not_called()


def test_backup_daemon_missing_state_values_reconstruction():
    """Verify missing cache values for last_activity and last_backup_time are handled properly."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    # Everything is missing
    mock_cache.get.return_value = None

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        with patch("time.time", side_effect=[current_time, current_time + 2000]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            # Should set last_activity to now
            mock_cache.set.assert_called_with(
                "spd:v1:global:last_activity", current_time + 2000
            )
            # idle = 0, so backup not triggered
            mock_snapshot.assert_not_called()


def test_backup_daemon_thread_safety_mock():
    """Mock the thread initialization to verify thread behavior."""
    import src.core.app_config as app_config
    from app.state_manager import init_backup_daemon

    # Reset flag for testing
    app_config._backup_daemon_started = False

    with patch("threading.Thread") as mock_thread:
        init_backup_daemon()
        mock_thread.assert_called_once()
        assert app_config._backup_daemon_started is True

        # Call again, should not spawn another thread
        init_backup_daemon()
        assert mock_thread.call_count == 1


def test_backup_daemon_last_activity_type_error_handling():
    """Verify handling when last_activity is an invalid type that throws TypeError."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": [1, 2, 3],  # List cannot be float()
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        current_time = time.time()
        with patch("time.time", side_effect=[current_time, current_time + 2000]):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_cache.set.assert_called_with(
                "spd:v1:global:last_activity", current_time + 2000
            )
            mock_snapshot.assert_not_called()


def test_backup_daemon_successive_backups():
    """Verify that multiple successive backups can occur if time progresses far enough."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    current_time = time.time()
    mock_cache.get.side_effect = lambda k: {
        "spd:v1:global:last_backup_time": "0.0",
        "spd:v1:global:last_activity": current_time - 7200,
    }.get(k, None)

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("src.db.database_backup.cleanup_old_backups"), patch(
        "src.core.app_config.get_backup_dir"
    ), patch("time.sleep", side_effect=[None, None, InterruptedError("Stop loop")]):
        # We need to simulate the cache returning the updated last_backup_time
        # after the first backup.
        def cache_get_mock(k):
            if k == "spd:v1:global:last_activity":
                return current_time - 7200
            if k == "spd:v1:global:last_backup_time":
                if mock_snapshot.call_count == 0:
                    return "0.0"
                else:
                    return str(current_time + 2000)
            return None

        mock_cache.get.side_effect = cache_get_mock

        with patch(
            "time.time",
            side_effect=[
                current_time,  # daemon_start_time
                current_time + 2000,  # Loop 1 now
                current_time + 2000,  # Loop 1 idle
                current_time + 4000,  # Loop 2 now
                current_time + 4000,  # Loop 2 idle
            ],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            # Because last_activity (-7200) is NOT > last_backup_time (+2000),
            # the second backup will actually NOT trigger unless activity was updated.
            # This validates the last_activity > last_backup_time logic!
            assert mock_snapshot.call_count == 1


def test_backup_daemon_successive_backups_with_activity():
    """Verify that multiple backups DO occur if activity is updated between them."""
    from app.state_manager import _run_backup_daemon

    mock_cache = Mock()
    current_time = time.time()

    with patch("app.state_manager.get_cache", return_value=mock_cache), patch(
        "src.core.app_config.get_backup_idle_timeout", return_value=1800
    ), patch("app.state_manager.get_active_sessions_count", return_value=0), patch(
        "src.db.database_backup.create_corpus_database_snapshot"
    ) as mock_snapshot, patch("src.db.database_backup.cleanup_old_backups"), patch(
        "src.core.app_config.get_backup_dir"
    ), patch("time.sleep", side_effect=[None, None, InterruptedError("Stop loop")]):

        def cache_get_mock(k):
            if k == "spd:v1:global:last_activity":
                if mock_snapshot.call_count == 0:
                    return current_time - 7200  # Before 1st backup
                else:
                    return current_time + 2500  # Before 2nd backup
            if k == "spd:v1:global:last_backup_time":
                if mock_snapshot.call_count == 0:
                    return "0.0"
                else:
                    return str(current_time + 2000)
            return None

        mock_cache.get.side_effect = cache_get_mock

        with patch(
            "time.time",
            side_effect=[
                current_time,  # daemon_start_time
                current_time + 2000,  # Loop 1 now
                current_time + 2000,  # Loop 1 idle check
                current_time + 5000,  # Loop 2 now
                current_time + 5000,  # Loop 2 idle check
            ],
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            # Now both backups should have triggered!
            assert mock_snapshot.call_count == 2
