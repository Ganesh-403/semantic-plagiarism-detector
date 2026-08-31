"""
tests/app/test_active_sessions_count.py
---------------------------------------
Unit tests for get_active_sessions_count error handling and backup daemon safety (Issue #2809).
"""

import time
from unittest.mock import Mock, patch

from app.state_manager import get_active_sessions_count


class TestGetActiveSessionsCount:
    """Tests for get_active_sessions_count function."""

    def test_active_sessions_count_success(self):
        """Test active sessions within timeout limit are counted correctly."""
        mock_cache = Mock()
        mock_cache.is_available.return_value = True
        mock_cache._client.scan_iter.return_value = [
            b"spd:v1:session:sess1:last_interaction",
            b"spd:v1:session:sess2:last_interaction",
        ]
        mock_cache.fallback_cache = {}

        now = time.time()

        def mock_get_session_state(session_id, key):
            if session_id == "sess1":
                return now - 60  # active (1 min ago)
            elif session_id == "sess2":
                return now - 1200  # expired (20 min ago)
            return None

        with (
            patch("app.state_manager.get_cache", return_value=mock_cache),
            patch(
                "app.state_manager.get_session_state",
                side_effect=mock_get_session_state,
            ),
        ):
            count = get_active_sessions_count()
            assert count == 1
            mock_cache._client.scan_iter.assert_called_once_with(
                match="spd:v1:session:*:last_interaction"
            )
            assert (
                not hasattr(mock_cache._client, "keys")
                or not mock_cache._client.keys.called
            )

    def test_active_sessions_count_uses_scan_iter_instead_of_keys(self):
        """Explicitly verify that .scan_iter(match=...) is used instead of .keys() (Issue #2786)."""
        mock_cache = Mock()
        mock_cache.is_available.return_value = True
        mock_cache._client.scan_iter.return_value = []
        mock_cache.fallback_cache = {}

        with patch("app.state_manager.get_cache", return_value=mock_cache):
            count = get_active_sessions_count()
            assert count == 0
            mock_cache._client.scan_iter.assert_called_once_with(
                match="spd:v1:session:*:last_interaction"
            )
            # Ensure .keys() was never called
            mock_cache._client.keys.assert_not_called()

    def test_active_sessions_count_zero_when_none_active(self):
        """Test returns 0 when no sessions are active."""
        mock_cache = Mock()
        mock_cache.is_available.return_value = True
        mock_cache._client.scan_iter.return_value = []
        mock_cache.fallback_cache = {}

        with patch("app.state_manager.get_cache", return_value=mock_cache):
            count = get_active_sessions_count()
            assert count == 0

    def test_active_sessions_count_returns_minus_one_on_redis_scan_failure(self):
        """Test returns -1 when Redis scan fails and fallback is unavailable."""
        mock_cache = Mock()
        mock_cache.is_available.return_value = True
        mock_cache._client.scan_iter.side_effect = Exception("Redis connection refused")
        mock_cache.fallback_cache = {}

        with patch("app.state_manager.get_cache", return_value=mock_cache):
            count = get_active_sessions_count()
            assert count == -1

    def test_active_sessions_count_returns_minus_one_when_cache_none(self):
        """Test returns -1 when get_cache returns None."""
        with patch("app.state_manager.get_cache", return_value=None):
            count = get_active_sessions_count()
            assert count == -1

    def test_active_sessions_count_returns_minus_one_on_unhandled_exception(self):
        """Test returns -1 when an unexpected exception is raised."""
        with patch(
            "app.state_manager.get_cache", side_effect=RuntimeError("Fatal error")
        ):
            count = get_active_sessions_count()
            assert count == -1


class TestBackupDaemonSafety:
    """Tests for backup daemon skipping execution when active sessions count is negative."""

    def test_backup_daemon_skips_when_active_sessions_negative(self):
        """Verify backup is not executed when get_active_sessions_count returns -1."""
        from app.state_manager import _run_backup_daemon

        mock_cache = Mock()
        mock_cache.get.side_effect = lambda k: {
            "spd:v1:global:last_backup_time": "0.0",
            "spd:v1:global:last_activity": time.time() - 3600,
        }.get(k, None)

        with (
            patch("app.state_manager.get_cache", return_value=mock_cache),
            patch("app.state_manager.get_active_sessions_count", return_value=-1),
            patch("src.core.app_config.get_backup_idle_timeout", return_value=1800),
            patch(
                "src.db.database_backup.create_corpus_database_snapshot"
            ) as mock_snapshot,
            patch("time.sleep", side_effect=InterruptedError("Stop loop")),
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            # Snapshot must NOT be called because active_sessions was -1
            mock_snapshot.assert_not_called()

    def test_backup_daemon_triggers_when_zero_sessions_and_idle(self, tmp_path):
        """Verify backup daemon triggers snapshot when active sessions count is 0 and idle."""
        from app.state_manager import _run_backup_daemon

        mock_cache = Mock()
        mock_cache.get.side_effect = lambda k: {
            "spd:v1:global:last_backup_time": 0.0,
            "spd:v1:global:last_activity": time.time() - 3600,
        }.get(k, None)

        fake_db = tmp_path / "corpus.db"
        fake_db.write_bytes(b"mock_db")

        with (
            patch("app.state_manager.get_cache", return_value=mock_cache),
            patch("app.state_manager.get_active_sessions_count", return_value=0),
            patch("src.core.app_config.get_backup_idle_timeout", return_value=1800),
            patch("src.db.corpus_db.get_corpus_db_path", return_value=fake_db),
            patch(
                "src.db.database_backup.create_corpus_database_snapshot",
                return_value=b"snapshot_data",
            ) as mock_snapshot,
            patch("time.sleep", side_effect=[None, InterruptedError("Stop loop")]),
        ):
            try:
                _run_backup_daemon()
            except InterruptedError:
                pass

            mock_snapshot.assert_called_once()
