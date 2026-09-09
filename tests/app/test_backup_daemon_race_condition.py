"""Deterministic backup-loop tests that execute complete polling iterations."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from app import state_manager as state


class StopDaemon(BaseException):
    pass


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    def run(events, *, last_activity=9000., last_backup=0., snapshot_error=None):
        clock = [10000.]
        values = {"spd:v1:global:last_activity": last_activity,
                  "spd:v1:global:last_backup_time": last_backup}
        cache = Mock()
        cache.get.side_effect = values.get
        cache.set.side_effect = lambda key, value: values.__setitem__(key, value)
        active = Mock(return_value=0)
        snapshot = Mock(return_value=b"valid snapshot", side_effect=snapshot_error)
        cleanup = Mock()
        events = iter(events)
        def poll(seconds):
            assert seconds == 30
            try:
                now, changes, count = next(events)
            except StopIteration:
                raise StopDaemon
            clock[0] = now
            values.update(changes)
            active.return_value = count
        monkeypatch.setattr(state, "time", SimpleNamespace(time=lambda: clock[0], sleep=poll))
        monkeypatch.setattr(state, "get_cache", lambda: cache)
        monkeypatch.setattr(state, "get_active_sessions_count", active)
        monkeypatch.setattr("src.core.app_config.get_backup_idle_timeout", lambda: 1800)
        monkeypatch.setattr("src.core.app_config.get_backup_dir", lambda: tmp_path)
        monkeypatch.setattr("src.db.database_backup.create_corpus_database_snapshot", snapshot)
        monkeypatch.setattr("src.db.database_backup.cleanup_old_backups", cleanup)
        with pytest.raises(StopDaemon):
            state._run_backup_daemon()
        return SimpleNamespace(snapshot=snapshot, cache=cache, values=values,
                               cleanup=cleanup, active=active, directory=tmp_path)
    return run


@pytest.mark.parametrize("now, expected", [(10005, 0), (11799, 0), (11800, 1), (12000, 1)])
def test_startup_grace_period_including_exact_boundary(daemon, now, expected):
    result = daemon([(now, {}, 0)])
    assert result.snapshot.call_count == expected
    result.active.assert_called_once()
    if expected:
        assert result.values["spd:v1:global:last_backup_time"] == now
        assert next(result.directory.glob("*.db")).read_bytes() == b"valid snapshot"
        result.cleanup.assert_called_once_with(result.directory, max_backups=10, max_age_days=30)


@pytest.mark.parametrize("count", [-1, 1, 4])
def test_active_or_unknown_session_count_prevents_backup(daemon, count):
    result = daemon([(12000, {}, count)])
    result.active.assert_called_once()
    result.snapshot.assert_not_called()


def test_transition_from_active_to_idle(daemon):
    result = daemon([(12000, {}, 1), (12030, {}, 0)])
    result.snapshot.assert_called_once()
    assert result.values["spd:v1:global:last_backup_time"] == 12030


@pytest.mark.parametrize("value", ["invalid", {}, float("nan"), float("inf"), -1])
def test_malformed_activity_restarts_idle_timer(daemon, value):
    result = daemon([(12000, {}, 0)], last_activity=value)
    result.snapshot.assert_not_called()
    assert result.values["spd:v1:global:last_activity"] == 12000


def test_missing_activity_is_initialized_at_startup(daemon):
    result = daemon([(10030, {}, 0)], last_activity=None)
    result.snapshot.assert_not_called()
    assert result.values["spd:v1:global:last_activity"] == 10000


@pytest.mark.parametrize("last_backup", [9000., "10000", 12500.])
def test_no_repeat_without_new_activity(daemon, last_backup):
    result = daemon([(12000, {}, 0)], last_backup=last_backup)
    result.snapshot.assert_not_called()


@pytest.mark.parametrize("last_backup", ["bad", float("nan"), -20])
def test_malformed_last_backup_does_not_disable_backups(daemon, last_backup):
    result = daemon([(12000, {}, 0)], last_backup=last_backup)
    result.snapshot.assert_called_once()


def test_recent_activity_during_startup_is_preserved(daemon):
    key = "spd:v1:global:last_activity"
    result = daemon([(11000, {key: 10999}, 0), (12000, {}, 0), (13000, {}, 0)])
    result.snapshot.assert_called_once()
    assert result.values[key] == 10999
    assert result.values["spd:v1:global:last_backup_time"] == 13000


def test_successive_backups_require_intervening_activity(daemon):
    key = "spd:v1:global:last_activity"
    result = daemon([(12000, {}, 0), (15000, {}, 0), (16000, {key: 15999}, 1), (18000, {}, 0)])
    assert result.snapshot.call_count == 2
    assert result.values["spd:v1:global:last_backup_time"] == 18000


def test_snapshot_failure_is_retried_without_advancing_checkpoint(daemon):
    result = daemon([(12000, {}, 0), (12030, {}, 0)], snapshot_error=OSError("disk error"))
    assert result.snapshot.call_count == 2
    assert result.values["spd:v1:global:last_backup_time"] == 0
    result.cleanup.assert_not_called()
    assert not list(result.directory.glob("*.db"))


def test_backup_daemon_thread_initialization_is_idempotent(monkeypatch):
    monkeypatch.setattr("src.core.app_config._backup_daemon_started", False, raising=False)
    thread = Mock()
    monkeypatch.setattr(state.threading, "Thread", thread)
    state.init_backup_daemon()
    state.init_backup_daemon()
    thread.assert_called_once_with(target=state._run_backup_daemon, daemon=True)
    thread.return_value.start.assert_called_once()
