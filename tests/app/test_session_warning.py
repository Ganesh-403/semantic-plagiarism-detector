from unittest.mock import MagicMock
import pytest
from app import state_manager
from app.components import session_countdown


def test_countdown_uses_server_deadline_and_two_minute_warning(monkeypatch):
    html = MagicMock()
    monkeypatch.setattr(session_countdown.components, "html", html)
    session_countdown.render_session_countdown(1000, 900)
    rendered = html.call_args.args[0]
    assert "const deadline = 1900000" in rendered
    assert "remaining > 120" in rendered
    assert 'role="alert"' in rendered


def test_expired_session_is_cleared(monkeypatch):
    ui = MagicMock()
    ui.session_state = {"authenticated": True, "username": "teacher", "role": "teacher", "last_interaction": 1000}
    ui.stop.side_effect = RuntimeError("session stopped")
    monkeypatch.setattr(state_manager, "st", ui)
    monkeypatch.setattr(state_manager, "get_session_state", lambda *args: 1000)
    monkeypatch.setattr(state_manager.time, "time", lambda: 1901)
    clear = MagicMock()
    monkeypatch.setattr(state_manager, "clear_session", clear)
    with pytest.raises(RuntimeError, match="session stopped"):
        state_manager.check_session_timeout("session1")
    assert "authenticated" not in ui.session_state
    clear.assert_called_once_with("session1")


def test_active_session_refreshes_server_timestamp(monkeypatch):
    ui = MagicMock()
    ui.session_state = {"authenticated": True, "last_interaction": 1000}
    monkeypatch.setattr(state_manager, "st", ui)
    monkeypatch.setattr(state_manager, "get_session_state", lambda *args: 1000)
    monkeypatch.setattr(state_manager.time, "time", lambda: 1100)
    cache = MagicMock()
    monkeypatch.setattr(state_manager, "cache_session_state", cache)
    state_manager.check_session_timeout("session1")
    assert ui.session_state["last_interaction"] == 1100
    cache.assert_called_once_with("session1", "last_interaction", 1100)
