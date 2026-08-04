from pathlib import Path


AUTH_PATH = Path("src/db/auth.py")


def test_failed_event_is_logged():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert '"password_change_failed"' in source
    assert "def _log_password_change_failure(" in source


def test_required_reasons_exist():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert '"incorrect_old_password"' in source
    assert '"complexity_failed"' in source


def test_update_password_accepts_old_password():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert "old_password: str | None = None" in source
    assert "_verify_stored_password(" in source
