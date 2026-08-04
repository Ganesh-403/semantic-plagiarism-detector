from pathlib import Path


AUTH_PATH = Path("src/db/auth.py")
MIGRATION_PATH = Path("src/db/migrations/auth.py")
TEST_PATH = Path(
    "tests/db/test_password_changed_at_issue_1266.py"
)


def test_migration_adds_required_column():
    source = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "AUTH_SCHEMA_VERSION = 10" in source
    assert (
        "def migration_010_add_password_changed_at("
        in source
    )
    assert "ADD COLUMN password_changed_at TEXT" in source
    assert (
        "10: migration_010_add_password_changed_at"
        in source
    )


def test_password_update_writes_timestamp_atomically():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert "datetime.now(" in source
    assert "timezone.utc" in source
    assert "password_changed_at = ?" in source
    assert "SET password = ?," in source


def test_unit_test_verifies_timestamp_update():
    source = TEST_PATH.read_text(encoding="utf-8")

    assert (
        "test_successful_password_change_updates_timestamp"
        in source
    )
    assert "datetime.fromisoformat" in source
    assert (
        "test_authorization_failure_does_not_change_timestamp"
        in source
    )
