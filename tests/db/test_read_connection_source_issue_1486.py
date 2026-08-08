from pathlib import Path


SOURCE = Path("src/db/common.py")
TESTS = Path("tests/db/test_common.py")


def test_required_helper_and_return_type_exist():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def get_read_connection(" in source
    assert "db_path: Path" in source
    assert ") -> sqlite3.Connection:" in source


def test_connection_uses_read_only_sqlite_uri():
    source = SOURCE.read_text(encoding="utf-8")

    assert "?mode=ro" in source
    assert "uri=True" in source
    assert "sqlite3.connect(" in source


def test_write_rejection_unit_test_exists():
    source = TESTS.read_text(encoding="utf-8")

    assert (
        "test_get_read_connection_rejects_write_attempts"
        in source
    )
    assert "sqlite3.OperationalError" in source
