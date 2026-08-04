import json
from unittest.mock import patch

from src.cli import run_db_status


def test_text_status_output(capsys):
    with patch(
        "src.db.migrations.get_migration_status",
        return_value={
            "current_version": 2,
            "target_version": 4,
            "pending_migrations": [3, 4],
        },
    ):
        result = run_db_status("corpus.db", "corpus")

    output = capsys.readouterr().out
    assert result == 0
    assert "Current version: 2" in output
    assert "Target version: 4" in output
    assert "Pending migrations: 3, 4" in output


def test_json_status_output(capsys):
    expected = {
        "current_version": 4,
        "target_version": 4,
        "pending_migrations": [],
    }
    with patch(
        "src.db.migrations.get_migration_status",
        return_value=expected,
    ):
        result = run_db_status(
            "corpus.db",
            "corpus",
            output_format="json",
        )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_status_error_returns_nonzero(capsys):
    with patch(
        "src.db.migrations.get_migration_status",
        side_effect=FileNotFoundError("missing"),
    ):
        result = run_db_status("missing.db", "auth")

    assert result == 1
    assert "missing" in capsys.readouterr().err
