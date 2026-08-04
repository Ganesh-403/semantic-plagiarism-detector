from pathlib import Path


def test_common_exposes_required_helper():
    source = Path("src/db/migrations/common.py").read_text(encoding="utf-8")
    assert "def get_migration_status(" in source
    assert '"current_version"' in source
    assert '"target_version"' in source
    assert '"pending_migrations"' in source


def test_cli_exposes_requested_flag_and_subcommand():
    source = Path("src/cli.py").read_text(encoding="utf-8")
    assert 'argv[0] == "--db-status"' in source
    assert '"db-status"' in source
    assert "run_db_status(" in source
