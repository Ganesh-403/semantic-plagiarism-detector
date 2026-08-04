from pathlib import Path


SCRIPT_PATH = Path("scripts/generate_seed_data.py")
CLI_TEST_PATH = Path("tests/cli/test_cli.py")


def test_seed_generator_supports_isolated_output_directory():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"--seed-dir"' in source
    assert "seed_dir=config.seed_dir" not in source
    assert "seed_dir = config.seed_dir" in source
    assert "_clean_seed_files(seed_dir)" in source


def test_cli_test_compares_generated_and_reference_schema():
    source = CLI_TEST_PATH.read_text(encoding="utf-8")

    assert (
        "def test_seed_data_database_matches_active_corpus_schema"
        in source
    )
    assert "_database_schema_snapshot(generated_db)" in source
    assert "_database_schema_snapshot(reference_db)" in source
    assert "assert generated_schema == reference_schema" in source


def test_schema_snapshot_covers_required_database_metadata():
    source = CLI_TEST_PATH.read_text(encoding="utf-8")

    assert "PRAGMA table_info" in source
    assert "PRAGMA index_list" in source
    assert "PRAGMA index_info" in source
    assert "PRAGMA foreign_key_list" in source
    assert "PRAGMA user_version" in source
