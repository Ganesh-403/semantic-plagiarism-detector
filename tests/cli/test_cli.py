import json
import sqlite3
import subprocess
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import MockDataFactory

# Mock ML libraries to prevent pytest segmentation faults on Apple Silicon
sys.modules["transformers"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from src.cli import main, run_prewarm, run_scan  # noqa: E402


@pytest.fixture
def temp_assignments_dir(tmp_path):
    """Creates a temporary folder with valid and invalid assignment files."""
    d = tmp_path / "assignments"
    d.mkdir()

    # Valid files
    (d / "doc1.txt").write_text("This is assignment one content.")
    (d / "doc2.txt").write_text("This is assignment two content.")

    # Unsupported file extension
    (d / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Hidden file
    (d / ".hidden.txt").write_text("This is a hidden file.")

    return d


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_scan_success(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """Test a successful CLI scan on a directory with valid documents yielding JSON output."""
    exit_code = run_scan(str(temp_assignments_dir), threshold=0.8, output_format="json")

    assert exit_code == 0
    captured = capsys.readouterr()

    # Parse output as JSON
    report = json.loads(captured.out)
    assert report["documents_processed"] == 2
    assert report["threshold"] == 0.8
    assert len(report["matches"]) == 1

    match = report["matches"][0]
    assert match["document_1"] == "doc1.txt"
    assert match["document_2"] == "doc2.txt"
    assert match["similarity_score"] == 1.0


def test_cli_scan_success_text_format(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """Test a successful CLI scan with plain text format."""
    exit_code = run_scan(str(temp_assignments_dir), threshold=0.8, output_format="text")

    assert exit_code == 0
    captured = capsys.readouterr()

    assert "Documents Processed: 2" in captured.out
    assert "Similarity Threshold: 0.8" in captured.out
    assert "Matches Found:" in captured.out
    assert "- doc1.txt <-> doc2.txt: 1.0000" in captured.out


def test_cli_scan_success_csv_format(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """Test a successful CLI scan with CSV format."""
    exit_code = run_scan(str(temp_assignments_dir), threshold=0.8, output_format="csv")

    assert exit_code == 0
    captured = capsys.readouterr()

    lines = captured.out.strip().split("\n")
    assert lines[0] == "document_1,document_2,similarity_score"
    assert lines[1] == "doc1.txt,doc2.txt,1.0"



def test_cli_scan_invalid_folder(capsys):
    """Test scanning a folder that does not exist."""
    exit_code = run_scan("/nonexistent_path_foo_bar", threshold=0.8)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error: Folder" in captured.err


def test_cli_scan_empty_folder(tmp_path, capsys):
    """Test scanning an empty folder."""
    d = tmp_path / "empty"
    d.mkdir()

    exit_code = run_scan(str(d), threshold=0.8)
    assert exit_code == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["documents_processed"] == 0
    assert len(report["matches"]) == 0


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_prewarm_folder_success(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """Test prewarming cache for a directory of documents."""
    exit_code = run_prewarm(str(temp_assignments_dir))
    assert exit_code == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["prewarmed_documents"] == 2
    assert report["status"] == "success"


def test_cli_prewarm_invalid_folder(capsys):
    """Test prewarming with a non-existent folder."""
    exit_code = run_prewarm("/nonexistent_path_foo_bar")
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Error: Folder" in captured.err


@patch(
    "src.db.corpus_db.get_all_documents",
    return_value=[{"filename": "doc1.pdf"}, {"filename": "doc2.pdf"}],
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_prewarm_db_success(mock_embed, mock_docs, capsys):
    """Test prewarming cache using database documents when no folder is provided."""
    exit_code = run_prewarm()
    assert exit_code == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["prewarmed_documents"] == 2
    assert report["status"] == "success"


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_main_prewarm_command(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """Test main CLI invocation with prewarm subcommand."""
    with patch("sys.argv", ["cli.py", "prewarm", "--folder", str(temp_assignments_dir)]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_cli_main_invalid_threshold():
    """Test main function with invalid threshold range."""
    with patch("sys.argv", ["cli.py", "scan", "./assignments", "--threshold", "1.5"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1


def test_cli_main_invalid_command():
    """Test main function with an invalid subcommand/command."""
    with patch("sys.argv", ["cli.py", "invalid_cmd", "./assignments"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        # argparse subparsers exit with 2 on invalid arguments/subcommands
        assert excinfo.value.code == 2


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_main_scan_format(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """Test main function with scan subcommand specifying output format."""
    with patch("sys.argv", ["cli.py", "scan", str(temp_assignments_dir), "--output-format", "json"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["documents_processed"] == 2




def _normalized_sql(sql: str | None) -> str:
    """Normalize SQLite DDL for stable schema comparisons."""
    return " ".join((sql or "").split()).casefold()


def _database_schema_snapshot(db_path: Path) -> dict:
    """Return tables, columns, indexes, foreign keys, and user version."""
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        table_rows = connection.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        snapshot = {
            "user_version": connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0],
            "tables": {},
        }

        for table_name, create_sql in table_rows:
            columns = [
                {
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2].upper(),
                    "not_null": row[3],
                    "default": row[4],
                    "primary_key": row[5],
                }
                for row in connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]

            indexes = []
            for index_row in connection.execute(
                f'PRAGMA index_list("{table_name}")'
            ).fetchall():
                index_name = index_row[1]
                index_columns = [
                    row[2]
                    for row in connection.execute(
                        f'PRAGMA index_info("{index_name}")'
                    ).fetchall()
                ]
                indexes.append(
                    {
                        "name": index_name,
                        "unique": index_row[2],
                        "origin": index_row[3],
                        "partial": index_row[4],
                        "columns": index_columns,
                    }
                )

            foreign_keys = [
                {
                    "table": row[2],
                    "from": row[3],
                    "to": row[4],
                    "on_update": row[5],
                    "on_delete": row[6],
                    "match": row[7],
                }
                for row in connection.execute(
                    f'PRAGMA foreign_key_list("{table_name}")'
                ).fetchall()
            ]

            snapshot["tables"][table_name] = {
                "sql": _normalized_sql(create_sql),
                "columns": columns,
                "indexes": sorted(
                    indexes,
                    key=lambda item: item["name"],
                ),
                "foreign_keys": sorted(
                    foreign_keys,
                    key=lambda item: (
                        item["table"],
                        item["from"],
                        item["to"],
                    ),
                ),
            }

        return snapshot


def test_seed_data_database_matches_active_corpus_schema(tmp_path):
    """Seed output must match a database initialized by corpus_db.py."""
    repository_root = Path(__file__).resolve().parents[2]
    generated_dir = tmp_path / "generated-seed"
    reference_db = tmp_path / "reference-corpus.db"

    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "generate_seed_data.py"),
            "--seed-dir",
            str(generated_dir),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Seed generation failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    generated_db = generated_dir / "corpus.db"
    assert generated_db.is_file()
    assert (generated_dir / "users.db").is_file()
    assert (generated_dir / "corpus.index").is_file()

    # Initialize a separate database using the active corpus_db definitions.
    from src.db import corpus_db

    original_path = corpus_db.get_corpus_db_path()
    try:
        corpus_db.configure_db_path(reference_db)
        corpus_db.init_corpus_db()
        corpus_db.close_connections()
    finally:
        corpus_db.configure_db_path(original_path)

    generated_schema = _database_schema_snapshot(generated_db)
    reference_schema = _database_schema_snapshot(reference_db)

    assert generated_schema == reference_schema
