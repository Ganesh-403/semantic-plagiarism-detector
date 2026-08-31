import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import MockDataFactory
from src.core.similarity import PLAGIARISM_THRESHOLD

# Mock ML libraries to prevent pytest segmentation faults on Apple Silicon
sys.modules["transformers"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from src.cli import _natural_sort_key, main, run_prewarm, run_scan  # noqa: E402


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


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_scan_success_text_format(
    mock_embed, mock_model_info, temp_assignments_dir, capsys
):
    """Test a successful CLI scan with plain text format."""
    exit_code = run_scan(str(temp_assignments_dir), threshold=0.8, output_format="text")

    assert exit_code == 0
    captured = capsys.readouterr()

    assert "Documents Processed: 2" in captured.out
    assert "Similarity Threshold: 0.8" in captured.out
    assert "Matches Found:" in captured.out
    assert "- doc1.txt <-> doc2.txt: 1.0000" in captured.out


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_scan_success_csv_format(
    mock_embed, mock_model_info, temp_assignments_dir, capsys
):
    """Test a successful CLI scan with CSV format including metadata headers (#2991)."""
    exit_code = run_scan(str(temp_assignments_dir), threshold=0.8, output_format="csv")

    assert exit_code == 0
    captured = capsys.readouterr()

    # Filter out commented metadata lines for parsing validation, or inspect metadata explicitly
    lines = [line for line in captured.out.strip().split("\n") if line.strip()]

    # Verify metadata header lines start with '#'
    assert lines[0].startswith("#")
    assert any("Threshold Used: 0.8" in line for line in lines)

    # Find the row index where standard CSV headers begin
    header_idx = next(i for i, line in enumerate(lines) if not line.startswith("#"))

    assert lines[header_idx] == "doc_a,doc_b,similarity_score"
    assert lines[header_idx + 1] == "doc1.txt,doc2.txt,1.0"


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

    exit_code = run_scan(str(d), threshold=0.8, output_format="json")
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
def test_cli_prewarm_folder_success(
    mock_embed, mock_model_info, temp_assignments_dir, capsys
):
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
    "src.db.corpus_db.get_all_documents",
    return_value=[],
)
def test_cli_prewarm_no_documents(mock_docs, capsys):
    """Test prewarming cache when no documents are found."""
    exit_code = run_prewarm()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No documents found. Exiting.\n" in captured.out


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks
)
def test_cli_main_prewarm_command(
    mock_embed, mock_model_info, temp_assignments_dir, capsys
):
    """Test main CLI invocation with prewarm subcommand."""
    with patch(
        "sys.argv", ["cli.py", "prewarm", "--folder", str(temp_assignments_dir)]
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_cli_main_invalid_threshold():
    """Test main function with invalid threshold range."""
    with patch("sys.argv", ["cli.py", "scan", "./assignments", "--threshold", "1.5"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1


def test_cli_scan_default_threshold(temp_assignments_dir):
    """Test that run_scan defaults to PLAGIARISM_THRESHOLD when threshold argument is omitted."""
    from src.core.similarity import PLAGIARISM_THRESHOLD

    with patch("src.cli.run_scan") as mock_run_scan:
        mock_run_scan.return_value = 0
        with patch("sys.argv", ["cli.py", "scan", str(temp_assignments_dir)]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
            mock_run_scan.assert_called_once_with(
                str(temp_assignments_dir),
                PLAGIARISM_THRESHOLD,
                output_format="text",
            )


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
def test_cli_main_scan_format(
    mock_embed, mock_model_info, temp_assignments_dir, capsys
):
    """Test main function with scan subcommand specifying output format."""
    with patch(
        "sys.argv",
        ["cli.py", "scan", str(temp_assignments_dir), "--output-format", "json"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["documents_processed"] == 2


def test_cli_main_invalid_output_format(temp_assignments_dir, capsys):
    """Test main function with an invalid output format argument (xml)."""
    with patch(
        "sys.argv",
        ["cli.py", "scan", str(temp_assignments_dir), "--output-format", "xml"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        # argparse exits with code 2 for invalid choices
        assert excinfo.value.code == 2
        
    captured = capsys.readouterr()
    assert "invalid choice: 'xml'" in captured.err


def _normalized_sql(sql: str | None) -> str:
    """Normalize SQLite DDL for stable schema comparisons."""
    return " ".join((sql or "").split()).casefold()


def _database_schema_snapshot(db_path: Path) -> dict:
    """Return tables, columns, indexes, foreign keys, and user version."""
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        table_rows = connection.execute("""
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """).fetchall()

        snapshot = {
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
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
        f"Seed generation failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
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


# ─── Tests for Database Schema Verification (Issue #1494) ──────────────────────

from pathlib import Path

import pytest

from src.db.migrations.common import verify_schema_integrity


class TestVerifySchemaIntegrity:
    """Test suite for database schema verification helper."""

    def test_valid_schema_returns_true(self, tmp_path):
        """A database with all expected tables should return True."""
        db_path = tmp_path / "valid.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        expected = ["documents", "chunks"]
        assert verify_schema_integrity(db_path, expected) is True

    def test_missing_table_returns_false(self, tmp_path, caplog):
        """A database missing an expected table should return False."""
        db_path = tmp_path / "missing.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
        # Missing 'chunks' table
        conn.commit()
        conn.close()

        expected = ["documents", "chunks"]

        import logging

        with caplog.at_level(logging.ERROR):
            result = verify_schema_integrity(db_path, expected)

        assert result is False
        assert "MISSING tables" in caplog.text
        assert "chunks" in caplog.text

    def test_unexpected_table_returns_false(self, tmp_path, caplog):
        """A database with unexpected tables should return False."""
        db_path = tmp_path / "extra.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE legacy_table (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        expected = ["documents"]

        import logging

        with caplog.at_level(logging.WARNING):
            result = verify_schema_integrity(db_path, expected)

        assert result is False
        assert "UNEXPECTED tables" in caplog.text
        assert "legacy_table" in caplog.text

    def test_nonexistent_file_raises_filenotfound(self, tmp_path):
        """A non-existent database path should raise FileNotFoundError."""
        db_path = tmp_path / "nonexistent.db"
        expected = ["documents"]

        with pytest.raises(FileNotFoundError):
            verify_schema_integrity(db_path, expected)

    def test_directory_path_raises_isadirectory(self, tmp_path):
        """A directory path instead of a file should raise IsADirectoryError."""
        db_dir = tmp_path / "not_a_file"
        db_dir.mkdir()
        expected = ["documents"]

        with pytest.raises(IsADirectoryError):
            verify_schema_integrity(db_dir, expected)

    def test_invalid_sqlite_file_raises_databaseerror(self, tmp_path):
        """A file that is not a valid SQLite database should raise DatabaseError."""
        db_path = tmp_path / "invalid.db"
        db_path.write_text("This is not a SQLite database file.")
        expected = ["documents"]

        with pytest.raises(sqlite3.DatabaseError):
            verify_schema_integrity(db_path, expected)

    def test_case_insensitive_table_matching(self, tmp_path):
        """Table name comparison should be case-insensitive."""
        db_path = tmp_path / "case.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE Documents (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        expected = ["documents"]  # lowercase
        assert verify_schema_integrity(db_path, expected) is True

    def test_excludes_sqlite_internal_tables(self, tmp_path):
        """Internal SQLite tables (sqlite_*) should be ignored."""
        db_path = tmp_path / "internal.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        # AUTOINCREMENT creates sqlite_sequence automatically
        conn.execute("INSERT INTO documents DEFAULT VALUES")
        conn.commit()
        conn.close()

        expected = ["documents"]
        # Should pass even though sqlite_sequence exists
        assert verify_schema_integrity(db_path, expected) is True

    def test_empty_expected_list_requires_empty_db(self, tmp_path):
        """If expected list is empty, DB must have no user tables."""
        db_path = tmp_path / "empty_expected.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        expected = []
        # Should fail because 'documents' is unexpected
        assert verify_schema_integrity(db_path, expected) is False

    def test_whitespace_in_table_names_stripped(self, tmp_path):
        """Whitespace in expected table names should be stripped."""
        db_path = tmp_path / "whitespace.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        expected = ["  documents  ", ""]  # Empty string should be ignored
        assert verify_schema_integrity(db_path, expected) is True


# ─── CLI Integration Tests for --verify-schema ────────────────────────────────


def test_cli_main_verify_schema_success(tmp_path, capsys):
    """Test main CLI invocation with --verify-schema flag on a valid DB."""
    db_path = tmp_path / "valid_cli.db"
    conn = sqlite3.connect(db_path)
    for table in [
        "documents",
        "chunks",
        "deleted_chunks",
        "plagiarism_incidents",
        "false_positives",
    ]:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    with patch("sys.argv", ["cli.py", "--verify-schema", str(db_path)]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "PASSED" in captured.out


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks",
    side_effect=MockDataFactory.embed_chunks,
)
def test_cli_scan_recursive(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """Test scanning documents in nested subdirectories."""
    nested_dir = temp_assignments_dir / "student_1" / "assignment"
    nested_dir.mkdir(parents=True)

    (nested_dir / "doc3.txt").write_text("This is nested assignment one.")
    (nested_dir / "doc4.txt").write_text("This is nested assignment two.")

    exit_code = run_scan(
        str(temp_assignments_dir),
        threshold=0.8,
        output_format="json",
        recursive=True,
    )

    assert exit_code == 0

    report = json.loads(capsys.readouterr().out)
    assert report["documents_processed"] == 4


def test_cli_main_verify_schema_failure(tmp_path, capsys):
    """Test main CLI invocation with --verify-schema flag on an invalid DB."""
    db_path = tmp_path / "invalid_cli.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
    # Missing other expected tables
    conn.commit()
    conn.close()

    with patch("sys.argv", ["cli.py", "--verify-schema", str(db_path)]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "FAILED" in captured.out


def test_cli_main_verify_schema_nonexistent_file(capsys):
    """Test --verify-schema with a non-existent file exits with code 1."""
    with patch("sys.argv", ["cli.py", "--verify-schema", "/nonexistent/path.db"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error" in captured.err


def test_cli_main_verify_schema_cannot_combine_with_subcommand(tmp_path):
    """Test that --verify-schema cannot be combined with a subcommand."""
    db_path = tmp_path / "test.db"
    db_path.touch()

    with patch("sys.argv", ["cli.py", "--verify-schema", str(db_path), "scan", "./"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        # argparse exits with 2 for argument errors
        assert excinfo.value.code == 2


def test_cli_main_scan_recursive(temp_assignments_dir):
    """Test the --recursive CLI flag."""
    with patch("src.cli.run_scan") as mock_run_scan:
        mock_run_scan.return_value = 0

        with patch(
            "sys.argv",
            ["cli.py", "scan", str(temp_assignments_dir), "--recursive"],
        ):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 0
            mock_run_scan.assert_called_once_with(
                str(temp_assignments_dir),
                PLAGIARISM_THRESHOLD,
                output_format="text",
                recursive=True,
            )


class TestNaturalSorting:
    """Test suite for natural file ordering in the CLI."""

    def test_natural_sort_key_orders_numeric_files(self):
        """Verify doc10 sorts after doc2, not before (lexicographical behaviour)."""
        files = ["doc10.pdf", "doc2.pdf", "doc1.pdf"]
        assert sorted(files, key=_natural_sort_key) == [
            "doc1.pdf",
            "doc2.pdf",
            "doc10.pdf",
        ]

    def test_natural_sort_key_mixed_paths(self):
        """Verify natural sorting works on full file paths."""
        files = [
            "/tmp/submission/doc20.txt",
            "/tmp/submission/doc3.txt",
            "/tmp/submission/doc10.txt",
        ]
        assert sorted(files, key=_natural_sort_key) == [
            "/tmp/submission/doc3.txt",
            "/tmp/submission/doc10.txt",
            "/tmp/submission/doc20.txt",
        ]

    def test_natural_sort_key_case_insensitive(self):
        """Verify sorting is case-insensitive for the text portions."""
        files = ["B.txt", "a.txt", "c.txt"]
        assert sorted(files, key=_natural_sort_key) == ["a.txt", "B.txt", "c.txt"]

    def test_natural_sort_key_stable_for_non_numeric(self):
        """Verify behaviour matches lexicographical order for non-numeric names."""
        files = ["readme.md", "report.pdf", "notes.txt"]
        assert sorted(files, key=_natural_sort_key) == sorted(files)



def test_cli_scan_permission_error(temp_assignments_dir, capsys):
    """
    Test that the CLI handles a PermissionError gracefully during directory scanning
    when os.scandir is mocked to raise a PermissionError.

    Specifically, this test simulates a scenario where the operating system prevents
    the application from scanning the contents of the target folder.
    
    Verifies that:
    1. The CLI exits with code 1.
    2. An appropriate error message indicating read failure is written to stderr.
    3. The application does not crash with an unhandled exception trace.
    """
    with patch("os.scandir") as mock_scandir:
        # Simulate OS PermissionError (Errno 13 - Permission Denied)
        mock_scandir.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "scan", str(temp_assignments_dir)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            # The CLI is expected to return status code 1 on directory read failures
            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error reading folder contents" in captured.err
    assert "Permission denied" in captured.err


def test_cli_scan_permission_error_iterdir(temp_assignments_dir, capsys):
    """
    Verify that if Path.iterdir itself throws a PermissionError when scanned
    (non-recursively), the main CLI tool intercepts the exception, prints
    the custom "Error reading folder contents" diagnostic to stderr, and exits
    with status code 1.
    
    This is another entry point for directory listings using pathlib instead of
    raw os.scandir, which is the default in non-recursive scan modes.
    """
    with patch("pathlib.Path.iterdir") as mock_iterdir:
        mock_iterdir.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "scan", str(temp_assignments_dir)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error reading folder contents" in captured.err
    assert "Permission denied" in captured.err


def test_cli_scan_permission_error_rglob(temp_assignments_dir, capsys):
    """
    Verify that if Path.rglob throws a PermissionError when scanning recursively,
    the main CLI tool intercepts the exception, prints the custom diagnostic
    to stderr, and exits with status code 1.

    This ensures that recursive folder scans are equally protected against
    unreadable subdirectories or files.
    """
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "scan", str(temp_assignments_dir), "--recursive"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error reading folder contents" in captured.err
    assert "Permission denied" in captured.err


@patch(
    "src.core.embedding_model.get_embedding_model_info",
    return_value=("all-MiniLM-L6-v2", 384),
)
@patch(
    "src.core.embedding_model.embed_chunks",
    side_effect=MockDataFactory.embed_chunks,
)
def test_cli_scan_file_permission_error_single(mock_embed, mock_model_info, temp_assignments_dir, capsys):
    """
    Verify that a file-level PermissionError during document loading is handled
    gracefully. It should print a warning or error to stderr for the unreadable
    file but continue processing the remaining readable documents.
    If at least two valid documents are successfully processed, the pipeline
    runs and the CLI exits with code 0.
    
    This matches production requirements where single file access failures should not
    prevent checking the rest of the available submissions.
    """
    # Create an additional file that we will mock as unreadable
    unreadable_file = temp_assignments_dir / "unreadable_doc.txt"
    unreadable_file.write_text("This file should trigger a PermissionError.")

    original_open = builtins.open

    def mocked_open(file, *args, **kwargs):
        if str(file).endswith("unreadable_doc.txt"):
            raise PermissionError(13, "Permission denied")
        return original_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=mocked_open):
        with patch("sys.argv", ["cli.py", "scan", str(temp_assignments_dir)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "Error processing unreadable_doc.txt" in captured.err
    assert "Permission denied" in captured.err


def test_cli_scan_file_permission_error_all_fail(temp_assignments_dir, capsys):
    """
    Verify that if ALL files in the scanned directory fail to open due to
    PermissionError, the CLI does not crash with an unhandled exception,
    but instead prints the diagnostic and exit code 1 because no valid
    documents were found to process.
    """
    with patch("builtins.open", side_effect=PermissionError(13, "Permission denied")):
        with patch("sys.argv", ["cli.py", "scan", str(temp_assignments_dir)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error processing" in captured.err
    assert "Permission denied" in captured.err
    assert "Error: No valid documents found to process" in captured.err


def test_cli_prewarm_permission_error(temp_assignments_dir, capsys):
    """
    Verify that when running the prewarm command with a folder path, if the directory
    cannot be scanned due to a PermissionError, the error is handled gracefully.
    The CLI must output the diagnostic message to stderr and return exit code 1.
    
    This ensures prewarming is protected against unreadable source directories.
    """
    with patch("os.scandir") as mock_scandir:
        mock_scandir.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "prewarm", "--folder", str(temp_assignments_dir)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error reading folder contents" in captured.err
    assert "Permission denied" in captured.err


def test_cli_db_status_permission_error(tmp_path, capsys):
    """
    Verify that the db-status command handles database permission errors gracefully.
    If the migration status checker cannot read the database file due to a
    PermissionError (represented as OSError), the CLI should report the error
    to stderr and exit with code 1.
    """
    db_path = tmp_path / "protected_corpus.db"
    db_path.touch()

    # Mock get_migration_status to raise PermissionError
    with patch("src.db.migrations.get_migration_status") as mock_status:
        mock_status.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "db-status", str(db_path)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "Permission denied" in captured.err


def test_cli_db_downgrade_permission_error(tmp_path, capsys):
    """
    Verify that the db downgrade command handles database permission/connection
    errors gracefully. If connecting to or reading the database raises a
    PermissionError/OSError, the CLI must report the failure to downgrade
    on stderr and return exit code 1.
    """
    db_path = tmp_path / "protected_rollback.db"
    db_path.touch()

    # Mock sqlite3.connect to raise PermissionError
    with patch("sqlite3.connect") as mock_connect:
        mock_connect.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "db", "downgrade", "--database", str(db_path)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error: Failed to downgrade database" in captured.err
    assert "Permission denied" in captured.err


def test_cli_verify_schema_permission_error(tmp_path, capsys):
    """
    Verify that the --verify-schema command handles permission errors during
    integrity check. If verify_schema_integrity raises a PermissionError,
    the CLI must print the error to stderr and exit with code 1.
    """
    db_path = tmp_path / "protected_schema.db"
    db_path.touch()

    # Mock verify_schema_integrity to raise PermissionError
    with patch("src.db.migrations.common.verify_schema_integrity") as mock_verify:
        mock_verify.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "--verify-schema", str(db_path)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error during schema verification" in captured.err
    assert "Permission denied" in captured.err


def test_cli_database_optimization_permission_error(tmp_path, capsys):
    """
    Verify that the --optimize database subcommand handles permission/write errors
    gracefully. If optimization fails due to file permissions (returning False
    from optimize_database), the CLI writes an error message to stderr and exits
    with code 1.
    """
    db_path = tmp_path / "protected_opt.db"

    # Mock optimize_database to return False, simulating a failure (e.g. read-only db)
    with patch("src.cli.optimize_database", return_value=False):
        with patch("sys.argv", ["cli.py", "--optimize", str(db_path)]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error: Database optimization failed" in captured.err


def test_cli_sync_index_permission_error(capsys):
    """
    Verify that the sync-index command handles FAISS index and DB synchronization
    errors gracefully. If the synchronization check raises a PermissionError (or any
    other exception due to file system permission issues), it writes the error message
    to stderr and exits with code 1.
    """
    # Mock verify_and_repair_index to raise PermissionError
    with patch("src.cli.verify_and_repair_index") as mock_sync:
        mock_sync.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "sync-index"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert "Error during synchronization" in captured.err
    assert "Permission denied" in captured.err


def test_cli_purge_cache_permission_error():
    """
    Verify that the purge-cache command correctly propagates a PermissionError
    if the translation cache database initialization fails due to insufficient
    file permissions.
    """
    with patch("src.cli.initialize_cache_db") as mock_init:
        mock_init.side_effect = PermissionError(13, "Permission denied")

        with patch("sys.argv", ["cli.py", "purge-cache"]):
            with pytest.raises(PermissionError) as excinfo:
                main()

            assert "Permission denied" in str(excinfo.value)
        
