import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import MockDataFactory

# Mock ML libraries to prevent pytest segmentation faults on Apple Silicon
sys.modules["transformers"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from src.cli import main, run_scan  # noqa: E402


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
    """Test a successful CLI scan on a directory with valid documents."""
    exit_code = run_scan(str(temp_assignments_dir), threshold=0.8)

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
