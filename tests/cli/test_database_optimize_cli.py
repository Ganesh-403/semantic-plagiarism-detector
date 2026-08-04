"""CLI tests for Issue #831 --optimize support."""

from unittest.mock import patch

import pytest

from src.cli import main, run_database_optimization


def test_run_database_optimization_success(capsys):
    with patch("src.cli.optimize_database", return_value=True) as optimize:
        exit_code = run_database_optimization("corpus.db")

    assert exit_code == 0
    optimize.assert_called_once_with("corpus.db")
    assert "Database optimized successfully" in capsys.readouterr().out


def test_run_database_optimization_failure(capsys):
    with patch("src.cli.optimize_database", return_value=False):
        exit_code = run_database_optimization("missing.db")

    assert exit_code == 1
    assert "Database optimization failed" in capsys.readouterr().err


def test_main_optimize_flag_dispatches_and_exits_zero():
    with patch("sys.argv", ["cli.py", "--optimize", "corpus.db"]):
        with patch(
            "src.cli.run_database_optimization",
            return_value=0,
        ) as runner:
            with pytest.raises(SystemExit) as exc_info:
                main()

    assert exc_info.value.code == 0
    runner.assert_called_once_with("corpus.db")


def test_main_rejects_optimize_with_subcommand():
    with patch(
        "sys.argv",
        ["cli.py", "--optimize", "corpus.db", "sync-index"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 2


def test_main_requires_command_or_optimize():
    with patch("sys.argv", ["cli.py"]):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 2
