# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
tests/scripts/test_generate_seed_data.py
----------------------------------------
Unit tests for the seed data generation script.

Validates seed data creation, dry-run mode, and error handling.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_seed_data


class TestGenerateSeedData:
    """Test suite for generate_seed_data() function."""

    def test_creates_seed_directory(self, tmp_path):
        """Verify seed directory is created if it doesn't exist."""
        seed_dir = tmp_path / "seeds"
        assert not seed_dir.exists()

        with patch("generate_seed_data.init_db"), patch(
            "generate_seed_data.init_corpus_db"
        ), patch("generate_seed_data.init_incident_db"), patch(
            "generate_seed_data.add_user"
        ), patch("generate_seed_data.add_document"), patch(
            "generate_seed_data.record_plagiarism_incident"
        ):
            generate_seed_data.generate_seed_data(seed_dir, dry_run=False)

        assert seed_dir.exists()
        assert seed_dir.is_dir()

    @patch("generate_seed_data.add_user")
    @patch("generate_seed_data.add_document")
    @patch("generate_seed_data.record_plagiarism_incident")
    @patch("generate_seed_data.init_db")
    @patch("generate_seed_data.init_corpus_db")
    @patch("generate_seed_data.init_incident_db")
    def test_creates_users(
        self,
        mock_init_inc,
        mock_init_corp,
        mock_init_auth,
        mock_record,
        mock_add_doc,
        mock_add_user,
        tmp_path,
    ):
        """Verify seed users are created."""
        seed_dir = tmp_path / "seeds"
        summary = generate_seed_data.generate_seed_data(seed_dir, dry_run=False)

        assert summary["users_created"] == len(generate_seed_data.SEED_USERS)
        assert mock_add_user.call_count == len(generate_seed_data.SEED_USERS)

    @patch("generate_seed_data.add_user")
    @patch("generate_seed_data.add_document")
    @patch("generate_seed_data.record_plagiarism_incident")
    @patch("generate_seed_data.init_db")
    @patch("generate_seed_data.init_corpus_db")
    @patch("generate_seed_data.init_incident_db")
    def test_creates_documents(
        self,
        mock_init_inc,
        mock_init_corp,
        mock_init_auth,
        mock_record,
        mock_add_doc,
        mock_add_user,
        tmp_path,
    ):
        """Verify seed documents are created."""
        seed_dir = tmp_path / "seeds"
        summary = generate_seed_data.generate_seed_data(seed_dir, dry_run=False)

        assert summary["documents_created"] == len(generate_seed_data.SEED_DOCUMENTS)
        assert mock_add_doc.call_count == len(generate_seed_data.SEED_DOCUMENTS)

        # Verify files were written to disk
        for doc_data in generate_seed_data.SEED_DOCUMENTS:
            file_path = seed_dir / doc_data["filename"]
            assert file_path.exists()
            assert file_path.read_text() == doc_data["content"]

    @patch("generate_seed_data.add_user")
    @patch("generate_seed_data.add_document")
    @patch("generate_seed_data.record_plagiarism_incident")
    @patch("generate_seed_data.init_db")
    @patch("generate_seed_data.init_corpus_db")
    @patch("generate_seed_data.init_incident_db")
    def test_creates_incidents(
        self,
        mock_init_inc,
        mock_init_corp,
        mock_init_auth,
        mock_record,
        mock_add_doc,
        mock_add_user,
        tmp_path,
    ):
        """Verify seed incidents are created."""
        seed_dir = tmp_path / "seeds"
        summary = generate_seed_data.generate_seed_data(seed_dir, dry_run=False)

        assert summary["incidents_created"] == len(generate_seed_data.SEED_INCIDENTS)
        assert mock_record.call_count == len(generate_seed_data.SEED_INCIDENTS)


class TestDryRunMode:
    """Test suite for --dry-run flag functionality (Issue #2020)."""

    @patch("generate_seed_data.init_db")
    @patch("generate_seed_data.init_corpus_db")
    @patch("generate_seed_data.init_incident_db")
    @patch("generate_seed_data.add_user")
    @patch("generate_seed_data.add_document")
    @patch("generate_seed_data.record_plagiarism_incident")
    def test_dry_run_skips_db_initialization(
        self,
        mock_record,
        mock_add_doc,
        mock_add_user,
        mock_init_inc,
        mock_init_corp,
        mock_init_auth,
        tmp_path,
    ):
        """Verify dry-run mode skips database initialization."""
        seed_dir = tmp_path / "seeds"
        generate_seed_data.generate_seed_data(seed_dir, dry_run=True)

        # DB init should not be called in dry-run mode
        mock_init_auth.assert_not_called()
        mock_init_corp.assert_not_called()
        mock_init_inc.assert_not_called()

    @patch("generate_seed_data.init_db")
    @patch("generate_seed_data.init_corpus_db")
    @patch("generate_seed_data.init_incident_db")
    @patch("generate_seed_data.add_user")
    @patch("generate_seed_data.add_document")
    @patch("generate_seed_data.record_plagiarism_incident")
    def test_dry_run_skips_db_writes(
        self,
        mock_record,
        mock_add_doc,
        mock_add_user,
        mock_init_inc,
        mock_init_corp,
        mock_init_auth,
        tmp_path,
    ):
        """Verify dry-run mode skips all database writes."""
        seed_dir = tmp_path / "seeds"
        generate_seed_data.generate_seed_data(seed_dir, dry_run=True)

        # No DB writes should occur
        mock_add_user.assert_not_called()
        mock_add_doc.assert_not_called()
        mock_record.assert_not_called()

    @patch("generate_seed_data.init_db")
    @patch("generate_seed_data.init_corpus_db")
    @patch("generate_seed_data.init_incident_db")
    @patch("generate_seed_data.add_user")
    @patch("generate_seed_data.add_document")
    @patch("generate_seed_data.record_plagiarism_incident")
    def test_dry_run_returns_correct_summary(
        self,
        mock_record,
        mock_add_doc,
        mock_add_user,
        mock_init_inc,
        mock_init_corp,
        mock_init_auth,
        tmp_path,
    ):
        """Verify dry-run returns correct summary counts."""
        seed_dir = tmp_path / "seeds"
        summary = generate_seed_data.generate_seed_data(seed_dir, dry_run=True)

        assert summary["dry_run"] is True
        assert summary["users_created"] == len(generate_seed_data.SEED_USERS)
        assert summary["documents_created"] == len(generate_seed_data.SEED_DOCUMENTS)
        assert summary["incidents_created"] == len(generate_seed_data.SEED_INCIDENTS)

    @patch("generate_seed_data.init_db")
    @patch("generate_seed_data.init_corpus_db")
    @patch("generate_seed_data.init_incident_db")
    @patch("generate_seed_data.add_user")
    @patch("generate_seed_data.add_document")
    @patch("generate_seed_data.record_plagiarism_incident")
    def test_dry_run_does_not_create_files(
        self,
        mock_record,
        mock_add_doc,
        mock_add_user,
        mock_init_inc,
        mock_init_corp,
        mock_init_auth,
        tmp_path,
    ):
        """Verify dry-run mode does not create seed files on disk."""
        seed_dir = tmp_path / "seeds"
        generate_seed_data.generate_seed_data(seed_dir, dry_run=True)

        # Directory should be created but no files inside
        assert seed_dir.exists()
        files = list(seed_dir.glob("*.txt"))
        assert len(files) == 0

    def test_dry_run_logs_operations(self, tmp_path, caplog):
        """Verify dry-run mode logs all operations with [DRY RUN] prefix."""
        import logging

        seed_dir = tmp_path / "seeds"

        with caplog.at_level(logging.INFO):
            generate_seed_data.generate_seed_data(seed_dir, dry_run=True)

        # Check for dry-run log messages
        log_messages = [record.message for record in caplog.records]

        assert any("[DRY RUN]" in msg for msg in log_messages)
        assert any(
            "Would create" in msg or "Would upload" in msg for msg in log_messages
        )
        assert any("No database modifications" in msg for msg in log_messages)


class TestArgumentParsing:
    """Test suite for command line argument parsing."""

    def test_parse_default_arguments(self):
        """Verify default argument values."""
        with patch("sys.argv", ["generate_seed_data.py"]):
            args = generate_seed_data.parse_arguments()

        assert args.seed_dir == generate_seed_data.ROOT_DIR / "data" / "seeds"
        assert args.dry_run is False

    def test_parse_dry_run_flag(self):
        """Verify --dry-run flag is parsed correctly."""
        with patch("sys.argv", ["generate_seed_data.py", "--dry-run"]):
            args = generate_seed_data.parse_arguments()

        assert args.dry_run is True

    def test_parse_custom_seed_dir(self, tmp_path):
        """Verify --seed-dir argument is parsed correctly."""
        custom_dir = tmp_path / "custom_seeds"

        with patch(
            "sys.argv", ["generate_seed_data.py", "--seed-dir", str(custom_dir)]
        ):
            args = generate_seed_data.parse_arguments()

        assert args.seed_dir == custom_dir

    def test_parse_combined_arguments(self, tmp_path):
        """Verify multiple arguments can be combined."""
        custom_dir = tmp_path / "custom_seeds"

        with patch(
            "sys.argv",
            ["generate_seed_data.py", "--seed-dir", str(custom_dir), "--dry-run"],
        ):
            args = generate_seed_data.parse_arguments()

        assert args.seed_dir == custom_dir
        assert args.dry_run is True


class TestMainFunction:
    """Test suite for main() entry point."""

    @patch("generate_seed_data.generate_seed_data")
    def test_main_returns_zero_on_success(self, mock_generate, tmp_path):
        """Verify main() returns 0 on successful execution."""
        mock_generate.return_value = {
            "users_created": 2,
            "documents_created": 3,
            "incidents_created": 2,
            "dry_run": False,
        }

        with patch("sys.argv", ["generate_seed_data.py", "--seed-dir", str(tmp_path)]):
            exit_code = generate_seed_data.main()

        assert exit_code == 0

    @patch("generate_seed_data.generate_seed_data")
    def test_main_returns_one_on_failure(self, mock_generate, tmp_path):
        """Verify main() returns 1 on failure."""
        mock_generate.side_effect = RuntimeError("Database error")

        with patch("sys.argv", ["generate_seed_data.py", "--seed-dir", str(tmp_path)]):
            exit_code = generate_seed_data.main()

        assert exit_code == 1

    @patch("generate_seed_data.generate_seed_data")
    def test_main_with_dry_run_flag(self, mock_generate, tmp_path):
        """Verify main() passes dry_run flag to generate_seed_data."""
        mock_generate.return_value = {
            "users_created": 2,
            "documents_created": 3,
            "incidents_created": 2,
            "dry_run": True,
        }

        with patch(
            "sys.argv",
            ["generate_seed_data.py", "--seed-dir", str(tmp_path), "--dry-run"],
        ):
            exit_code = generate_seed_data.main()

        assert exit_code == 0
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["dry_run"] is True
