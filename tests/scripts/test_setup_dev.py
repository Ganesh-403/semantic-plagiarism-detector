import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import setup_dev


class TestSetupDev:
    """Test suite for developer setup script."""

    # Test 1 - Python version check
    def test_python_version_check_passes(self):
        """Python >= 3.10 passes."""
        with patch("setup_dev.sys") as mock_sys:
            mock_sys.version_info = (3, 10, 0)
            assert setup_dev.check_python_version() is True

    def test_python_version_check_fails(self):
        """Python < 3.10 fails."""
        import collections
        VersionInfo = collections.namedtuple('VersionInfo', ['major', 'minor', 'micro', 'releaselevel', 'serial'])
        with patch("setup_dev.sys") as mock_sys:
            mock_sys.version_info = VersionInfo(3, 9, 5, 'final', 0)
            assert setup_dev.check_python_version() is False

    # Test 2 - Directory creation
    def test_directory_creation(self, tmp_path):
        """Verify data/ and logs/ are created when missing."""
        with patch("setup_dev.PROJECT_ROOT", tmp_path):
            assert setup_dev.create_directories() is True
            assert (tmp_path / "data").exists()
            assert (tmp_path / "data").is_dir()
            assert (tmp_path / "logs").exists()
            assert (tmp_path / "logs").is_dir()

    # Test 3 - Existing directories
    def test_existing_directories(self, tmp_path):
        """Verify running setup does not fail when directories already exist."""
        (tmp_path / "data").mkdir()
        (tmp_path / "logs").mkdir()
        
        # Create a file inside to ensure they are not deleted/overwritten destructively
        test_file = tmp_path / "data" / "test.txt"
        test_file.write_text("keep me")

        with patch("setup_dev.PROJECT_ROOT", tmp_path):
            assert setup_dev.create_directories() is True
            assert test_file.exists()
            assert test_file.read_text() == "keep me"

    # Test 4 - .env creation
    def test_env_creation_from_example(self, tmp_path):
        """.env is created and contents match .env.example when .env is missing."""
        example_env = tmp_path / ".env.example"
        example_env.write_text("TEST_VAR=123")

        with patch("setup_dev.PROJECT_ROOT", tmp_path):
            assert setup_dev.setup_env() is True
            
        new_env = tmp_path / ".env"
        assert new_env.exists()
        assert new_env.read_text() == "TEST_VAR=123"

    def test_env_missing_example(self, tmp_path, capsys):
        """Fails gracefully if .env.example is missing."""
        with patch("setup_dev.PROJECT_ROOT", tmp_path):
            assert setup_dev.setup_env() is False
        
        captured = capsys.readouterr()
        assert "[WARN]" in captured.out

    # Test 5 - Existing .env
    def test_existing_env_not_overwritten(self, tmp_path):
        """Verify an existing .env is NOT overwritten."""
        example_env = tmp_path / ".env.example"
        example_env.write_text("TEST_VAR=123")
        
        existing_env = tmp_path / ".env"
        existing_env.write_text("TEST_VAR=456")

        with patch("setup_dev.PROJECT_ROOT", tmp_path):
            assert setup_dev.setup_env() is True
            
        assert existing_env.read_text() == "TEST_VAR=456"

    # Test 6 - Tesseract detection
    @patch("setup_dev.shutil.which")
    @patch("setup_dev.subprocess.run")
    def test_tesseract_detection_installed(self, mock_run, mock_which):
        """Tesseract installed is handled correctly."""
        mock_which.return_value = "/usr/bin/tesseract"
        
        # Mock successful tesseract --version
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "tesseract 5.0.0-alpha"
        mock_run.return_value = mock_result
        
        assert setup_dev.check_tesseract() is True
        mock_run.assert_called_once()

    @patch("setup_dev.shutil.which")
    def test_tesseract_detection_missing(self, mock_which):
        """Tesseract missing is handled correctly."""
        mock_which.return_value = None
        
        assert setup_dev.check_tesseract() is False

    # Test 7 - NLTK Download
    @patch("nltk.download")
    def test_setup_nltk_success(self, mock_download):
        """Verify NLTK returns True when all downloads succeed."""
        mock_download.return_value = True
        assert setup_dev.setup_nltk() is True
        assert mock_download.call_count == 6

    @patch("nltk.download")
    def test_setup_nltk_failure(self, mock_download, capsys):
        """Verify NLTK returns False and fails immediately on a download failure."""
        # Fail on the first download
        mock_download.return_value = False
        assert setup_dev.setup_nltk() is False
        assert mock_download.call_count == 1
        
        captured = capsys.readouterr()
        assert "[FAIL] Failed to download NLTK corpus:" in captured.out

    # Test 7 - Database initialization
    def test_database_initialization_success(self):
        """Verify explicit database initializations are called."""
        # Using patch on setup_dev is slightly complex because they are imported inside the function,
        # so we will patch them at their source
        with patch("src.db.auth.init_db") as m_auth, \
             patch("src.db.corpus_db.init_corpus_db") as m_corpus, \
             patch("src.db.incidents.init_incident_db") as m_inc:
            assert setup_dev.initialize_databases() is True
            m_auth.assert_called_once()
            m_corpus.assert_called_once()
            m_inc.assert_called_once()

    def test_database_initialization_failure(self, capsys):
        """Verify failure in one database initializer fails the step."""
        with patch("src.db.auth.init_db") as m_auth:
            m_auth.side_effect = Exception("Auth DB Error")
            assert setup_dev.initialize_databases() is False
        
        captured = capsys.readouterr()
        assert "[FAIL] Exception during DB initialization: Auth DB Error" in captured.out

    # Test 8 - Readiness reporting
    def test_readiness_reporting_success(self, capsys):
        """Verify the final report correctly distinguishes successful checks."""
        results = {
            'python': True,
            'deps': True,
            'nltk': True,
            'dirs': True,
            'env': True,
            'db': True,
            'tesseract': True
        }
        setup_dev.print_readiness_report(results)
        captured = capsys.readouterr()
        
        assert "Environment is ready for development." in captured.out
        assert "[OK] Python" in captured.out
        assert "[OK] Dependencies" in captured.out
        assert "[FAIL]" not in captured.out
        assert "[WARN]" not in captured.out

    def test_readiness_reporting_warnings_and_failures(self, capsys):
        """Verify the final report correctly handles warnings and failures."""
        results = {
            'python': True,
            'deps': False, # Critical failure
            'nltk': True,
            'dirs': True,
            'env': False, # Warning
            'db': True,
            'tesseract': False # Warning
        }
        setup_dev.print_readiness_report(results)
        captured = capsys.readouterr()
        
        assert "Environment is NOT ready" in captured.out
        assert "[FAIL] Dependencies" in captured.out
        assert "[WARN] .env configured" in captured.out
        assert "[WARN] Tesseract detected" in captured.out
