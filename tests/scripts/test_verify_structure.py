"""
tests/scripts/test_verify_structure.py
--------------------------------------
Unit tests for the directory structure verification script.

Validates:
- Directory existence checking
- __init__.py file verification
- Exit code logic
- Missing path reporting
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import verify_structure


class TestVerifyProjectStructure:
    """Test suite for the core verification logic."""

    def test_valid_structure_returns_true(self, tmp_path):
        """A complete valid structure should return True and empty missing list."""
        # Create all required directories
        for dir_name in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_name).mkdir(parents=True, exist_ok=True)
            
        # Create all required __init__.py files
        for init_file in verify_structure.REQUIRED_INIT_FILES:
            (tmp_path / init_file).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / init_file).touch()
            
        # Create required files
        for file_name in verify_structure.REQUIRED_FILES:
            (tmp_path / file_name).touch()
            
        is_valid, missing = verify_structure.verify_project_structure(
            root_dir=tmp_path,
            check_dirs=verify_structure.REQUIRED_DIRECTORIES,
            check_inits=verify_structure.REQUIRED_INIT_FILES,
            check_files=verify_structure.REQUIRED_FILES,
        )
        
        assert is_valid is True
        assert len(missing) == 0

    def test_missing_directory_returns_false(self, tmp_path):
        """A missing directory should return False and list the missing path."""
        # Create everything EXCEPT src/core
        for dir_name in verify_structure.REQUIRED_DIRECTORIES:
            if dir_name == "src/core":
                continue
            (tmp_path / dir_name).mkdir(parents=True, exist_ok=True)
            
        for init_file in verify_structure.REQUIRED_INIT_FILES:
            (tmp_path / init_file).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / init_file).touch()
            
        for file_name in verify_structure.REQUIRED_FILES:
            (tmp_path / file_name).touch()
            
        is_valid, missing = verify_structure.verify_project_structure(
            root_dir=tmp_path,
            check_dirs=verify_structure.REQUIRED_DIRECTORIES,
            check_inits=verify_structure.REQUIRED_INIT_FILES,
            check_files=verify_structure.REQUIRED_FILES,
        )
        
        assert is_valid is False
        assert len(missing) == 1
        assert "src/core" in missing[0]

    def test_missing_init_file_returns_false(self, tmp_path):
        """A missing __init__.py should return False."""
        for dir_name in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_name).mkdir(parents=True, exist_ok=True)
            
        # Create all init files EXCEPT src/core/__init__.py
        for init_file in verify_structure.REQUIRED_INIT_FILES:
            if init_file == "src/core/__init__.py":
                continue
            (tmp_path / init_file).touch()
            
        for file_name in verify_structure.REQUIRED_FILES:
            (tmp_path / file_name).touch()
            
        is_valid, missing = verify_structure.verify_project_structure(
            root_dir=tmp_path,
            check_dirs=verify_structure.REQUIRED_DIRECTORIES,
            check_inits=verify_structure.REQUIRED_INIT_FILES,
            check_files=verify_structure.REQUIRED_FILES,
        )
        
        assert is_valid is False
        assert any("src/core/__init__.py" in m for m in missing)

    def test_file_instead_of_directory(self, tmp_path):
        """If a required directory path is actually a file, it should be flagged."""
        # Create src as a FILE instead of a directory
        (tmp_path / "src").touch()
        
        is_valid, missing = verify_structure.verify_project_structure(
            root_dir=tmp_path,
            check_dirs=["src"],
            check_inits=[],
            check_files=[],
        )
        
        assert is_valid is False
        assert any("is file" in m for m in missing)

    def test_multiple_missing_paths(self, tmp_path):
        """Verify all missing paths are reported, not just the first one."""
        # Create nothing
        is_valid, missing = verify_structure.verify_project_structure(
            root_dir=tmp_path,
            check_dirs=["src", "app", "tests"],
            check_inits=[],
            check_files=[],
        )
        
        assert is_valid is False
        assert len(missing) == 3


class TestMainExecution:
    """Test suite for the main() entry point and exit codes."""

    def test_main_returns_zero_on_success(self, tmp_path):
        """main() should return 0 when structure is valid."""
        # Setup valid structure
        for dir_name in verify_structure.REQUIRED_DIRECTORIES:
            (tmp_path / dir_name).mkdir(parents=True, exist_ok=True)
        for init_file in verify_structure.REQUIRED_INIT_FILES:
            (tmp_path / init_file).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / init_file).touch()
        for file_name in verify_structure.REQUIRED_FILES:
            (tmp_path / file_name).touch()
            
        with patch("sys.argv", ["verify_structure.py", "--root-dir", str(tmp_path)]):
            exit_code = verify_structure.main()
            
        assert exit_code == 0

    def test_main_returns_one_on_failure(self, tmp_path):
        """main() should return 1 when structure is invalid."""
        # Empty directory
        with patch("sys.argv", ["verify_structure.py", "--root-dir", str(tmp_path)]):
            exit_code = verify_structure.main()
            
        assert exit_code == 1

    def test_main_returns_one_for_nonexistent_root(self, tmp_path):
        """main() should return 1 if the specified root directory doesn't exist."""
        fake_path = tmp_path / "nonexistent"
        with patch("sys.argv", ["verify_structure.py", "--root-dir", str(fake_path)]):
            exit_code = verify_structure.main()
            
        assert exit_code == 1


class TestReporting:
    """Test suite for output formatting."""

    def test_print_verification_report_success(self, capsys):
        """Verify success report contains checkmark and success message."""
        verify_structure.print_verification_report(is_valid=True, missing_paths=[])
        
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out
        assert "✅" in captured.out

    def test_print_verification_report_failure(self, capsys):
        """Verify failure report lists all missing paths."""
        missing = ["DIR: src/core", "FILE: requirements.txt"]
        verify_structure.print_verification_report(is_valid=False, missing_paths=missing)
        
        captured = capsys.readouterr()
        assert "FAILURE" in captured.out
        assert "❌" in captured.out
        assert "src/core" in captured.out
        assert "requirements.txt" in captured.out
