from unittest.mock import patch
from scripts.verify_structure import main

def test_verify_structure_success(tmp_path):
    # Create required structure
    for d in ["src/core", "src/db", "app", "tests", "config"]:
        (tmp_path / d).mkdir(parents=True)

    for f in ["src/__init__.py", "src/core/__init__.py", "src/db/__init__.py"]:
        (tmp_path / f).touch()

    with patch("scripts.verify_structure.Path") as mock_path, \
         patch("sys.exit") as mock_exit, \
         patch("builtins.print") as mock_print:

        # Make the resolved parent point to our tmp_path
        mock_path.return_value.resolve.return_value.parent.parent = tmp_path
        mock_exit.side_effect = SystemExit

        try:
            main()
        except SystemExit:
            pass

        mock_exit.assert_called_once_with(0)
        mock_print.assert_called_with("Project directory structure verified successfully.")

def test_verify_structure_missing(tmp_path):
    # Empty directory, everything is missing
    with patch("scripts.verify_structure.Path") as mock_path, \
         patch("sys.exit") as mock_exit, \
         patch("builtins.print") as mock_print:

        mock_path.return_value.resolve.return_value.parent.parent = tmp_path
        mock_exit.side_effect = SystemExit

        try:
            main()
        except SystemExit:
            pass

        mock_exit.assert_called_once_with(1)
        mock_print.assert_any_call("Verification failed. The following items are missing:")
