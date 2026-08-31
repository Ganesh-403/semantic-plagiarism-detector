from unittest.mock import patch

import pytest

from src.utils.hash_util import calculate_file_sha256


def test_calculate_file_sha256_file_not_found():
    with pytest.raises(ValueError, match="The specified file was not found"):
        calculate_file_sha256("nonexistent_file_12345.txt")


@patch("builtins.open", side_effect=PermissionError("Permission denied"))
def test_calculate_file_sha256_permission_error(mock_open):
    with pytest.raises(ValueError, match="Permission denied when accessing file"):
        calculate_file_sha256("restricted_file.txt")
