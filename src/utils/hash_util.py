import hashlib
import logging

logger = logging.getLogger(__name__)


def calculate_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str | None:
    """
    Calculate the SHA256 hash of a file efficiently by reading it in chunks.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Size of the chunks to read (default 1MB).

    Returns:
        The hexadecimal SHA256 hash string.

    Raises:
        ValueError: If the file is not found or permission is denied.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise ValueError(f"The specified file was not found: {file_path}") from e
    except PermissionError as e:
        logger.error(f"Permission denied accessing file: {file_path}")
        raise ValueError(f"Permission denied when accessing file: {file_path}") from e
