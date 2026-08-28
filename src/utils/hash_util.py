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
