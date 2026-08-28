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
exceptions.py
--------------
Centralized Exception classes used across the application.

These were previously defined in src/errors.py, which is reserved for
string message constants. Keeping Exception classes in their own module
avoids circular imports and keeps errors.py focused on message text.
"""

__all__ = [
    "ExportFailedError",
    "ModelInitializationError",
    "OCRFileBatchError",
    "StaleDataException",
    "UnsupportedFormatError",
]


class ExportFailedError(RuntimeError):
    """Raised when an export cannot be generated or written safely."""


class ModelInitializationError(RuntimeError):
    """Raised when neither the primary nor fallback embedding model can load."""


class OCRFileBatchError(Exception):
    """Raised when OCR extraction fails on one or more files in a batch."""

    def __init__(self, failed_files: list, failure_details: list) -> None:
        self.failed_files = failed_files
        self.failure_details = failure_details
        joined = (
            "; ".join(failure_details) if failure_details else ", ".join(failed_files)
        )
        super().__init__(
            f"OCR extraction failed for {len(failed_files)} file(s): {joined}"
        )


class StaleDataException(Exception):
    """Raised when an update fails because the version has changed (optimistic locking)."""

    pass


class UnsupportedFormatError(Exception):
    """Raised when an optional dependency required to parse a file format is missing or the format is unsupported."""

    pass
