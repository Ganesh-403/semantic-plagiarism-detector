from __future__ import annotations

from typing import Any


class DocumentWarning:
    """Represents a warning generated during document processing or parsing."""

    def __init__(self, code: str, message: str, severity: str, filename: str) -> None:
        self.code = code
        self.message = message
        self.severity = severity
        self.filename = filename

    def to_dict(self) -> dict[str, Any]:
        """Return a standardized dictionary representation for JSON serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "filename": self.filename,
        }
    