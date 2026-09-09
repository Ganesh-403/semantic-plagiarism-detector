"""Document records shared by file parsing services."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib
import uuid


class DocumentType(str, Enum):
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    RTF = "rtf"
    ODT = "odt"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Document:
    filename: str = ""
    original_filename: str = ""
    file_path: str = ""
    file_size: int = 0
    file_type: DocumentType = DocumentType.UNKNOWN
    mime_type: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: dict = field(default_factory=dict)
    word_count: int = 0
    character_count: int = 0
    file_hash: str = ""
    error_message: str = ""
    processed_at: datetime | None = None

    def generate_hash(self) -> str:
        with Path(self.file_path).open("rb") as stream:
            self.file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        return self.file_hash

    def _update_content_stats(self) -> None:
        self.word_count = len(self.content.split())
        self.character_count = len(self.content)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchUpload:
    documents: list[Document] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
