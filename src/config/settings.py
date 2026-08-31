"""
Configuration settings for the Semantic Plagiarism Detector
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Application settings."""
    
    # File upload settings
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: [
        '.txt', '.pdf', '.docx', '.doc', '.rtf', '.odt'
    ])
    ALLOWED_MIME_TYPES: List[str] = field(default_factory=lambda: [
        'text/plain', 'application/pdf', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'application/rtf',
        'application/vnd.oasis.opendocument.text'
    ])
    
    # Upload directories
    UPLOAD_DIR: str = "uploads"
    TEMP_DIR: str = "temp"
    OUTPUT_DIR: str = "output"
    
    # Text extraction settings
    MAX_EXTRACT_LENGTH: int = 100000
    MIN_TEXT_LENGTH: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    def get_upload_path(self, filename: str) -> Path:
        return Path(self.UPLOAD_DIR) / filename
    
    def get_temp_path(self, filename: str) -> Path:
        return Path(self.TEMP_DIR) / filename
    
    def get_output_path(self, filename: str) -> Path:
        return Path(self.OUTPUT_DIR) / filename


@dataclass
class ModelSettings:
    """Model configuration settings."""
    
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_CACHE_DIR: str = "models/cache"
    
    SIMILARITY_THRESHOLD: float = 0.7
    HIGH_SIMILARITY_THRESHOLD: float = 0.85
    LOW_SIMILARITY_THRESHOLD: float = 0.5
    
    BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512


settings = Settings()
model_settings = ModelSettings()