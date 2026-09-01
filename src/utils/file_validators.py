"""
File validators for the Semantic Plagiarism Detector
"""

import os
import magic
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass
import mimetypes

from src.config.settings import settings


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    error_message: str = ""
    file_type: str = ""
    mime_type: str = ""
    extension: str = ""


class FileValidator:
    """
    Validates uploaded files for the plagiarism detector.
    """
    
    def __init__(self):
        self.max_size = settings.MAX_FILE_SIZE_BYTES
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS
        self.allowed_mime_types = settings.ALLOWED_MIME_TYPES
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a file for upload.
        
        Args:
            file_path: Path to the file
        
        Returns:
            ValidationResult object
        """
        # Check if file exists
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                error_message="File does not exist"
            )
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                error_message="File is empty"
            )
        
        if file_size > self.max_size:
            return ValidationResult(
                is_valid=False,
                error_message=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit"
            )
        
        # Check extension
        extension = file_path.suffix.lower()
        if extension not in self.allowed_extensions:
            return ValidationResult(
                is_valid=False,
                error_message=f"File extension '{extension}' is not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            )
        
        # Check MIME type using python-magic
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            
            # If mime type detection fails, fallback to mimetypes
            if not mime_type:
                mime_type = mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'
            
            # Check if MIME type is allowed
            if mime_type not in self.allowed_mime_types:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"MIME type '{mime_type}' is not allowed"
                )
            
            return ValidationResult(
                is_valid=True,
                file_type=extension[1:],  # Remove leading dot
                mime_type=mime_type,
                extension=extension
            )
            
        except Exception as e:
            # Fallback: check only extension
            return ValidationResult(
                is_valid=True,
                file_type=extension[1:],
                mime_type='application/octet-stream',
                extension=extension
            )
    
    def validate_filename(self, filename: str) -> bool:
        """
        Validate a filename for security.
        
        Args:
            filename: Original filename
        
        Returns:
            True if valid
        """
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for dangerous characters
        dangerous_chars = [';', '|', '&', '$', '>', '<', '`']
        for char in dangerous_chars:
            if char in filename:
                return False
        
        # Check file extension
        extension = Path(filename).suffix.lower()
        if extension not in self.allowed_extensions:
            return False
        
        return True
    
    def get_safe_filename(self, filename: str) -> str:
        """
        Generate a safe filename for storage.
        
        Args:
            filename: Original filename
        
        Returns:
            Safe filename
        """
        import uuid
        import time
        
        # Get extension
        extension = Path(filename).suffix
        
        # Generate safe name
        safe_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{extension}"
        return safe_name


def validate_file(file_path: Path) -> ValidationResult:
    """Convenience function to validate a file."""
    validator = FileValidator()
    return validator.validate_file(file_path)


def validate_filename(filename: str) -> bool:
    """Convenience function to validate a filename."""
    validator = FileValidator()
    return validator.validate_filename(filename)


def get_safe_filename(filename: str) -> str:
    """Convenience function to get a safe filename."""
    validator = FileValidator()
    return validator.get_safe_filename(filename)