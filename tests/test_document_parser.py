"""
Unit tests for document parser
"""

import pytest
from pathlib import Path
import tempfile

from src.models.document import Document, DocumentType, DocumentStatus
from src.services.document_parser import DocumentParser
from src.utils.file_validators import FileValidator


class TestDocumentParser:
    
    def setup_method(self):
        self.parser = DocumentParser()
        self.temp_dir = tempfile.mkdtemp()
    
    def test_parse_txt_file(self):
        """Test parsing a text file."""
        file_path = Path(self.temp_dir) / "test.txt"
        file_path.write_text("Hello world! This is a test document.", encoding='utf-8')
        
        doc = self.parser.parse_document(file_path)
        
        assert doc.status == DocumentStatus.COMPLETED
        assert doc.word_count == 7
        assert "Hello world" in doc.content
    
    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        file_path = Path(self.temp_dir) / "empty.txt"
        file_path.touch()
        
        doc = self.parser.parse_document(file_path)
        
        assert doc.status == DocumentStatus.FAILED
        assert "empty" in doc.error_message.lower()
    
    def test_invalid_file_extension(self):
        """Test parsing a file with invalid extension."""
        file_path = Path(self.temp_dir) / "test.exe"
        file_path.write_text("test", encoding='utf-8')
        
        doc = self.parser.parse_document(file_path)
        
        assert doc.status == DocumentStatus.FAILED
        assert "extension" in doc.error_message.lower()
    
    def test_file_too_large(self):
        """Test file size validation."""
        validator = FileValidator()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'0' * (11 * 1024 * 1024))
            file_path = Path(f.name)
        
        result = validator.validate_file(file_path)
        
        assert result.is_valid is False
        assert "exceeds" in result.error_message
        
        file_path.unlink()
    
    def test_get_document_info(self):
        """Test getting document information."""
        file_path = Path(self.temp_dir) / "info_test.txt"
        file_path.write_text("test content", encoding='utf-8')
        
        info = self.parser.get_document_info(file_path)
        
        assert 'filename' in info
        assert 'size_bytes' in info
        assert info['is_readable'] is True
    
    def test_clean_text(self):
        """Test text cleaning."""
        raw_text = "  Hello   world!  This is   a test.  "
        cleaned = self.parser._clean_text(raw_text)
        
        assert cleaned == "Hello world! This is a test."


class TestFileValidator:
    
    def setup_method(self):
        self.validator = FileValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def test_validate_filename_valid(self):
        """Test validating a valid filename."""
        assert self.validator.validate_filename("document.pdf") is True
        assert self.validator.validate_filename("file.txt") is True
        assert self.validator.validate_filename("my_resume.docx") is True
    
    def test_validate_filename_invalid(self):
        """Test validating an invalid filename."""
        assert self.validator.validate_filename("file.exe") is False
        assert self.validator.validate_filename("../file.txt") is False
        assert self.validator.validate_filename("file;txt") is False
    
    def test_get_safe_filename(self):
        """Test generating a safe filename."""
        safe = self.validator.get_safe_filename("test file.pdf")
        assert safe.endswith(".pdf")
        assert len(safe) > 10
        assert " " not in safe