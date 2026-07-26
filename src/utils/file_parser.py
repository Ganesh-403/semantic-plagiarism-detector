"""
src/utils/file_parser.py
------------------------
Utility functions for parsing PDF, DOCX, and TXT files.
Supports decrypted and password-protected PDF parsing using PyMuPDF (fitz).
"""

from typing import Optional, Tuple

import fitz  # PyMuPDF


class EncryptedPDFError(Exception):
    """Custom exception raised when a PDF requires a password to be read."""
    pass


def extract_text_from_pdf(file_bytes: bytes, password: Optional[str] = None) -> Tuple[str, bool]:
    """
    Extracts text from PDF bytes.
    
    Args:
        file_bytes (bytes): Raw bytes of the uploaded PDF file.
        password (str, optional): Password to decrypt the PDF if protected.
        
    Returns:
        Tuple[str, bool]: Extracted text, and a boolean flag indicating if the PDF was password-protected.
        
    Raises:
        EncryptedPDFError: If the PDF is encrypted and no password (or an incorrect password) is provided.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    is_protected = doc.is_encrypted or doc.needs_pass

    if is_protected:
        if not password:
            raise EncryptedPDFError("PDF is password-protected. Password required.")
        
        # doc.authenticate returns > 0 on success
        auth_success = doc.authenticate(password)
        if not auth_success:
            raise EncryptedPDFError("Incorrect password for PDF.")

    text_content = []
    for page in doc:
        text_content.append(page.get_text())

    doc.close()
    return "\n".join(text_content), is_protected