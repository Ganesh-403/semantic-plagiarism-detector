"""
document_parser.py
------------------
Handles extraction of raw text from uploaded PDF, DOCX, and TXT files.
Supports both file paths and file-like objects.
"""

import io
import pdfplumber
import docx
from typing import Union, List, Dict


def extract_text_from_pdf(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract all text from a PDF file using pdfplumber for robust layout handling.
    """
    text = ""
    try:
        if isinstance(file, str):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif isinstance(file, bytes):
            with pdfplumber.open(io.BytesIO(file)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        else:
            # Assume it is already a file-like object
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
    except Exception as e:
        print(f"[document_parser] Error reading PDF: {e}")

    return text.strip()


def extract_text_from_docx(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract text from a DOCX file using python-docx.
    """
    text = ""
    try:
        if isinstance(file, bytes):
            doc_file = io.BytesIO(file)
        else:
            doc_file = file
        
        doc = docx.Document(doc_file)
        paragraphs = [p.text for p in doc.paragraphs]
        text = "\n\n".join(paragraphs)
    except Exception as e:
        print(f"[document_parser] Error reading DOCX: {e}")
    
    return text.strip()


def extract_text_from_txt(file: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract text from a TXT file with UTF-8 decoding fallback.
    """
    text = ""
    try:
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif isinstance(file, bytes):
            text = file.decode("utf-8", errors="ignore")
        else:
            data = file.read()
            if isinstance(data, bytes):
                text = data.decode("utf-8", errors="ignore")
            else:
                text = data
    except Exception as e:
        print(f"[document_parser] Error reading TXT: {e}")
    
    return text.strip()


def extract_text(file: Union[str, bytes, io.BytesIO], filename: str) -> str:
    """
    Unified text extraction function routing based on filename extension.
    """
    ext = filename.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext == "docx":
        return extract_text_from_docx(file)
    elif ext == "txt":
        return extract_text_from_txt(file)
    else:
        return extract_text_from_txt(file)


def extract_texts_from_pdfs(files: list) -> dict:
    """
    Legacy compatibility function for extracting multiple PDF/document files.
    """
    return extract_texts(files)


def extract_texts(files: list) -> dict:
    """
    Extract text from multiple files (PDF, DOCX, TXT).
    """
    results = {}
    for file in files:
        if hasattr(file, "name"):
            name = file.name
        elif isinstance(file, str):
            name = file.split("/")[-1].split("\\")[-1]
        else:
            name = f"document_{len(results) + 1}"

        results[name] = extract_text(file, name)
    return results
