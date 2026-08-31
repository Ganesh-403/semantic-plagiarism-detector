"""
Services module for the Semantic Plagiarism Detector
"""

from .document_parser import DocumentParser, parse_document, get_document_info

__all__ = ['DocumentParser', 'parse_document', 'get_document_info']