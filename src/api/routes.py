"""
Additional API routes for document management
"""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import List, Optional
import json
import shutil

from src.config.settings import settings
from src.models.document import Document, DocumentStatus
from src.services.document_parser import DocumentParser
from src.utils.file_validators import validate_file

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/stats")
async def get_document_stats() -> dict:
    """
    Get statistics about uploaded documents.
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    
    if not upload_dir.exists():
        return {
            'success': True,
            'data': {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'by_extension': {},
                'by_date': {}
            }
        }
    
    files = list(upload_dir.iterdir())
    total_size = 0
    by_extension = {}
    by_date = {}
    
    for file_path in files:
        if file_path.is_file():
            total_size += file_path.stat().st_size
            
            ext = file_path.suffix.lower() or 'no_extension'
            by_extension[ext] = by_extension.get(ext, 0) + 1
            
            date_key = file_path.stat().st_ctime_date
            by_date[str(date_key)] = by_date.get(str(date_key), 0) + 1
    
    return {
        'success': True,
        'data': {
            'total_files': len([f for f in files if f.is_file()]),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'by_extension': by_extension,
            'by_date': by_date
        }
    }


@router.get("/search")
async def search_documents(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100)
) -> dict:
    """
    Search for documents by filename or content.
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    
    if not upload_dir.exists():
        return {
            'success': True,
            'data': [],
            'count': 0
        }
    
    results = []
    parser = DocumentParser()
    query_lower = query.lower()
    
    for file_path in upload_dir.iterdir():
        if not file_path.is_file():
            continue
        
        # Check filename
        filename_match = query_lower in file_path.name.lower()
        
        # Check content (if text file)
        content_match = False
        if file_path.suffix.lower() in ['.txt']:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                content_match = query_lower in content.lower()
            except:
                pass
        
        if filename_match or content_match:
            doc_info = parser.get_document_info(file_path)
            results.append({
                'filename': file_path.name,
                **doc_info,
                'matched_in': 'filename' if filename_match else 'content'
            })
    
    results = results[:limit]
    
    return {
        'success': True,
        'data': results,
        'count': len(results)
    }


@router.post("/batch/analyze")
async def analyze_batch_documents(
    filenames: List[str]
) -> dict:
    """
    Analyze multiple documents and return their metadata.
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    parser = DocumentParser()
    results = []
    
    for filename in filenames:
        file_path = upload_dir / filename
        
        if not file_path.exists():
            results.append({
                'filename': filename,
                'error': 'File not found'
            })
            continue
        
        try:
            doc = parser.parse_document(file_path)
            results.append({
                'filename': filename,
                'success': True,
                'word_count': doc.word_count,
                'character_count': doc.character_count,
                'sentence_count': doc.sentence_count,
                'status': doc.status.value
            })
        except Exception as e:
            results.append({
                'filename': filename,
                'error': str(e)
            })
    
    return {
        'success': True,
        'data': results
    }


@router.get("/download/{filename}")
async def download_document_content(filename: str) -> dict:
    """
    Get the extracted content of a document.
    """
    file_path = Path(settings.UPLOAD_DIR) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    parser = DocumentParser()
    doc = parser.parse_document(file_path)
    
    return {
        'success': True,
        'data': {
            'filename': filename,
            'content': doc.content,
            'word_count': doc.word_count,
            'character_count': doc.character_count,
            'sentence_count': doc.sentence_count
        }
    }


@router.post("/batch/export")
async def export_batch_metadata(
    filenames: List[str]
) -> dict:
    """
    Export metadata for multiple documents as CSV-ready JSON.
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    parser = DocumentParser()
    results = []
    
    for filename in filenames:
        file_path = upload_dir / filename
        
        if file_path.exists():
            doc = parser.parse_document(file_path)
            results.append({
                'filename': filename,
                'file_size_bytes': doc.file_size,
                'word_count': doc.word_count,
                'character_count': doc.character_count,
                'sentence_count': doc.sentence_count,
                'file_hash': doc.file_hash,
                'file_type': doc.file_type.value
            })
    
    return {
        'success': True,
        'data': results,
        'count': len(results)
    }