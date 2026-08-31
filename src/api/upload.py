"""
Upload API endpoints for the Semantic Plagiarism Detector
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import tempfile

from src.config.settings import settings
from src.models.document import Document, BatchUpload, DocumentStatus
from src.services.document_parser import DocumentParser
from src.utils.file_validators import validate_filename, get_safe_filename

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("/single")
async def upload_single_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    try:
        if not validate_filename(file.filename):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit"
            )
        
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        safe_filename = get_safe_filename(file.filename)
        file_path = upload_dir / safe_filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        parser = DocumentParser()
        document = parser.parse_document(file_path)
        document.original_filename = file.filename
        
        return {
            'success': True,
            'message': 'File uploaded and processed successfully',
            'data': document.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def upload_batch_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    try:
        batch = BatchUpload()
        batch.total_files = len(files)
        
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        parser = DocumentParser()
        
        for file in files:
            try:
                if not validate_filename(file.filename):
                    batch.errors.append(f"Invalid filename: {file.filename}")
                    batch.failed_uploads += 1
                    continue
                
                content = await file.read()
                
                if len(content) > settings.MAX_FILE_SIZE_BYTES:
                    batch.errors.append(f"File {file.filename} exceeds size limit")
                    batch.failed_uploads += 1
                    continue
                
                safe_filename = get_safe_filename(file.filename)
                file_path = upload_dir / safe_filename
                
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                document = parser.parse_document(file_path)
                document.original_filename = file.filename
                
                batch.documents.append(document)
                batch.successful_uploads += 1
            except Exception as e:
                batch.errors.append(f"Error processing {file.filename}: {str(e)}")
                batch.failed_uploads += 1
        
        return {
            'success': True,
            'message': f'Batch upload completed: {batch.successful_uploads} successful, {batch.failed_uploads} failed',
            'data': batch.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents() -> Dict[str, Any]:
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        if not upload_dir.exists():
            return {'success': True, 'data': [], 'count': 0}
        
        documents = []
        parser = DocumentParser()
        
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                doc_info = parser.get_document_info(file_path)
                documents.append({'filename': file_path.name, **doc_info})
        
        return {'success': True, 'data': documents, 'count': len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{filename}")
async def delete_document(filename: str) -> Dict[str, Any]:
    try:
        file_path = Path(settings.UPLOAD_DIR) / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path.unlink()
        
        return {'success': True, 'message': f'File {filename} deleted successfully'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def clear_documents() -> Dict[str, Any]:
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        
        if not upload_dir.exists():
            return {'success': True, 'message': 'No files to clear', 'files_deleted': 0}
        
        files = list(upload_dir.iterdir())
        count = len([f for f in files if f.is_file()])
        
        for file_path in files:
            if file_path.is_file():
                file_path.unlink()
        
        return {'success': True, 'message': f'Cleared {count} files', 'files_deleted': count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, Any]:
    return {
        'success': True,
        'data': {
            'extensions': settings.ALLOWED_EXTENSIONS,
            'mime_types': settings.ALLOWED_MIME_TYPES,
            'max_file_size_mb': settings.MAX_FILE_SIZE_MB,
            'supported_parsers': {
                'pdf': 'PyPDF2',
                'docx': 'python-docx',
                'rtf': 'striprtf',
                'odt': 'odfpy',
                'txt': 'built-in',
                'doc': 'fallback'
            }
        }
    }


@router.post("/extract-text")
async def extract_text_from_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        if not validate_filename(file.filename):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            parser = DocumentParser()
            document = parser.parse_document(tmp_path)
            
            return {
                'success': True,
                'data': {
                    'filename': file.filename,
                    'content': document.content,
                    'word_count': document.word_count,
                    'character_count': document.character_count,
                    'sentence_count': document.sentence_count
                }
            }
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))