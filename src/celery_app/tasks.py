import io
import logging
from typing import Dict, Any, List

from celery import shared_task
from .celery_config import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="src.celery_app.tasks.process_document")
def process_document(self, file_name: str, file_bytes: bytes, config: Dict[str, Any]) -> Dict[str, Any]:
    from src.core.document_parser import extract_text
    try:
        from src.core.document_parser import remove_ignore_phrases
    except ImportError:
        remove_ignore_phrases = None
    from src.core.embedding_model import embed_documents
    from src.core.text_chunking import chunk_documents
    
    self.update_state(state="PROCESSING", meta={"file": file_name, "step": "ocr"})
    
    ocr_language = config.get("ocr_language", "eng")
    ocr_dpi = config.get("ocr_dpi", 300)
    ignore_phrases = config.get("ignore_phrases")
    
    try:
        raw_text = extract_text(io.BytesIO(file_bytes), file_name, ocr_language=ocr_language, ocr_dpi=ocr_dpi)
        if ignore_phrases and remove_ignore_phrases:
            raw_text = remove_ignore_phrases(raw_text, ignore_phrases)
            
        self.update_state(state="PROCESSING", meta={"file": file_name, "step": "chunking"})
        chunk_size = config.get("chunk_size", 500)
        chunk_overlap = config.get("chunk_overlap", 50)
        chunked_docs = chunk_documents({file_name: raw_text}, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        self.update_state(state="PROCESSING", meta={"file": file_name, "step": "embedding"})
        embeddings = embed_documents(chunked_docs)
        
        return {
            "status": "success",
            "file_name": file_name,
            "raw_text": raw_text,
            "chunks": chunked_docs.get(file_name, []),
            "embedding": embeddings[file_name].tolist() if file_name in embeddings else None
        }
    except Exception as e:
        logger.exception(f"Error processing document {file_name}")
        return {
            "status": "error",
            "file_name": file_name,
            "error": str(e)
        }

@celery_app.task(bind=True, name="src.celery_app.tasks.run_pipeline_job")
def run_pipeline_job(self, file_bytes_dict: Dict[str, bytes], config: Dict[str, Any]) -> Dict[str, Any]:
    from src.core.faiss_index import build_index
    from src.core.similarity import document_similarity_matrix, flag_plagiarism
    from src.db.incidents import sync_flagged_incidents
    from src.core.ai_detector import detect_documents_ai_probability
    import numpy as np
    
    total_files = len(file_bytes_dict)
    self.update_state(state="PROCESSING", meta={"step": "processing_documents", "progress": 0, "total": total_files})
    
    results = []
    processed = 0
    raw_texts = {}
    chunked_docs = {}
    embeddings = {}
    
    # Process each document (in a real highly distributed setup, this would use chords)
    for file_name, file_bytes in file_bytes_dict.items():
        res = process_document.apply(args=[file_name, file_bytes, config]).get()
        if res["status"] == "success":
            raw_texts[file_name] = res["raw_text"]
            chunked_docs[file_name] = res["chunks"]
            if res["embedding"] is not None:
                embeddings[file_name] = np.array(res["embedding"])
        processed += 1
        self.update_state(state="PROCESSING", meta={"step": "processing_documents", "progress": processed, "total": total_files, "current_file": file_name})
        
    self.update_state(state="PROCESSING", meta={"step": "similarity_scoring", "progress": processed, "total": total_files})
    
    sim_df = document_similarity_matrix(embeddings)
    
    threshold = config.get("threshold", 0.6)
    
    self.update_state(state="PROCESSING", meta={"step": "faiss_index", "progress": processed, "total": total_files})
    faiss_index, registry = build_index(embeddings, chunked_docs)
    
    flags = flag_plagiarism(
        sim_df,
        threshold=threshold,
        chunked_docs=chunked_docs,
        embeddings=embeddings,
    )
    
    self.update_state(state="PROCESSING", meta={"step": "sync_incidents", "progress": processed, "total": total_files})
    incidents = sync_flagged_incidents(flags)
    
    return {
        "document_count": len(raw_texts),
        "flags_count": len(flags),
        "incidents_count": len(incidents)
    }
