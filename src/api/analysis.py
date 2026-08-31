"""
Analysis API endpoints for the Hybrid Similarity Pipeline
"""

from fastapi import APIRouter, HTTPException, Body, Query
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.models.similarity import SimilarityConfig, SimilarityType
from src.analysis.hybrid_analyzer import HybridAnalyzer
from src.analysis.lexical_analyzer import LexicalAnalyzer
from src.analysis.semantic_analyzer import SemanticAnalyzer
from src.services.document_parser import DocumentParser

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Global analyzer instances
_hybrid_analyzer = HybridAnalyzer()
_lexical_analyzer = LexicalAnalyzer()
_semantic_analyzer = SemanticAnalyzer()


@router.post("/hybrid")
async def analyze_hybrid(
    source_document_id: str = Body(...),
    target_document_ids: List[str] = Body(...),
    config: Optional[Dict[str, Any]] = Body(None)
) -> Dict[str, Any]:
    """Run hybrid analysis on documents."""
    try:
        parser = DocumentParser()
        source_path = Path("uploads") / source_document_id
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source document {source_document_id} not found")
        
        source_doc = parser.parse_document(source_path)
        target_contents = []
        
        for target_id in target_document_ids:
            target_path = Path("uploads") / target_id
            if target_path.exists():
                target_doc = parser.parse_document(target_path)
                target_contents.append({
                    'id': target_id,
                    'content': target_doc.content
                })
        
        if not target_contents:
            raise HTTPException(status_code=404, detail="No valid target documents found")
        
        if config:
            similarity_config = SimilarityConfig(**config)
            analyzer = HybridAnalyzer(similarity_config)
        else:
            analyzer = _hybrid_analyzer
        
        result = analyzer.analyze_batch(source_doc.content, target_contents)
        
        return {
            'success': True,
            'data': result.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lexical")
async def analyze_lexical(
    source_content: str = Body(...),
    target_content: str = Body(...)
) -> Dict[str, Any]:
    """Run lexical analysis on two texts."""
    try:
        score, matches = _lexical_analyzer.compare_documents(source_content, target_content)
        
        return {
            'success': True,
            'data': {
                'score': score,
                'matches': matches,
                'threshold_met': score >= SimilarityConfig().lexical_threshold
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic")
async def analyze_semantic(
    source_content: str = Body(...),
    target_content: str = Body(...)
) -> Dict[str, Any]:
    """Run semantic analysis on two texts."""
    try:
        score, matches = _semantic_analyzer.compare_documents(source_content, target_content)
        
        return {
            'success': True,
            'data': {
                'score': score,
                'matches': matches,
                'threshold_met': score >= SimilarityConfig().semantic_threshold
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def analyze_batch(
    source_document_id: str = Body(...),
    target_document_ids: List[str] = Body(...),
    analysis_type: str = Body("hybrid")
) -> Dict[str, Any]:
    """Run batch analysis on multiple documents."""
    try:
        parser = DocumentParser()
        source_path = Path("uploads") / source_document_id
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source document {source_document_id} not found")
        
        source_doc = parser.parse_document(source_path)
        target_contents = []
        
        for target_id in target_document_ids:
            target_path = Path("uploads") / target_id
            if target_path.exists():
                target_doc = parser.parse_document(target_path)
                target_contents.append({
                    'id': target_id,
                    'content': target_doc.content
                })
        
        if not target_contents:
            raise HTTPException(status_code=404, detail="No valid target documents found")
        
        if analysis_type.lower() == "lexical":
            results = _lexical_analyzer.batch_compare(source_doc.content, target_contents)
        elif analysis_type.lower() == "semantic":
            results = _semantic_analyzer.batch_compare(source_doc.content, target_contents)
        else:
            result = _hybrid_analyzer.analyze_batch(source_doc.content, target_contents)
            return {
                'success': True,
                'data': result.to_dict()
            }
        
        return {
            'success': True,
            'data': results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_analysis_config() -> Dict[str, Any]:
    """Get current analysis configuration."""
    config = SimilarityConfig()
    return {
        'success': True,
        'data': config.to_dict()
    }


@router.post("/config")
async def update_analysis_config(
    config: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Update analysis configuration."""
    try:
        new_config = SimilarityConfig(**config)
        return {
            'success': True,
            'message': 'Configuration updated successfully',
            'data': new_config.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """Get semantic model information."""
    return {
        'success': True,
        'data': _semantic_analyzer.get_model_info()
    }


@router.get("/compare")
async def compare_documents(
    source_id: str = Query(...),
    target_id: str = Query(...),
    method: str = Query("hybrid")
) -> Dict[str, Any]:
    """Compare two documents by ID."""
    try:
        parser = DocumentParser()
        source_path = Path("uploads") / source_id
        target_path = Path("uploads") / target_id
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source document {source_id} not found")
        if not target_path.exists():
            raise HTTPException(status_code=404, detail=f"Target document {target_id} not found")
        
        source_doc = parser.parse_document(source_path)
        target_doc = parser.parse_document(target_path)
        
        if method.lower() == "lexical":
            score, matches = _lexical_analyzer.compare_documents(source_doc.content, target_doc.content)
            result_type = "lexical"
        elif method.lower() == "semantic":
            score, matches = _semantic_analyzer.compare_documents(source_doc.content, target_doc.content)
            result_type = "semantic"
        else:
            result = _hybrid_analyzer.analyze_pair(
                source_doc.content, target_doc.content,
                source_id, target_id
            )
            return {
                'success': True,
                'data': result.to_dict()
            }
        
        return {
            'success': True,
            'data': {
                'analysis_type': result_type,
                'score': score,
                'matches': matches,
                'source_id': source_id,
                'target_id': target_id
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(
    source_id: str = Query(...),
    target_id: str = Query(...)
) -> Dict[str, Any]:
    """Get recommendations based on document comparison."""
    try:
        parser = DocumentParser()
        source_path = Path("uploads") / source_id
        target_path = Path("uploads") / target_id
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source document {source_id} not found")
        if not target_path.exists():
            raise HTTPException(status_code=404, detail=f"Target document {target_id} not found")
        
        source_doc = parser.parse_document(source_path)
        target_doc = parser.parse_document(target_path)
        
        recommendations = _hybrid_analyzer.get_recommendations(
            source_doc.content, target_doc.content
        )
        
        return {
            'success': True,
            'data': recommendations
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_analysis_stats() -> Dict[str, Any]:
    """Get analysis statistics."""
    try:
        upload_dir = Path("uploads")
        if not upload_dir.exists():
            return {
                'success': True,
                'data': {
                    'total_documents': 0,
                    'total_comparisons': 0,
                    'average_processing_time': 0
                }
            }
        
        files = list(upload_dir.iterdir())
        return {
            'success': True,
            'data': {
                'total_documents': len([f for f in files if f.is_file()]),
                'total_comparisons': 0,
                'average_processing_time': 0,
                'model_loaded': _semantic_analyzer._model_loaded
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))