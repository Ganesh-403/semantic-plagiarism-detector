"""
Hybrid Analyzer for the Similarity Pipeline
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple

from src.models.similarity import (
    AnalysisResult, MatchResult, MatchSeverity,
    SimilarityConfig, SimilarityType, DocumentPair,
    AnalysisStatistics
)
from .lexical_analyzer import LexicalAnalyzer
from .semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)


class HybridAnalyzer:
    """
    Hybrid similarity analyzer combining lexical and semantic approaches.
    """
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        self.lexical_analyzer = LexicalAnalyzer(self.config)
        self.semantic_analyzer = SemanticAnalyzer(self.config)
    
    def analyze_pair(
        self,
        source_content: str,
        target_content: str,
        source_id: str = "",
        target_id: str = ""
    ) -> AnalysisResult:
        """Analyze a single pair of documents."""
        start_time = time.time()
        
        result = AnalysisResult(
            source_document_id=source_id,
            target_document_ids=[target_id],
            analysis_type=SimilarityType.HYBRID
        )
        
        # Lexical analysis
        lexical_score, lexical_matches = self.lexical_analyzer.compare_documents(
            source_content, target_content
        )
        
        # Semantic analysis
        semantic_score, semantic_matches = self.semantic_analyzer.compare_documents(
            source_content, target_content
        )
        
        # Combine scores
        hybrid_score = (
            lexical_score * self.config.lexical_weight +
            semantic_score * self.config.semantic_weight
        )
        
        match = MatchResult(
            source_document=source_id,
            target_document=target_id,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
            hybrid_score=hybrid_score,
            metadata={
                'lexical_matches_count': len(lexical_matches),
                'semantic_matches_count': len(semantic_matches)
            }
        )
        match.severity = match.get_severity()
        result.add_match(match)
        
        # Generate summary
        result.summary = {
            'lexical_score': lexical_score,
            'semantic_score': semantic_score,
            'hybrid_score': hybrid_score,
            'severity': match.severity.value,
            'threshold_met': hybrid_score >= self.config.hybrid_threshold
        }
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.mark_completed()
        
        return result
    
    def analyze_batch(
        self,
        source_content: str,
        target_contents: List[Dict[str, str]]
    ) -> AnalysisResult:
        """Analyze a source document against multiple targets."""
        start_time = time.time()
        
        result = AnalysisResult(
            source_document_id="source",
            target_document_ids=[t.get('id', 'unknown') for t in target_contents],
            analysis_type=SimilarityType.HYBRID
        )
        
        matches = []
        
        for target in target_contents:
            target_id = target.get('id', 'unknown')
            target_content = target.get('content', '')
            
            lexical_score, _ = self.lexical_analyzer.compare_documents(
                source_content, target_content
            )
            semantic_score, _ = self.semantic_analyzer.compare_documents(
                source_content, target_content
            )
            hybrid_score = (
                lexical_score * self.config.lexical_weight +
                semantic_score * self.config.semantic_weight
            )
            
            match = MatchResult(
                source_document="source",
                target_document=target_id,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                hybrid_score=hybrid_score
            )
            match.severity = match.get_severity()
            matches.append(match)
        
        matches.sort(key=lambda m: m.hybrid_score, reverse=True)
        result.matches = matches
        
        # Generate summary
        stats = AnalysisStatistics().compute(matches)
        result.summary = stats.to_dict()
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.mark_completed()
        
        return result
    
    def analyze_with_threshold(
        self,
        source_content: str,
        target_content: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, AnalysisResult]:
        """Analyze with a custom threshold."""
        if threshold is None:
            threshold = self.config.hybrid_threshold
        
        result = self.analyze_pair(source_content, target_content)
        
        if result.matches:
            match = result.matches[0]
            threshold_met = match.hybrid_score >= threshold
            return threshold_met, result
        
        return False, result
    
    def get_recommendations(
        self,
        source_content: str,
        target_content: str
    ) -> Dict[str, Any]:
        """Get recommendations based on analysis."""
        result = self.analyze_pair(source_content, target_content)
        
        if not result.matches:
            return {'recommendations': ['No matches found'], 'action_required': False}
        
        match = result.matches[0]
        recommendations = []
        
        if match.hybrid_score >= 0.8:
            recommendations.append("High similarity detected - review content for plagiarism")
            action_required = True
        elif match.hybrid_score >= 0.6:
            recommendations.append("Moderate similarity detected - consider reviewing specific sections")
            action_required = True
        elif match.hybrid_score >= 0.4:
            recommendations.append("Low similarity detected - review at your discretion")
            action_required = False
        else:
            recommendations.append("No significant similarity detected")
            action_required = False
        
        if match.lexical_score > match.semantic_score:
            recommendations.append("Lexical similarity is higher - check for direct copy")
        else:
            recommendations.append("Semantic similarity is higher - check for paraphrased content")
        
        return {
            'recommendations': recommendations,
            'action_required': action_required,
            'score': match.hybrid_score,
            'severity': match.severity.value
        }
    
    def get_analysis_summary(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Get a summary of the analysis result."""
        return {
            'analysis_id': analysis_result.id,
            'source_document': analysis_result.source_document_id,
            'target_documents': analysis_result.target_document_ids,
            'analysis_type': analysis_result.analysis_type.value,
            'matches_count': len(analysis_result.matches),
            'processing_time_ms': analysis_result.processing_time_ms,
            'summary': analysis_result.summary,
            'created_at': analysis_result.created_at.isoformat()
        }