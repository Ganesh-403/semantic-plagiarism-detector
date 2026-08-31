"""
Semantic Analyzer for the Hybrid Similarity Pipeline
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.models.similarity import SimilarityConfig
from .similarity_metrics import SimilarityMetrics

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Semantic similarity analysis using sentence embeddings.
    """
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        self._model = None
        self._model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            self._model_loaded = True
            logger.info("Semantic model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed.")
            self._model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self._model_loaded = False
    
    def _check_model(self) -> bool:
        """Check if the model is loaded."""
        if not self._model_loaded:
            logger.warning("Semantic model not available")
        return self._model_loaded
    
    def _chunk_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        if not text:
            return []
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self._check_model() or not texts:
            return np.array([])
        
        try:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return np.array([])
    
    def compare_documents(
        self,
        source_content: str,
        target_content: str
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Compare two documents semantically."""
        if not self._check_model() or not source_content or not target_content:
            return 0.0, []
        
        # Chunk into paragraphs
        source_paragraphs = self._chunk_paragraphs(source_content)
        target_paragraphs = self._chunk_paragraphs(target_content)
        
        if not source_paragraphs or not target_paragraphs:
            return 0.0, []
        
        # Generate embeddings
        all_texts = source_paragraphs + target_paragraphs
        embeddings = self.generate_embeddings(all_texts)
        
        if len(embeddings) == 0:
            return 0.0, []
        
        source_embeddings = embeddings[:len(source_paragraphs)]
        target_embeddings = embeddings[len(source_paragraphs):]
        
        # Calculate similarity matrix
        detailed_matches = []
        total_score = 0.0
        matched_pairs = 0
        
        for i, source_emb in enumerate(source_embeddings):
            similarities = []
            for j, target_emb in enumerate(target_embeddings):
                sim = SimilarityMetrics.cosine_similarity(
                    source_emb.tolist(),
                    target_emb.tolist()
                )
                similarities.append(sim)
            
            if similarities:
                best_score = max(similarities)
                best_idx = similarities.index(best_score)
                
                if best_score >= self.config.semantic_threshold:
                    total_score += best_score
                    matched_pairs += 1
                    
                    detailed_matches.append({
                        'source_paragraph': source_paragraphs[i][:200] + "...",
                        'target_paragraph': target_paragraphs[best_idx][:200] + "...",
                        'score': best_score,
                        'source_position': i,
                        'target_position': best_idx
                    })
        
        overall_score = total_score / matched_pairs if matched_pairs > 0 else 0.0
        return overall_score, detailed_matches
    
    def batch_compare(
        self,
        source_content: str,
        target_contents: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Compare a source document against multiple targets."""
        results = []
        
        for target in target_contents:
            score, matches = self.compare_documents(
                source_content,
                target.get('content', '')
            )
            
            results.append({
                'document_id': target.get('id', 'unknown'),
                'document_name': target.get('filename', 'unknown'),
                'score': score,
                'matches': matches,
                'threshold_met': score >= self.config.semantic_threshold
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_similarity_matrix(
        self,
        documents: List[Dict[str, str]]
    ) -> List[List[float]]:
        """Generate a semantic similarity matrix."""
        if not self._check_model():
            return [[0.0] * len(documents) for _ in range(len(documents))]
        
        n = len(documents)
        contents = [doc.get('content', '') for doc in documents]
        embeddings = self.generate_embeddings(contents)
        
        if len(embeddings) == 0:
            return [[0.0] * n for _ in range(n)]
        
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix.tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the semantic model."""
        return {
            'loaded': self._model_loaded,
            'model_name': 'all-MiniLM-L6-v2' if self._model_loaded else None,
            'embedding_dimension': 384 if self._model_loaded else None,
            'requires_installation': not self._model_loaded
        }