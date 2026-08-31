"""
Lexical Analyzer for the Hybrid Similarity Pipeline
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
import numpy as np
import time

from src.models.similarity import MatchResult, SimilarityConfig
from .similarity_metrics import SimilarityMetrics


class LexicalAnalyzer:
    """
    Lexical similarity analysis using TF-IDF, n-grams, and string matching.
    """
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        self.stopwords = self._load_stopwords()
        
    def _load_stopwords(self) -> Set[str]:
        """Load common stopwords."""
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at',
            'to', 'by', 'in', 'of', 'with', 'without', 'about', 'against',
            'between', 'through', 'during', 'within', 'upon', 'towards',
            'this', 'that', 'these', 'those', 'then', 'now', 'so', 'than',
            'very', 'too', 'much', 'more', 'most', 'less', 'least', 'few',
            'some', 'any', 'all', 'both', 'each', 'every', 'other', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than'
        }
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for lexical analysis."""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep words
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.config.use_stopwords:
            words = [w for w in text.split() if w not in self.stopwords]
            text = ' '.join(words)
        
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for comparison."""
        if not text:
            return []
        
        words = text.split()
        if len(words) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap_size
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(' '.join(chunk_words))
            
            if i + chunk_size >= len(words):
                break
        
        return chunks[:self.config.max_chunks]
    
    def compare_documents(
        self,
        source_content: str,
        target_content: str
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compare two documents lexically.
        
        Returns:
            Tuple of (overall_score, detailed_matches)
        """
        if not source_content or not target_content:
            return 0.0, []
        
        # Preprocess
        source_clean = self._preprocess(source_content)
        target_clean = self._preprocess(target_content)
        
        if not source_clean or not target_clean:
            return 0.0, []
        
        # Calculate TF-IDF similarity
        try:
            tfidf_score = SimilarityMetrics.tfidf_similarity(source_clean, target_clean)
        except:
            tfidf_score = 0.0
        
        # Calculate n-gram similarity
        ngram_score = SimilarityMetrics.ngram_similarity(source_clean, target_clean, n=3)
        
        # Calculate Jaccard similarity
        source_words = set(source_clean.split())
        target_words = set(target_clean.split())
        jaccard_score = SimilarityMetrics.jaccard_similarity(source_words, target_words)
        
        # Calculate LCS similarity
        lcs_score = SimilarityMetrics.lcs_similarity(source_clean, target_clean)
        
        # Calculate Levenshtein similarity
        levenshtein_score = SimilarityMetrics.levenshtein_similarity(
            source_clean[:1000], target_clean[:1000]
        )
        
        # Combine scores
        scores = [tfidf_score, ngram_score, jaccard_score, lcs_score, levenshtein_score]
        weights = [0.35, 0.15, 0.15, 0.15, 0.20]
        combined_score = SimilarityMetrics.combine_scores(scores, weights)
        
        # Find matched chunks
        detailed_matches = self._find_matched_chunks(
            source_clean, target_clean, self.config.lexical_threshold
        )
        
        return combined_score, detailed_matches
    
    def _find_matched_chunks(
        self,
        source: str,
        target: str,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Find matching chunks between two documents."""
        chunks_source = self._chunk_text(source)
        chunks_target = self._chunk_text(target)
        
        matches = []
        
        for i, chunk_s in enumerate(chunks_source):
            best_score = 0.0
            best_chunk = ""
            best_pos = 0
            
            for j, chunk_t in enumerate(chunks_target):
                score = SimilarityMetrics.ngram_similarity(chunk_s, chunk_t, n=3)
                
                if score > best_score:
                    best_score = score
                    best_chunk = chunk_t
                    best_pos = j
            
            if best_score >= threshold * 0.5:
                matches.append({
                    'source_chunk': chunk_s[:150] + "..." if len(chunk_s) > 150 else chunk_s,
                    'target_chunk': best_chunk[:150] + "..." if len(best_chunk) > 150 else best_chunk,
                    'score': best_score,
                    'source_position': i,
                    'target_position': best_pos
                })
        
        return matches
    
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
                'threshold_met': score >= self.config.lexical_threshold
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def get_similarity_matrix(
        self,
        documents: List[Dict[str, str]]
    ) -> List[List[float]]:
        """Generate a similarity matrix for multiple documents."""
        n = len(documents)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                score, _ = self.compare_documents(
                    documents[i].get('content', ''),
                    documents[j].get('content', '')
                )
                matrix[i][j] = score
                matrix[j][i] = score
        
        return matrix
    
    def get_word_frequency_comparison(
        self,
        doc1: str,
        doc2: str,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Compare word frequencies between two documents."""
        doc1_clean = self._preprocess(doc1)
        doc2_clean = self._preprocess(doc2)
        
        words1 = doc1_clean.split()
        words2 = doc2_clean.split()
        
        freq1 = Counter(words1)
        freq2 = Counter(words2)
        
        common_words = set(freq1.keys()) & set(freq2.keys())
        unique_to_doc1 = set(freq1.keys()) - set(freq2.keys())
        unique_to_doc2 = set(freq2.keys()) - set(freq1.keys())
        
        return {
            'total_words_doc1': len(words1),
            'total_words_doc2': len(words2),
            'common_words': len(common_words),
            'unique_to_doc1': len(unique_to_doc1),
            'unique_to_doc2': len(unique_to_doc2),
            'top_common_words': [
                {'word': w, 'freq1': freq1[w], 'freq2': freq2[w]}
                for w in sorted(common_words, key=lambda x: freq1[x] + freq2[x], reverse=True)[:top_n]
            ]
        }