"""
Similarity metrics for the analysis pipeline
"""

import math
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import numpy as np


class SimilarityMetrics:
    """
    Collection of similarity metrics for text comparison.
    """

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    @staticmethod
    def manhattan_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Manhattan distance between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        return sum(abs(a - b) for a, b in zip(vec1, vec2))

    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def dice_similarity(set1: Set, set2: Set) -> float:
        """Calculate Dice coefficient between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        return (2.0 * intersection) / (len(set1) + len(set2))

    @staticmethod
    def overlap_coefficient(set1: Set, set2: Set) -> float:
        """Calculate overlap coefficient between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        return intersection / min(len(set1), len(set2))

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return SimilarityMetrics.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity ratio."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        distance = SimilarityMetrics.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)

    @staticmethod
    def longest_common_subsequence(s1: str, s2: str) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    @staticmethod
    def lcs_similarity(s1: str, s2: str) -> float:
        """Calculate LCS similarity ratio."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        lcs_len = SimilarityMetrics.longest_common_subsequence(s1, s2)
        return (2.0 * lcs_len) / (len(s1) + len(s2))

    @staticmethod
    def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
        """Calculate n-gram similarity."""
        if not s1 or not s2:
            return 0.0
        
        def get_ngrams(text: str, n: int) -> Set[str]:
            text = re.sub(r'\s+', ' ', text.strip())
            if len(text) < n:
                return {text}
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        ngrams1 = get_ngrams(s1, n)
        ngrams2 = get_ngrams(s2, n)
        
        return SimilarityMetrics.jaccard_similarity(ngrams1, ngrams2)

    @staticmethod
    def tfidf_similarity(doc1: str, doc2: str) -> float:
        """Calculate TF-IDF similarity between two documents."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except:
            return 0.0

    @staticmethod
    def word_mover_distance(doc1: str, doc2: str) -> float:
        """Calculate Word Mover's Distance (simplified)."""
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Simplified WMD using word frequency
        common = words1.intersection(words2)
        if not common:
            return 1.0
        
        return 1.0 - (len(common) / max(len(words1), len(words2)))

    @staticmethod
    def combine_scores(
        scores: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """Combine multiple similarity scores with weights."""
        if not scores:
            return 0.0
        
        if weights is None:
            weights = [1.0 / len(scores)] * len(scores)
        
        if len(scores) != len(weights):
            weights = [1.0 / len(scores)] * len(scores)
        
        return sum(s * w for s, w in zip(scores, weights))

    @staticmethod
    def confidence_score(score: float, num_matches: int) -> float:
        """Calculate confidence score for a similarity result."""
        base_confidence = min(score, 0.95)
        match_boost = min(num_matches / 10, 0.05)
        return min(base_confidence + match_boost, 1.0)

    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a score to a range."""
        if score < min_val:
            return 0.0
        if score > max_val:
            return 1.0
        return (score - min_val) / (max_val - min_val)

    @staticmethod
    def similarity_to_percentage(score: float) -> str:
        """Convert similarity score to percentage string."""
        return f"{score * 100:.1f}%"

    @staticmethod
    def get_similarity_level(score: float) -> str:
        """Get text description of similarity level."""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"