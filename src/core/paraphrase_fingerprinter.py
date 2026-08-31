"""
src/core/paraphrase_fingerprinter.py
------------------------------------
Automated Paraphrase Tool Fingerprinting and Attribution Engine.

Extracts statistical fingerprints from text to attribute it to specific
automated paraphrasing tools (e.g., Quillbot, Spinbot). These tools leave
distinct artifacts such as specific synonym substitution distributions,
predictable sentence-splitting patterns, and transition matrix anomalies.
"""

import re
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Known transition matrices and synonym distributions for common tools
# These are simplified heuristics for demonstration purposes.
KNOWN_TOOL_SIGNATURES = {
    "quillbot": {
        "avg_sentence_length_delta": -0.15, # Tends to shorten sentences slightly
        "synonym_entropy": 0.65,            # Moderate synonym replacement
        "transition_anomaly_score": 0.82    # High structural preservation
    },
    "spinbot": {
        "avg_sentence_length_delta": 0.05,  # Keeps sentence length similar
        "synonym_entropy": 0.85,            # High, often awkward synonym replacement
        "transition_anomaly_score": 0.45    # Lower structural preservation
    },
    "wordtune": {
        "avg_sentence_length_delta": -0.25, # Often splits or shortens sentences
        "synonym_entropy": 0.50,            # Lower synonym replacement, more rewriting
        "transition_anomaly_score": 0.70
    }
}


def compute_synonym_entropy(text: str, baseline_vocab: Optional[set] = None) -> float:
    """Compute the entropy of vocabulary usage to detect synonym substitution.
    
    Automated tools often replace common words with less common synonyms,
    increasing the vocabulary entropy compared to natural writing.
    
    Args:
        text: The input text.
        baseline_vocab: Optional set of expected vocabulary words.
        
    Returns:
        Normalized entropy score between 0.0 and 1.0.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
        
    freq = Counter(words)
    total = len(words)
    
    # Compute Shannon entropy
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
            
    # Normalize entropy (max entropy for N unique words is log2(N))
    unique_words = len(freq)
    if unique_words <= 1:
        return 0.0
        
    max_entropy = math.log2(unique_words)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return round(normalized_entropy, 4)


def compute_sentence_length_variance(text: str) -> float:
    """Compute the variance of sentence lengths.
    
    Paraphrasing tools often normalize sentence lengths, reducing variance
    compared to natural human writing which has higher "burstiness".
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return 0.0
        
    lengths = [len(s.split()) for s in sentences]
    n = len(lengths)
    
    if n <= 1:
        return 0.0
        
    mean_len = sum(lengths) / n
    variance = sum((x - mean_len) ** 2 for x in lengths) / (n - 1)
    
    return round(variance, 4)


def compute_transition_anomaly(text: str) -> float:
    """Compute a heuristic score for syntactic transition anomalies.
    
    This is a simplified metric that measures the frequency of specific
    part-of-speech transitions that are common in machine-paraphrased text
    but rare in natural writing (e.g., excessive use of passive voice or
    specific conjunction patterns).
    """
    # Simplified heuristic: count passive voice indicators and specific transitions
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 3:
        return 0.0
        
    # Count "to be" verbs followed by past participles (passive voice indicator)
    to_be_verbs = {'is', 'was', 'were', 'are', 'been', 'being'}
    passive_count = 0
    
    for i in range(len(words) - 1):
        if words[i] in to_be_verbs and words[i+1].endswith('ed'):
            passive_count += 1
            
    # Normalize by total word count
    anomaly_score = (passive_count / len(words)) * 100.0
    
    # Cap at 1.0 for scoring purposes
    return round(min(1.0, anomaly_score), 4)


def extract_paraphrase_fingerprint(text: str) -> dict[str, float]:
    """Extract a complete statistical fingerprint for paraphrase tool attribution.
    
    Args:
        text: The input text to analyze.
        
    Returns:
        Dictionary containing the extracted statistical features.
    """
    if not text or not text.strip():
        return {
            "synonym_entropy": 0.0,
            "sentence_length_variance": 0.0,
            "transition_anomaly": 0.0
        }
        
    return {
        "synonym_entropy": compute_synonym_entropy(text),
        "sentence_length_variance": compute_sentence_length_variance(text),
        "transition_anomaly": compute_transition_anomaly(text)
    }


def attribute_paraphrase_tool(fingerprint: dict[str, float]) -> dict[str, Any]:
    """Attribute the text to a specific paraphrasing tool based on its fingerprint.
    
    Computes the Euclidean distance between the extracted fingerprint and
    the known signatures of commercial paraphrase tools.
    
    Args:
        fingerprint: The extracted statistical fingerprint.
        
    Returns:
        Dictionary containing the most likely tool and confidence scores.
    """
    scores = {}
    
    for tool_name, signature in KNOWN_TOOL_SIGNATURES.items():
        # Compute Euclidean distance between fingerprint and tool signature
        # We map the fingerprint features to the signature features
        dist = math.sqrt(
            (fingerprint.get("synonym_entropy", 0) - signature["synonym_entropy"]) ** 2 +
            (fingerprint.get("transition_anomaly", 0) - signature["transition_anomaly_score"]) ** 2
        )
        # Convert distance to a similarity score (inverse relationship)
        scores[tool_name] = 1.0 / (1.0 + dist)
        
    # Find the tool with the highest similarity score
    if not scores:
        return {"attributed_tool": "unknown", "confidence": 0.0, "scores": {}}
        
    best_tool = max(scores, key=scores.get)
    best_score = scores[best_tool]
    
    # Normalize confidence to 0-100%
    confidence = best_score * 100.0
    
    return {
        "attributed_tool": best_tool,
        "confidence": round(confidence, 2),
        "scores": {k: round(v * 100.0, 2) for k, v in scores.items()}
    }

# semantic-plagiarism-detector/src/core/paraphrase_fingerprinter.py

import numpy as np
from typing import Dict, Any, List

class ParaphraseFingerprinter:
    """
    Extracts statistical artifacts (synonym-replacement entropy, sentence length variance deltas,
    and transition matrices) to fingerprint automated paraphrasing tools like Quillbot or Spinbot.
    """

    @staticmethod
    def calculate_sentence_length_variance(text: str) -> float:
        """Computes variance in sentence lengths as a measure of robotic uniformity."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        return float(np.var(lengths))

    @staticmethod
    def calculate_synonym_entropy(text: str) -> float:
        """Estimates lexical diversity / synonym entropy using word frequency distributions."""
        words = text.lower().split()
        if not words:
            return 0.0
        unique, counts = np.unique(words, return_counts=True)
        probabilities = counts / len(words)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return float(entropy)

    @classmethod
    def extract_fingerprint(cls, text: str) -> dict[str, float]:
        """Extracts complete statistical signature vector for paraphrasing tool attribution."""
        return {
            "sentence_length_variance": cls.calculate_sentence_length_variance(text),
            "synonym_entropy": cls.calculate_synonym_entropy(text),
            "burstiness_index": round(float(np.random.uniform(0.1, 0.9)), 3) # Heuristic placeholder for stylistic burstiness
        }
