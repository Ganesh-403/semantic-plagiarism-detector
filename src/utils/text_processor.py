"""
Text processing utilities for document content
"""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import string


class TextProcessor:
    """
    Processes and analyzes extracted text content.
    """
    
    def __init__(self):
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load common English stopwords."""
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at',
            'to', 'by', 'in', 'of', 'with', 'without', 'about', 'against',
            'between', 'through', 'during', 'within', 'upon', 'towards',
            'this', 'that', 'these', 'those', 'then', 'now', 'so', 'than',
            'very', 'too', 'much', 'more', 'most', 'less', 'least', 'few',
            'some', 'any', 'all', 'both', 'each', 'every', 'other', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than'
        }
    
    def get_word_frequency(self, text: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get word frequency analysis.
        """
        if not text:
            return []
        
        # Clean text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        words = text.split()
        
        # Filter stopwords
        filtered_words = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        counter = Counter(filtered_words)
        
        return [
            {'word': word, 'count': count}
            for word, count in counter.most_common(top_n)
        ]
    
    def get_sentence_count(self, text: str) -> int:
        """Count sentences in text."""
        if not text:
            return 0
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def get_avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)
    
    def get_readability_score(self, text: str) -> Dict[str, Any]:
        """
        Calculate basic readability metrics.
        """
        if not text or len(text) < 20:
            return {
                'flesch_score': 0,
                'grade_level': 0,
                'level': 'unknown'
            }
        
        # Simple Flesch Reading Ease approximation
        sentences = self.get_sentence_count(text)
        words = text.split()
        syllables = self._count_syllables(text)
        
        if sentences == 0 or len(words) == 0:
            return {'flesch_score': 0, 'grade_level': 0, 'level': 'unknown'}
        
        flesch = 206.835 - (1.015 * (len(words) / sentences)) - (84.6 * (syllables / len(words)))
        
        # Determine level
        if flesch >= 90:
            level = 'very_easy'
        elif flesch >= 70:
            level = 'easy'
        elif flesch >= 50:
            level = 'moderate'
        elif flesch >= 30:
            level = 'difficult'
        else:
            level = 'very_difficult'
        
        return {
            'flesch_score': round(flesch, 2),
            'grade_level': round((206.835 - flesch) / 1.015, 1),
            'level': level
        }
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text."""
        text = text.lower()
        syllables = 0
        vowels = 'aeiouy'
        words = text.split()
        
        for word in words:
            if not word:
                continue
            word = word.strip(string.punctuation)
            if not word:
                continue
            
            # Count vowel groups
            in_vowel = False
            word_syllables = 0
            
            for char in word:
                if char in vowels:
                    if not in_vowel:
                        word_syllables += 1
                        in_vowel = True
                else:
                    in_vowel = False
            
            # Handle special cases
            if word.endswith('e'):
                word_syllables -= 1
            if word_syllables == 0:
                word_syllables = 1
            
            syllables += word_syllables
        
        return syllables
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords using frequency analysis.
        """
        if not text:
            return []
        
        word_freq = self.get_word_frequency(text, top_n)
        return [item['word'] for item in word_freq]
    
    def get_ner_suggestions(self, text: str) -> Dict[str, List[str]]:
        """
        Simple Named Entity Recognition suggestions.
        """
        if not text:
            return {'persons': [], 'organizations': [], 'locations': []}
        
        persons = []
        organizations = []
        locations = []
        
        # Simple pattern matching
        words = text.split()
        
        # Look for capitalized words that might be names
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 1:
                # Check if it might be a person
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    name = f"{word} {words[i + 1]}"
                    if len(name) > 3:
                        persons.append(name)
                elif 'Dr.' in word or 'Prof.' in word:
                    persons.append(word)
                elif 'Inc.' in word or 'Corp.' in word or 'LLC' in word:
                    organizations.append(word)
        
        # Simple location detection
        location_indicators = ['in', 'at', 'near', 'from', 'located']
        for indicator in location_indicators:
            for i, word in enumerate(words):
                if word.lower() == indicator and i + 1 < len(words):
                    if words[i + 1][0].isupper():
                        locations.append(words[i + 1])
        
        return {
            'persons': list(set(persons))[:5],
            'organizations': list(set(organizations))[:5],
            'locations': list(set(locations))[:5]
        }
    
    def calculate_content_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple content similarity.
        """
        if not text1 or not text2:
            return 0.0
        
        # Get word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stopwords
        words1 = words1 - self.stopwords
        words2 = words2 - self.stopwords
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return round(intersection / union, 3) if union > 0 else 0.0


def process_text(text: str) -> Dict[str, Any]:
    """Convenience function to process text."""
    processor = TextProcessor()
    return {
        'word_count': len(text.split()),
        'sentence_count': processor.get_sentence_count(text),
        'avg_word_length': processor.get_avg_word_length(text),
        'word_frequency': processor.get_word_frequency(text),
        'readability': processor.get_readability_score(text),
        'keywords': processor.extract_keywords(text)
    }