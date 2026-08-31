"""
src/core/pos_normalizer.py
--------------------------
Part-of-Speech (POS) Normalization Engine for Mosaic Plagiarism Detection.

Tokenizes text and extracts normalized POS tag sequences to detect structural
cloning (patchwriting). By stripping lexical content and focusing purely on
syntactic structure (e.g., DET-NOUN-VERB-ADJ), the system can identify
students who copy the exact sentence structure but swap out synonyms.
"""

import re
import logging
from typing import List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

# Simplified POS tag mapping to normalize granular tags (e.g., NN, NNS -> NOUN)
# This reduces the vocabulary space and focuses on macro-syntax.
POS_NORMALIZATION_MAP = {
    # Nouns
    'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',
    # Verbs
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
    # Adjectives
    'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
    # Adverbs
    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
    # Determiners
    'DT': 'DET', 'WDT': 'DET',
    # Pronouns
    'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
    # Prepositions
    'IN': 'ADP', 'TO': 'ADP',
    # Conjunctions
    'CC': 'CONJ',
    # Particles
    'RP': 'PRT',
    # Punctuation (often stripped, but kept as 'PUNCT' if needed)
    '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', ';': 'PUNCT',
    # Others
    'CD': 'NUM', 'MD': 'VERB', 'EX': 'PRON', 'FW': 'X', 'LS': 'X', 'PDT': 'DET',
    'SYM': 'SYM', 'UH': 'INTJ', 'WRB': 'ADV', '#': 'SYM', '$': 'SYM'
}


def _mock_pos_tagger(text: str) -> list[tuple[str, str]]:
    """A lightweight, regex-based heuristic POS tagger for environments without NLTK.
    
    This is a fallback for when nltk is not installed. It uses simple regex
    patterns to guess POS tags. For production, NLTK or spaCy should be used.
    """
    words = re.findall(r'\b\w+\b', text)
    tagged = []
    
    # Simple heuristic dictionaries
    determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your'}
    prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'about', 'to', 'from'}
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    conjunctions = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}
    
    for word in words:
        w_lower = word.lower()
        if w_lower in determiners:
            tagged.append((word, 'DT'))
        elif w_lower in prepositions:
            tagged.append((word, 'IN'))
        elif w_lower in pronouns:
            tagged.append((word, 'PRP'))
        elif w_lower in conjunctions:
            tagged.append((word, 'CC'))
        elif w_lower.endswith('ing'):
            tagged.append((word, 'VBG'))
        elif w_lower.endswith('ed'):
            tagged.append((word, 'VBD'))
        elif w_lower.endswith('ly'):
            tagged.append((word, 'RB'))
        elif w_lower.endswith('s') and len(word) > 3:
            tagged.append((word, 'NNS'))
        else:
            # Default to Noun for unknown words in this simple heuristic
            tagged.append((word, 'NN'))
            
    return tagged


def extract_pos_sequence(text: str, use_nltk: bool = False) -> list[str]:
    """Extract a normalized Part-of-Speech tag sequence from text.
    
    Args:
        text: The input text string.
        use_nltk: If True, attempts to use nltk.pos_tag. Falls back to 
                  heuristic if nltk is unavailable.
                  
    Returns:
        A list of normalized POS tags (e.g., ['DET', 'NOUN', 'VERB', 'ADP', 'DET', 'NOUN']).
    """
    if not text or not text.strip():
        return []
        
    tagged_words = []
    
    if use_nltk:
        try:
            import nltk
            # Ensure punkt and averaged_perceptron_tagger are downloaded
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            from nltk import word_tokenize, pos_tag
            
            tokens = word_tokenize(text)
            tagged_words = pos_tag(tokens)
        except ImportError:
            logger.warning("NLTK not installed. Falling back to heuristic POS tagger.")
            tagged_words = _mock_pos_tagger(text)
        except Exception as e:
            logger.error("NLTK POS tagging failed: %s. Falling back.", e)
            tagged_words = _mock_pos_tagger(text)
    else:
        tagged_words = _mock_pos_tagger(text)
        
    # Normalize the tags
    normalized_sequence = []
    for word, tag in tagged_words:
        # Strip punctuation tags if we only care about syntactic words
        if tag in ('.', ',', ':', ';', '``', "''", '-LRB-', '-RRB-'):
            continue
            
        norm_tag = POS_NORMALIZATION_MAP.get(tag, 'X')
        normalized_sequence.append(norm_tag)
        
    return normalized_sequence


def compute_pos_ngrams(pos_sequence: list[str], n: int = 3) -> list[tuple[str, ...]]:
    """Generate n-grams from a POS sequence.
    
    Args:
        pos_sequence: List of POS tags.
        n: The size of the n-gram (default 3 for trigrams).
        
    Returns:
        List of n-gram tuples.
    """
    if len(pos_sequence) < n:
        return []
    return [tuple(pos_sequence[i:i+n]) for i in range(len(pos_sequence) - n + 1)]


# semantic-plagiarism-detector/src/core/pos_normalizer.py

import nltk
from typing import List

# Ensure required NLTK corpuses/taggers are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

class POSNormalizer:
    """
    Normalizes text into Part-of-Speech (POS) tag sequences to detect 
    syntactic structural cloning (mosaic plagiarism / patchwriting).
    """

    @staticmethod
    def extract_pos_sequence(text: str) -> list[str]:
        """
        Tokenizes text and extracts a normalized sequence of POS tags.
        Example: "The quick brown fox jumps" -> ['DT', 'JJ', 'JJ', 'NN', 'VBZ']
        """
        if not text or not text.strip():
            return []
            
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        
        # Extract just the POS tags and standardize/simplify if needed
        pos_tags = [tag for word, tag in tagged_tokens]
        return pos_tags

    @staticmethod
    def get_pos_string(text: str, separator: str = "-") -> str:
        """Returns the POS sequence as a hyphen-separated string."""
        tags = POSNormalizer.extract_pos_sequence(text)
        return separator.join(tags)
