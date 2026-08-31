"""
Multilingual Support for Plagiarism Detection

Provides language-specific preprocessing for non-Latin scripts
including Arabic, Devanagari, Cyrillic, and more.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from src.core.script_normalizer import ScriptDetector, get_script_normalizer

logger = logging.getLogger(__name__)


# ============================================================================
# LANGUAGE-SPECIFIC STOPWORDS
# ============================================================================

LANGUAGE_STOPWORDS = {
    "arabic": {
        "من",
        "في",
        "على",
        "إلى",
        "عن",
        "مع",
        "بين",
        "بعد",
        "قبل",
        "كان",
        "كانت",
        "يكون",
        "تصبح",
        "أصبح",
        "ظل",
        "بات",
        "أن",
        "إن",
        "إذا",
        "لو",
        "لما",
        "حيث",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "الذي",
        "التي",
        "الذين",
        "اللاتي",
        "عند",
        "لدي",
        "لدى",
        "لذلك",
        "هناك",
        "هنا",
        "كل",
        "بعض",
        "أي",
        "أكثر",
        "أقل",
        "بينما",
        "بسبب",
        "منذ",
        "حتى",
        "دون",
        "ضمن",
        "بلا",
        "ومع",
        "لكن",
        "لقد",
        "قد",
        "سوف",
        "سيكون",
        "كانوا",
        "لهم",
        "علي",
        "عليه",
        "عليها",
        "علينا",
        "عليكم",
        "بنا",
        "بك",
        "به",
        "بها",
        "بهم",
        "بكم",
    },
    "devanagari": {
        "और",
        "यह",
        "वह",
        "जो",
        "से",
        "को",
        "में",
        "पर",
        "तक",
        "का",
        "की",
        "के",
        "ने",
        "है",
        "था",
        "थी",
        "थे",
        "था",
        "थी",
        "थे",
        "हूँ",
        "हो",
        "हैं",
        "मैं",
        "तुम",
        "आप",
        "वे",
        "हम",
        "उन्होंने",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पाँच",
        "यही",
        "वही",
        "जिस",
        "जिन",
        "उस",
        "इस",
        "कर",
        "हुआ",
        "हुई",
        "हुए",
        "रहे",
        "रही",
        "सकता",
        "सकती",
        "सकते",
        "चाहिए",
        "सकें",
        "बहुत",
        "काफी",
        "थोड़ा",
        "ज्यादा",
        "कम",
    },
    "cyrillic": {
        "и",
        "в",
        "на",
        "с",
        "по",
        "к",
        "у",
        "из",
        "за",
        "от",
        "до",
        "для",
        "без",
        "между",
        "через",
        "это",
        "тот",
        "весь",
        "который",
        "такой",
        "быть",
        "являться",
        "становиться",
        "находиться",
        "очень",
        "также",
        "ещё",
        "уже",
        "теперь",
        "тогда",
        "там",
        "здесь",
        "туда",
        "сюда",
        "когда",
        "где",
        "куда",
        "откуда",
        "почему",
        "потому",
        "поэтому",
        "затем",
        "после",
        "прежде",
        "все",
        "вся",
        "всё",
        "всех",
        "всем",
        "всеми",
        "этого",
        "этому",
        "этим",
        "этом",
        "этой",
    },
    "hebrew": {
        "את",
        "על",
        "אל",
        "מן",
        "בין",
        "אצל",
        "עם",
        "של",
        "לי",
        "לך",
        "לו",
        "לה",
        "לנו",
        "לכם",
        "להם",
        "אני",
        "אתה",
        "את",
        "אנחנו",
        "אתם",
        "הם",
        "הוא",
        "היא",
        "זה",
        "זאת",
        "אלה",
        "אלו",
        "כי",
        "אם",
        "כאשר",
        "מאחר",
        "לפני",
        "אחרי",
        "עוד",
        "גם",
        "רק",
        "כל",
        "אין",
        "יש",
    },
}


class MultilingualPreprocessor:
    """
    Preprocess multilingual text for plagiarism detection.

    Features:
    - Script detection
    - Character normalization
    - Language-specific stopword removal
    - Tokenization
    - Transliteration support
    """

    def __init__(self):
        self.script_detector = ScriptDetector()
        self.normalizer = get_script_normalizer()
        self._stopword_cache: Dict[str, Set[str]] = {}

    def preprocess(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Preprocess text for multilingual detection.

        Steps:
        1. Detect script
        2. Normalize characters
        3. Remove stopwords (language-specific)
        4. Tokenize
        """
        if not text:
            return text

        script = self.script_detector.detect(text)
        normalized = self.normalizer.normalize(text, script)

        if remove_stopwords and script in LANGUAGE_STOPWORDS:
            stopwords = self.get_stopwords(script)
            tokens = normalized.split()
            filtered = [t for t in tokens if t not in stopwords]
            normalized = " ".join(filtered)

        return normalized

    def get_stopwords(self, script: str) -> Set[str]:
        """Get stopwords for a specific script/language."""
        if script in self._stopword_cache:
            return self._stopword_cache[script]

        stopwords = set()
        if script in LANGUAGE_STOPWORDS:
            stopwords = set(LANGUAGE_STOPWORDS[script])

        self._stopword_cache[script] = stopwords
        return stopwords

    def add_stopwords(self, script: str, words: List[str]) -> None:
        """Add custom stopwords for a script."""
        if script not in LANGUAGE_STOPWORDS:
            LANGUAGE_STOPWORDS[script] = set()
        LANGUAGE_STOPWORDS[script].update(words)
        if script in self._stopword_cache:
            self._stopword_cache[script].update(words)

    def detect_language(self, text: str) -> str:
        """Detect the primary language/script of text."""
        return self.script_detector.detect(text)

    def normalize_batch(self, texts: Dict[str, str]) -> Dict[str, str]:
        """Normalize a batch of texts."""
        results = {}
        for key, text in texts.items():
            results[key] = self.preprocess(text)
        return results

    def get_language_stats(self, texts: Dict[str, str]) -> Dict[str, int]:
        """Get language statistics for a collection of texts."""
        stats = defaultdict(int)
        for text in texts.values():
            script = self.script_detector.detect(text)
            stats[script] += 1
        return dict(stats)

    def is_multilingual_corpus(self, texts: Dict[str, str]) -> bool:
        """Check if a corpus contains multiple languages."""
        scripts = set()
        for text in texts.values():
            script = self.script_detector.detect(text)
            if script:
                scripts.add(script)
        return len(scripts) > 1

    def get_script_percentage(self, texts: Dict[str, str]) -> Dict[str, float]:
        """Get percentage of each script in a corpus."""
        stats = self.get_language_stats(texts)
        total = sum(stats.values())
        if total == 0:
            return {}
        return {script: (count / total) * 100 for script, count in stats.items()}


# ============================================================================
# MULTILINGUAL TEXT COMPARISON
# ============================================================================


def compare_multilingual_texts(
    text_a: str,
    text_b: str,
    preprocessor: Optional[MultilingualPreprocessor] = None,
) -> dict[str, float]:
    """
    Compare two texts with multilingual support.

    Returns:
        Dict with similarity scores
    """
    if preprocessor is None:
        preprocessor = MultilingualPreprocessor()

    script_a = preprocessor.detect_language(text_a)
    script_b = preprocessor.detect_language(text_b)

    # Normalize both texts
    norm_a = preprocessor.preprocess(text_a)
    norm_b = preprocessor.preprocess(text_b)

    # Compute lexical similarity
    from src.core.lexical_similarity import jaccard_similarity, n_gram_overlap

    jaccard = jaccard_similarity(norm_a, norm_b)
    ngram = n_gram_overlap(norm_a, norm_b, n=3)

    return {
        "jaccard_similarity": jaccard,
        "ngram_similarity": ngram,
        "script_a": script_a,
        "script_b": script_b,
        "same_script": script_a == script_b,
    }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_preprocessor: Optional[MultilingualPreprocessor] = None


def get_multilingual_preprocessor() -> MultilingualPreprocessor:
    """Get global multilingual preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = MultilingualPreprocessor()
    return _preprocessor
