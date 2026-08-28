# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: ADVANCED TEXT ANALYSIS & LINGUISTIC FEATURES (Issue #2003) ──────
# ───────────────────────────────────────────────────────────────────────────────

import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import (
    automated_readability_index,
    coleman_liau_index,
    flesch_kincaid_grade,
    flesch_reading_ease,
    smog_index,
)

warnings.filterwarnings("ignore")

# Download NLTK data if not available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")
try:
    nltk.data.find("chunkers/maxent_ne_chunker")
except LookupError:
    nltk.download("maxent_ne_chunker")
try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words")

# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class ReadabilityMetrics:
    """Readability metrics for a document"""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    difficulty_level: str
    recommended_grade: int
    avg_sentence_length: float
    avg_word_length: float
    avg_syllables_per_word: float
    complex_words_count: int
    polysyllable_words_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StyleMetrics:
    """Writing style metrics for a document"""

    vocabulary_richness: float  # Type-token ratio
    lexical_diversity: float  # Moving average TTR
    sentence_variety: float  # Variation in sentence length
    word_length_distribution: dict[int, int]
    punctuation_frequency: dict[str, int]
    pronoun_frequency: float
    conjunction_frequency: float
    nominalization_frequency: float
    passive_voice_frequency: float
    avg_word_complexity: float
    style_consistency_score: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LinguisticFeatures:
    """Complete linguistic feature set"""

    text_length: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    unique_words: int
    readability: ReadabilityMetrics
    style: StyleMetrics
    pos_distribution: dict[str, int]
    named_entities: list[dict[str, str]]
    vocabulary_profile: dict[str, Any]
    language_detection: dict[str, float]
    document_complexity_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {**asdict(self), "timestamp": self.timestamp.isoformat()}


# ── Text Analysis Engine ──────────────────────────────────────────────────


class TextAnalysisEngine:
    """Advanced text analysis and linguistic feature extraction"""

    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.syllable_cache = {}
        self.analysis_history: list[dict] = []
        self._init_patterns()

    def _init_patterns(self):
        """Initialize linguistic patterns"""
        self.passive_patterns = [
            r"\b(be|am|is|are|was|were|been|being)\s+\w+ed\b",
            r"\b(be|am|is|are|was|were|been|being)\s+\w+en\b",
            r"\b(be|am|is|are|was|were|been|being)\s+\w+t\b",
        ]

        self.nominalization_patterns = [
            r"\w+tion\b",
            r"\w+ment\b",
            r"\w+ness\b",
            r"\w+ity\b",
            r"\w+ence\b",
            r"\w+ance\b",
        ]

        self.pronouns = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "us",
            "our",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
        }

    def analyze_document(
        self, content: str, doc_name: str = None
    ) -> LinguisticFeatures:
        """Perform comprehensive linguistic analysis on a document"""
        if not content or len(content.strip()) < 10:
            return self._empty_features(content, doc_name)

        # Basic text statistics
        words = word_tokenize(content)
        sentences = sent_tokenize(content)
        paragraphs = [p for p in content.split("\n\n") if p.strip()]

        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        unique_words = len(set(w.lower() for w in words))

        # Readability metrics
        readability = self._calculate_readability(content)

        # Style metrics
        style = self._calculate_style_metrics(content, words, sentences)

        # POS distribution
        pos_distribution = self._get_pos_distribution(words)

        # Named entities
        named_entities = self._extract_named_entities(content)

        # Vocabulary profile
        vocabulary_profile = self._get_vocabulary_profile(words)

        # Language detection
        language_detection = self._detect_language(content)

        # Document complexity
        complexity_score = self._calculate_complexity_score(
            readability, style, pos_distribution
        )

        # Create features object
        features = LinguisticFeatures(
            text_length=len(content),
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            unique_words=unique_words,
            readability=readability,
            style=style,
            pos_distribution=pos_distribution,
            named_entities=named_entities,
            vocabulary_profile=vocabulary_profile,
            language_detection=language_detection,
            document_complexity_score=complexity_score,
            metadata={"doc_name": doc_name or "unknown", "analysis_version": "1.0"},
        )

        # Store in history
        self.analysis_history.append(
            {
                "doc_name": doc_name or "unknown",
                "timestamp": features.timestamp,
                "word_count": word_count,
                "complexity_score": complexity_score,
                "readability_score": readability.flesch_reading_ease,
            }
        )

        return features

    def _empty_features(self, content: str, doc_name: str) -> LinguisticFeatures:
        """Return empty features for invalid content"""
        return LinguisticFeatures(
            text_length=len(content),
            word_count=0,
            sentence_count=0,
            paragraph_count=0,
            unique_words=0,
            readability=ReadabilityMetrics(
                flesch_reading_ease=0,
                flesch_kincaid_grade=0,
                smog_index=0,
                coleman_liau_index=0,
                automated_readability_index=0,
                difficulty_level="unknown",
                recommended_grade=0,
                avg_sentence_length=0,
                avg_word_length=0,
                avg_syllables_per_word=0,
                complex_words_count=0,
                polysyllable_words_count=0,
            ),
            style=StyleMetrics(
                vocabulary_richness=0,
                lexical_diversity=0,
                sentence_variety=0,
                word_length_distribution={},
                punctuation_frequency={},
                pronoun_frequency=0,
                conjunction_frequency=0,
                nominalization_frequency=0,
                passive_voice_frequency=0,
                avg_word_complexity=0,
                style_consistency_score=0,
            ),
            pos_distribution={},
            named_entities=[],
            vocabulary_profile={},
            language_detection={},
            document_complexity_score=0,
            metadata={"doc_name": doc_name or "unknown"},
        )

    def _calculate_readability(self, text: str) -> ReadabilityMetrics:
        """Calculate various readability scores"""
        try:
            flesch = flesch_reading_ease(text)
            kincaid = flesch_kincaid_grade(text)
            smog = smog_index(text)
            coleman = coleman_liau_index(text)
            ari = automated_readability_index(text)
        except:
            flesch = kincaid = smog = coleman = ari = 0

        # Additional metrics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        if sentences:
            avg_sentence_len = len(words) / len(sentences) if sentences else 0
        else:
            avg_sentence_len = 0

        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            avg_syllables = (
                sum(self._count_syllables(w) for w in words) / len(words)
                if words
                else 0
            )
        else:
            avg_word_len = 0
            avg_syllables = 0

        # Count complex words
        complex_words = [w for w in words if self._count_syllables(w) >= 3]
        polysyllable_words = [w for w in words if self._count_syllables(w) >= 4]

        # Determine difficulty level
        if flesch > 80:
            difficulty = "Very Easy"
            grade = 4
        elif flesch > 60:
            difficulty = "Easy"
            grade = 8
        elif flesch > 50:
            difficulty = "Moderately Easy"
            grade = 10
        elif flesch > 30:
            difficulty = "Moderately Difficult"
            grade = 12
        elif flesch > 0:
            difficulty = "Difficult"
            grade = 14
        else:
            difficulty = "Very Difficult"
            grade = 16

        return ReadabilityMetrics(
            flesch_reading_ease=flesch,
            flesch_kincaid_grade=kincaid,
            smog_index=smog,
            coleman_liau_index=coleman,
            automated_readability_index=ari,
            difficulty_level=difficulty,
            recommended_grade=grade,
            avg_sentence_length=avg_sentence_len,
            avg_word_length=avg_word_len,
            avg_syllables_per_word=avg_syllables,
            complex_words_count=len(complex_words),
            polysyllable_words_count=len(polysyllable_words),
        )

    def _calculate_style_metrics(
        self, text: str, words: list[str], sentences: list[str]
    ) -> StyleMetrics:
        """Calculate writing style metrics"""
        if not words or not sentences:
            return self._empty_style_metrics()

        # Vocabulary richness (Type-Token Ratio)
        unique_words = len(set(w.lower() for w in words))
        vocabulary_richness = unique_words / len(words) if words else 0

        # Lexical diversity (moving average TTR)
        lexical_diversity = self._calculate_lexical_diversity(words)

        # Sentence variety
        sentence_lengths = [len(sent_tokenize(s)) for s in sentences]
        sentence_variety = np.std(sentence_lengths) if sentence_lengths else 0

        # Word length distribution
        word_lengths = [len(w) for w in words]
        length_dist = Counter(word_lengths)

        # Punctuation frequency
        punctuation = Counter(c for c in text if c in ".,;:!?\"'()[]{}")

        # Pronoun frequency
        pronoun_count = sum(1 for w in words if w.lower() in self.pronouns)
        pronoun_freq = pronoun_count / len(words) if words else 0

        # Conjunction frequency
        conjunctions = {"and", "or", "but", "for", "nor", "so", "yet"}
        conj_count = sum(1 for w in words if w.lower() in conjunctions)
        conj_freq = conj_count / len(words) if words else 0

        # Nominalization frequency
        nominalization_pattern = "|".join(self.nominalization_patterns)
        nominalization_count = len(
            re.findall(nominalization_pattern, text, re.IGNORECASE)
        )
        nominalization_freq = nominalization_count / len(words) if words else 0

        # Passive voice frequency
        passive_count = 0
        for pattern in self.passive_patterns:
            passive_count += len(re.findall(pattern, text, re.IGNORECASE))
        passive_freq = passive_count / len(words) if words else 0

        # Average word complexity
        complexity_scores = [self._word_complexity(w) for w in words]
        avg_complexity = (
            sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        )

        # Style consistency
        consistency = self._calculate_style_consistency(words, sentences)

        return StyleMetrics(
            vocabulary_richness=vocabulary_richness,
            lexical_diversity=lexical_diversity,
            sentence_variety=sentence_variety,
            word_length_distribution=dict(length_dist),
            punctuation_frequency=dict(punctuation),
            pronoun_frequency=pronoun_freq,
            conjunction_frequency=conj_freq,
            nominalization_frequency=nominalization_freq,
            passive_voice_frequency=passive_freq,
            avg_word_complexity=avg_complexity,
            style_consistency_score=consistency,
        )

    def _empty_style_metrics(self) -> StyleMetrics:
        """Return empty style metrics"""
        return StyleMetrics(
            vocabulary_richness=0,
            lexical_diversity=0,
            sentence_variety=0,
            word_length_distribution={},
            punctuation_frequency={},
            pronoun_frequency=0,
            conjunction_frequency=0,
            nominalization_frequency=0,
            passive_voice_frequency=0,
            avg_word_complexity=0,
            style_consistency_score=0,
        )

    def _calculate_lexical_diversity(self, words: list[str]) -> float:
        """Calculate moving average type-token ratio"""
        if len(words) < 50:
            return 0

        window_size = min(100, len(words) // 2)
        tt_ratios = []

        for i in range(0, len(words) - window_size, window_size // 2):
            window = words[i : i + window_size]
            unique = len(set(w.lower() for w in window))
            ratio = unique / len(window) if window else 0
            tt_ratios.append(ratio)

        return sum(tt_ratios) / len(tt_ratios) if tt_ratios else 0

    def _calculate_style_consistency(
        self, words: list[str], sentences: list[str]
    ) -> float:
        """Calculate writing style consistency score"""
        if len(sentences) < 3:
            return 0

        # Calculate sentence length variance
        sent_lengths = [len(w) for w in words]
        length_variance = np.var(sent_lengths) if sent_lengths else 0

        # Calculate word length variance
        word_lengths = [len(w) for w in words]
        word_variance = np.var(word_lengths) if word_lengths else 0

        # Normalize variances
        max_variance = 100
        consistency = 1 - (
            min(length_variance, max_variance) + min(word_variance, max_variance)
        ) / (2 * max_variance)

        return max(0, min(consistency, 1))

    def _get_pos_distribution(self, words: list[str]) -> dict[str, int]:
        """Get part-of-speech tag distribution"""
        try:
            tagged = pos_tag(words)
            pos_counts = Counter(tag for word, tag in tagged)
            return dict(pos_counts)
        except:
            return {}

    def _extract_named_entities(self, text: str) -> list[dict[str, str]]:
        """Extract named entities from text"""
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            chunked = ne_chunk(tagged)

            entities = []
            current_entity = None

            for chunk in chunked:
                if hasattr(chunk, "label"):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "type": chunk.label(),
                        "text": " ".join([token for token, pos in chunk]),
                    }
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            return entities
        except:
            return []

    def _get_vocabulary_profile(self, words: list[str]) -> dict[str, Any]:
        """Generate vocabulary profile"""
        if not words:
            return {}

        word_freq = Counter(w.lower() for w in words)
        unique_words = len(word_freq)
        total_words = len(words)

        # Word frequency distribution
        freq_distribution = {
            "unique": unique_words,
            "total": total_words,
            "type_token_ratio": unique_words / total_words if total_words > 0 else 0,
            "most_common": word_freq.most_common(10),
        }

        # Vocabulary levels
        simple_words = sum(1 for w in words if len(w) <= 4)
        complex_words = sum(1 for w in words if len(w) >= 8)

        freq_distribution.update(
            {
                "simple_word_ratio": (
                    simple_words / total_words if total_words > 0 else 0
                ),
                "complex_word_ratio": (
                    complex_words / total_words if total_words > 0 else 0
                ),
            }
        )

        return freq_distribution

    def _detect_language(self, text: str) -> dict[str, float]:
        """Simple language detection"""
        languages = {
            "en": ["the", "and", "for", "with", "that", "this", "from", "have", "are"],
            "es": ["el", "la", "los", "las", "de", "en", "por", "para", "con"],
            "fr": ["le", "la", "les", "des", "de", "en", "pour", "avec", "sur"],
            "de": ["der", "die", "das", "und", "oder", "aber", "mit", "von", "auf"],
            "zh": ["的", "了", "在", "是", "我", "有", "和", "这", "不"],
            "ja": ["の", "に", "を", "は", "が", "で", "た", "です", "ます"],
        }

        text_lower = text.lower()[:1000]
        scores = {}

        for lang, words in languages.items():
            matches = sum(text_lower.count(word) for word in words)
            scores[lang] = matches / len(words)

        return scores

    def _calculate_complexity_score(
        self,
        readability: ReadabilityMetrics,
        style: StyleMetrics,
        pos_distribution: dict[str, int],
    ) -> float:
        """Calculate overall document complexity score"""
        score = 0
        total_weight = 0

        # Readability contribution
        if readability.flesch_reading_ease > 0:
            readability_score = 1 - (readability.flesch_reading_ease / 100)
            score += readability_score * 0.4
            total_weight += 0.4

        # Vocabulary richness
        if style.vocabulary_richness > 0:
            score += style.vocabulary_richness * 0.3
            total_weight += 0.3

        # Sentence complexity
        if readability.avg_sentence_length > 0:
            sentence_score = min(readability.avg_sentence_length / 30, 1)
            score += sentence_score * 0.2
            total_weight += 0.2

        # Word complexity
        if style.avg_word_complexity > 0:
            word_score = min(style.avg_word_complexity / 3, 1)
            score += word_score * 0.1
            total_weight += 0.1

        return score / total_weight if total_weight > 0 else 0

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        if word in self.syllable_cache:
            return self.syllable_cache[word]

        word = word.lower()
        count = 0
        vowels = "aeiouy"

        # Count vowel groups
        i = 0
        while i < len(word):
            if word[i] in vowels:
                count += 1
                while i < len(word) and word[i] in vowels:
                    i += 1
            else:
                i += 1

        # Handle silent e
        if word.endswith("e") and count > 1:
            count -= 1

        # Handle words with no vowels
        if count == 0:
            count = 1

        self.syllable_cache[word] = count
        return count

    def _word_complexity(self, word: str) -> float:
        """Calculate word complexity score"""
        if len(word) <= 3:
            return 0.1
        elif len(word) <= 5:
            return 0.3
        elif len(word) <= 7:
            return 0.6
        else:
            return 0.9

    def batch_analyze(self, documents: dict[str, str]) -> dict[str, LinguisticFeatures]:
        """Analyze multiple documents"""
        results = {}
        for doc_name, content in documents.items():
            features = self.analyze_document(content, doc_name)
            results[doc_name] = features
        return results

    def compare_linguistic_features(
        self, features_a: LinguisticFeatures, features_b: LinguisticFeatures
    ) -> dict:
        """Compare linguistic features between two documents"""
        comparison = {"similarity_scores": {}, "differences": {}, "summary": {}}

        # Compare readability
        read_a = features_a.readability
        read_b = features_b.readability
        comparison["similarity_scores"]["readability"] = (
            1 - abs(read_a.flesch_reading_ease - read_b.flesch_reading_ease) / 100
        )

        # Compare style
        style_a = features_a.style
        style_b = features_b.style
        style_similarity = 1 - abs(
            style_a.vocabulary_richness - style_b.vocabulary_richness
        )
        comparison["similarity_scores"]["style"] = min(style_similarity, 1)

        # Compare vocabulary
        vocab_sim = 1 - abs(features_a.unique_words - features_b.unique_words) / max(
            features_a.unique_words, features_b.unique_words
        )
        comparison["similarity_scores"]["vocabulary"] = vocab_sim

        # Compare complexity
        comp_sim = 1 - abs(
            features_a.document_complexity_score - features_b.document_complexity_score
        )
        comparison["similarity_scores"]["complexity"] = comp_sim

        # Overall similarity
        overall = sum(comparison["similarity_scores"].values()) / len(
            comparison["similarity_scores"]
        )
        comparison["summary"]["overall_similarity"] = overall
        comparison["summary"]["risk_level"] = (
            "high" if overall > 0.7 else "medium" if overall > 0.4 else "low"
        )

        # Differences
        comparison["differences"] = {
            "readability": abs(read_a.flesch_reading_ease - read_b.flesch_reading_ease),
            "vocabulary_richness": abs(
                style_a.vocabulary_richness - style_b.vocabulary_richness
            ),
            "avg_sentence_length": abs(
                read_a.avg_sentence_length - read_b.avg_sentence_length
            ),
            "word_complexity": abs(
                style_a.avg_word_complexity - style_b.avg_word_complexity
            ),
        }

        return comparison


# ── UI Components ──────────────────────────────────────────────────────────


def render_text_analysis_ui(analyzer: TextAnalysisEngine):
    """Render text analysis UI"""
    st.subheader("📊 Advanced Text Analysis")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📄 Analyze Document",
            "📊 Readability",
            "🎨 Style Analysis",
            "📈 Compare Documents",
        ]
    )

    with tab1:
        render_analysis_tab(analyzer)

    with tab2:
        render_readability_tab(analyzer)

    with tab3:
        render_style_tab(analyzer)

    with tab4:
        render_comparison_tab(analyzer)


def render_analysis_tab(analyzer: TextAnalysisEngine):
    """Render analysis tab"""
    st.subheader("📄 Document Analysis")

    # Select document
    documents = st.session_state.get("raw_texts", {})

    if not documents:
        st.warning("No documents available. Upload documents first.")
        return

    doc_name = st.selectbox(
        "Select Document", options=list(documents.keys()), key="analysis_doc_select"
    )

    if doc_name and st.button("🔍 Analyze Document", type="primary"):
        content = documents.get(doc_name, "")

        if content:
            with st.spinner("Analyzing document..."):
                features = analyzer.analyze_document(content, doc_name)
                st.session_state["current_analysis"] = features

                # Display results
                display_analysis_results(features, doc_name)


def display_analysis_results(features: LinguisticFeatures, doc_name: str):
    """Display analysis results"""
    st.success(f"✅ Analysis complete for: {doc_name}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Words", features.word_count)
    col2.metric("Sentences", features.sentence_count)
    col3.metric("Unique Words", features.unique_words)
    col4.metric("Complexity Score", f"{features.document_complexity_score:.2f}")

    # Create tabs for detailed views
    detail_tab1, detail_tab2, detail_tab3 = st.tabs(
        ["📖 Readability", "✍️ Style", "📊 Features"]
    )

    with detail_tab1:
        st.subheader("📖 Readability Metrics")

        read = features.readability
        col1, col2, col3 = st.columns(3)
        col1.metric("Flesch Reading Ease", f"{read.flesch_reading_ease:.1f}")
        col2.metric("Flesch-Kincaid Grade", f"{read.flesch_kincaid_grade:.1f}")
        col3.metric("Difficulty Level", read.difficulty_level)

        col1, col2, col3 = st.columns(3)
        col1.metric("SMOG Index", f"{read.smog_index:.1f}")
        col2.metric("Coleman-Liau Index", f"{read.coleman_liau_index:.1f}")
        col3.metric("Recommended Grade", read.recommended_grade)

        # Readability gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=read.flesch_reading_ease,
                title={"text": "Flesch Reading Ease"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "red"},
                        {"range": [30, 60], "color": "yellow"},
                        {"range": [60, 100], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 60,
                    },
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with detail_tab2:
        st.subheader("✍️ Style Metrics")

        style = features.style
        col1, col2, col3 = st.columns(3)
        col1.metric("Vocabulary Richness", f"{style.vocabulary_richness:.3f}")
        col2.metric("Lexical Diversity", f"{style.lexical_diversity:.3f}")
        col3.metric("Style Consistency", f"{style.style_consistency_score:.3f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pronoun Frequency", f"{style.pronoun_frequency:.3f}")
        col2.metric("Conjunction Frequency", f"{style.conjunction_frequency:.3f}")
        col3.metric("Passive Voice", f"{style.passive_voice_frequency:.3f}")

        # Word length distribution
        if style.word_length_distribution:
            st.subheader("📊 Word Length Distribution")
            length_df = pd.DataFrame(
                {
                    "Word Length": list(style.word_length_distribution.keys()),
                    "Frequency": list(style.word_length_distribution.values()),
                }
            )
            st.bar_chart(length_df.set_index("Word Length"))

    with detail_tab3:
        st.subheader("📊 Feature Distribution")

        # POS distribution
        if features.pos_distribution:
            pos_df = pd.DataFrame(
                {
                    "POS Tag": list(features.pos_distribution.keys()),
                    "Count": list(features.pos_distribution.values()),
                }
            )
            st.dataframe(pos_df, use_container_width=True)

        # Named entities
        if features.named_entities:
            st.subheader("🔍 Named Entities")
            for entity in features.named_entities:
                st.markdown(f"- **{entity['type']}**: {entity['text']}")

        # Vocabulary profile
        if features.vocabulary_profile:
            st.subheader("📚 Vocabulary Profile")
            profile = features.vocabulary_profile
            col1, col2, col3 = st.columns(3)
            col1.metric("Unique Words", profile.get("unique", 0))
            col2.metric("Type-Token Ratio", f"{profile.get('type_token_ratio', 0):.3f}")
            col3.metric(
                "Simple Words Ratio", f"{profile.get('simple_word_ratio', 0):.3f}"
            )

            if profile.get("most_common"):
                st.subheader("📝 Most Common Words")
                common_df = pd.DataFrame(
                    profile["most_common"], columns=["Word", "Frequency"]
                )
                st.dataframe(common_df, use_container_width=True)


def render_readability_tab(analyzer: TextAnalysisEngine):
    """Render readability tab"""
    st.subheader("📖 Readability Analysis")

    documents = st.session_state.get("raw_texts", {})

    if not documents:
        st.warning("No documents available.")
        return

    if st.button("📊 Analyze All Documents Readability"):
        with st.spinner("Analyzing readability..."):
            results = []
            for doc_name, content in documents.items():
                features = analyzer.analyze_document(content, doc_name)
                read = features.readability
                results.append(
                    {
                        "Document": doc_name,
                        "Flesch Reading Ease": read.flesch_reading_ease,
                        "Flesch-Kincaid Grade": read.flesch_kincaid_grade,
                        "SMOG Index": read.smog_index,
                        "Coleman-Liau Index": read.coleman_liau_index,
                        "Difficulty": read.difficulty_level,
                        "Words": features.word_count,
                    }
                )

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Visualization
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=df["Document"],
                        y=df["Flesch Reading Ease"],
                        name="Flesch Reading Ease",
                        marker_color="#2196F3",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["Document"],
                        y=df["Flesch-Kincaid Grade"] * 10,
                        name="Flesch-Kincaid Grade (x10)",
                        mode="lines+markers",
                        line=dict(color="#FF9800"),
                    )
                )
                fig.update_layout(
                    title="Readability Comparison",
                    xaxis_title="Document",
                    yaxis_title="Score",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)


def render_style_tab(analyzer: TextAnalysisEngine):
    """Render style analysis tab"""
    st.subheader("✍️ Style Analysis")

    documents = st.session_state.get("raw_texts", {})

    if not documents:
        st.warning("No documents available.")
        return

    if st.button("📊 Analyze All Documents Style"):
        with st.spinner("Analyzing writing styles..."):
            results = []
            for doc_name, content in documents.items():
                features = analyzer.analyze_document(content, doc_name)
                style = features.style
                results.append(
                    {
                        "Document": doc_name,
                        "Vocab Richness": style.vocabulary_richness,
                        "Lexical Diversity": style.lexical_diversity,
                        "Style Consistency": style.style_consistency_score,
                        "Pronoun Freq": style.pronoun_frequency,
                        "Passive Voice": style.passive_voice_frequency,
                        "Complexity": features.document_complexity_score,
                    }
                )

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Visualization
                style_metrics = [
                    "Vocab Richness",
                    "Lexical Diversity",
                    "Style Consistency",
                ]
                fig = go.Figure()

                for metric in style_metrics:
                    fig.add_trace(go.Bar(name=metric, x=df["Document"], y=df[metric]))

                fig.update_layout(
                    title="Style Metrics Comparison",
                    xaxis_title="Document",
                    yaxis_title="Score",
                    barmode="group",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)


def render_comparison_tab(analyzer: TextAnalysisEngine):
    """Render comparison tab"""
    st.subheader("📈 Document Comparison")

    documents = st.session_state.get("raw_texts", {})

    if len(documents) < 2:
        st.warning("Need at least 2 documents to compare.")
        return

    col1, col2 = st.columns(2)
    with col1:
        doc_a = st.selectbox("Document A", list(documents.keys()), key="comp_a")
    with col2:
        doc_b = st.selectbox(
            "Document B", [d for d in documents.keys() if d != doc_a], key="comp_b"
        )

    if st.button("🔍 Compare Documents", type="primary"):
        with st.spinner("Comparing documents..."):
            content_a = documents.get(doc_a, "")
            content_b = documents.get(doc_b, "")

            features_a = analyzer.analyze_document(content_a, doc_a)
            features_b = analyzer.analyze_document(content_b, doc_b)

            comparison = analyzer.compare_linguistic_features(features_a, features_b)

            # Display results
            st.subheader("📊 Comparison Results")

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Overall Similarity",
                f"{comparison['summary']['overall_similarity']:.1%}",
            )
            col2.metric("Risk Level", comparison["summary"]["risk_level"].upper())

            similarity_scores = comparison["similarity_scores"]
            col3.metric("Style Similarity", f"{similarity_scores.get('style', 0):.1%}")

            # Similarity chart
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=list(similarity_scores.keys()),
                    y=list(similarity_scores.values()),
                    marker_color=[
                        "#4CAF50" if v > 0.6 else "#FF9800" if v > 0.3 else "#F44336"
                        for v in similarity_scores.values()
                    ],
                )
            )
            fig.update_layout(
                title="Feature Similarity",
                yaxis_title="Similarity Score",
                yaxis_range=[0, 1],
                template="plotly_white",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Differences
            st.subheader("📋 Key Differences")
            diffs = comparison["differences"]
            for key, value in diffs.items():
                if value > 0.3:
                    st.warning(
                        f"**{key.replace('_', ' ').title()}**: {value:.2f} difference"
                    )
                else:
                    st.info(
                        f"**{key.replace('_', ' ').title()}**: {value:.2f} difference"
                    )


# ── Integration with Main App ─────────────────────────────────────────────


def integrate_text_analysis():
    """Initialize and integrate text analysis engine"""
    if "text_analyzer" not in st.session_state:
        st.session_state["text_analyzer"] = TextAnalysisEngine()

    # Add text analysis tab to main app
    render_text_analysis_ui(st.session_state["text_analyzer"])


# ── End of Text Analysis System ──────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
