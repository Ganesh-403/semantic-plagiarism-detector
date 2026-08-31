"""
Paraphrase & Sentence-Level Plagiarism Detection System
Advanced text analysis for detecting paraphrasing and plagiarism at sentence level
"""

import re
import string
import math
import hashlib
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import datetime
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


class PlagiarismLevel(Enum):
    """Plagiarism severity levels"""
    NONE = "none"  # 0-10% similarity
    LOW = "low"  # 10-30% similarity
    MODERATE = "moderate"  # 30-50% similarity
    HIGH = "high"  # 50-70% similarity
    SEVERE = "severe"  # 70-90% similarity
    EXTREME = "extreme"  # 90-100% similarity


class ParaphraseType(Enum):
    """Types of paraphrasing detected"""
    SYNONYM_REPLACEMENT = "synonym_replacement"
    STRUCTURAL_CHANGE = "structural_change"
    WORD_ORDER_CHANGE = "word_order_change"
    VOICE_CHANGE = "voice_change"  # Active to passive or vice versa
    LENGTH_MODIFICATION = "length_modification"
    PHRASE_REPLACEMENT = "phrase_replacement"
    COMPLEX_PARAPHRASE = "complex_paraphrase"


@dataclass
class SentenceAnalysis:
    """Analysis result for a single sentence"""
    sentence_id: int
    original_sentence: str
    compared_sentence: str
    similarity_score: float
    plagiarism_level: PlagiarismLevel
    paraphrase_type: Optional[ParaphraseType] = None
    matched_phrases: List[str] = field(default_factory=list)
    unmatched_phrases: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    word_overlap: float = 0.0
    structural_similarity: float = 0.0
    semantic_similarity: float = 0.0


@dataclass
class DocumentAnalysis:
    """Analysis result for an entire document"""
    document_id: str
    original_text: str
    compared_text: str
    overall_similarity: float
    plagiarism_level: PlagiarismLevel
    sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)
    high_risk_sentences: List[SentenceAnalysis] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    paraphrasing_detected: bool = False
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


class TextPreprocessor:
    """Text preprocessing for plagiarism detection"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
    
    def preprocess(self, text: str, use_stemming: bool = True, 
                   use_lemmatization: bool = False, 
                   remove_stopwords: bool = True) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([c for c in text if c not in self.punctuation])
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Apply stemming or lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def get_tokens(self, text: str, use_stemming: bool = True) -> List[str]:
        """Get tokens from text"""
        preprocessed = self.preprocess(text, use_stemming=use_stemming)
        return preprocessed.split()
    
    def get_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        return sent_tokenize(text)


class PlagiarismDetector:
    """Main plagiarism detection engine"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.fingerprint_cache = {}
        self.document_store = {}
        
        # Common synonyms for paraphrase detection
        self.synonym_dict = self._initialize_synonym_dict()
        
        # Pattern-based paraphrase detection
        self.paraphrase_patterns = self._initialize_paraphrase_patterns()
    
    def _initialize_synonym_dict(self) -> Dict[str, List[str]]:
        """Initialize common synonym dictionary"""
        return {
            'start': ['begin', 'initiate', 'commence', 'launch'],
            'end': ['finish', 'complete', 'conclude', 'terminate'],
            'help': ['assist', 'aid', 'support', 'facilitate'],
            'change': ['modify', 'alter', 'adjust', 'transform'],
            'create': ['make', 'generate', 'produce', 'fabricate'],
            'reduce': ['decrease', 'diminish', 'minimize', 'lessen'],
            'increase': ['raise', 'grow', 'expand', 'magnify'],
            'improve': ['enhance', 'better', 'upgrade', 'optimize'],
            'analyze': ['examine', 'study', 'inspect', 'investigate'],
            'develop': ['evolve', 'progress', 'advance', 'mature'],
            'use': ['utilize', 'employ', 'apply', 'exploit'],
            'show': ['demonstrate', 'display', 'exhibit', 'present'],
            'need': ['require', 'demand', 'necessitate'],
            'want': ['desire', 'wish', 'crave'],
            'think': ['believe', 'consider', 'contemplate', 'reflect'],
            'know': ['understand', 'comprehend', 'grasp'],
            'get': ['obtain', 'acquire', 'secure', 'procure'],
            'give': ['provide', 'supply', 'offer', 'donate'],
            'take': ['grab', 'seize', 'capture', 'acquire'],
            'make': ['create', 'build', 'construct', 'fabricate'],
            'do': ['perform', 'execute', 'conduct', 'accomplish'],
            'say': ['state', 'declare', 'announce', 'proclaim'],
            'go': ['move', 'proceed', 'advance', 'travel'],
            'come': ['arrive', 'approach', 'appear'],
            'see': ['observe', 'perceive', 'notice', 'view'],
            'look': ['observe', 'examine', 'inspect', 'survey'],
            'work': ['operate', 'function', 'perform'],
            'run': ['operate', 'manage', 'control', 'direct'],
            'find': ['discover', 'locate', 'detect', 'uncover'],
            'learn': ['study', 'acquire', 'absorb', 'master']
        }
    
    def _initialize_paraphrase_patterns(self) -> List[Dict]:
        """Initialize patterns for detecting paraphrasing techniques"""
        return [
            {
                'type': ParaphraseType.SYNONYM_REPLACEMENT,
                'description': 'Words replaced with synonyms',
                'detect_pattern': r'\b(start|begin|initiate|commence|launch|end|finish|complete|conclude|terminate|help|assist|aid|support|facilitate|change|modify|alter|adjust|transform)\b'
            },
            {
                'type': ParaphraseType.VOICE_CHANGE,
                'description': 'Active to passive voice conversion',
                'detect_pattern': r'\b(by|was|were|is being|are being|has been|have been)\b'
            },
            {
                'type': ParaphraseType.WORD_ORDER_CHANGE,
                'description': 'Words reordered in sentence',
                'detect_pattern': r'\b(however|therefore|moreover|furthermore|consequently|accordingly)\b'
            },
            {
                'type': ParaphraseType.LENGTH_MODIFICATION,
                'description': 'Sentence length modified',
                'detect_pattern': r'\b(in addition to|as well as|along with|together with|not only|but also)\b'
            }
        ]
    
    def compute_similarity(self, text1: str, text2: str) -> Dict:
        """Compute multiple similarity metrics between two texts"""
        # Preprocess texts
        processed1 = self.preprocessor.preprocess(text1)
        processed2 = self.preprocessor.preprocess(text2)
        
        # 1. TF-IDF Cosine Similarity
        tfidf_similarity = self._compute_tfidf_similarity(processed1, processed2)
        
        # 2. Jaccard Similarity
        tokens1 = set(processed1.split())
        tokens2 = set(processed2.split())
        jaccard_similarity = self._compute_jaccard_similarity(tokens1, tokens2)
        
        # 3. Word Overlap
        word_overlap = self._compute_word_overlap(processed1, processed2)
        
        # 4. Structural Similarity
        structural_similarity = self._compute_structural_similarity(text1, text2)
        
        # 5. Semantic Similarity (using word embeddings/context)
        semantic_similarity = self._compute_semantic_similarity(text1, text2)
        
        # Overall similarity (weighted average)
        weights = {
            'tfidf': 0.35,
            'jaccard': 0.15,
            'word_overlap': 0.20,
            'structural': 0.15,
            'semantic': 0.15
        }
        
        overall = (tfidf_similarity * weights['tfidf'] +
                   jaccard_similarity * weights['jaccard'] +
                   word_overlap * weights['word_overlap'] +
                   structural_similarity * weights['structural'] +
                   semantic_similarity * weights['semantic'])
        
        return {
            'overall_similarity': overall,
            'tfidf_similarity': tfidf_similarity,
            'jaccard_similarity': jaccard_similarity,
            'word_overlap': word_overlap,
            'structural_similarity': structural_similarity,
            'semantic_similarity': semantic_similarity
        }
    
    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF cosine similarity"""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except:
            return 0.0
    
    def _compute_jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Compute word overlap percentage"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        return overlap / total if total > 0 else 0.0
    
    def _compute_structural_similarity(self, text1: str, text2: str) -> float:
        """Compute structural similarity based on sentence structure"""
        # Compare sentence counts
        sentences1 = self.preprocessor.get_sentences(text1)
        sentences2 = self.preprocessor.get_sentences(text2)
        
        if not sentences1 or not sentences2:
            return 0.0
        
        # Compare average sentence length
        avg_len1 = sum(len(s.split()) for s in sentences1) / len(sentences1)
        avg_len2 = sum(len(s.split()) for s in sentences2) / len(sentences2)
        
        # Length ratio
        length_ratio = min(avg_len1, avg_len2) / max(avg_len1, avg_len2) if max(avg_len1, avg_len2) > 0 else 0
        
        # Sentence count ratio
        count_ratio = min(len(sentences1), len(sentences2)) / max(len(sentences1), len(sentences2)) if max(len(sentences1), len(sentences2)) > 0 else 0
        
        # Structural similarity is average of these ratios
        structural_similarity = (length_ratio + count_ratio) / 2
        
        return structural_similarity
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using word embeddings (simplified)"""
        # Simplified: Use common word overlap weighted by semantic importance
        words1 = set(self.preprocessor.get_tokens(text1, use_stemming=True))
        words2 = set(self.preprocessor.get_tokens(text2, use_stemming=True))
        
        if not words1 or not words2:
            return 0.0
        
        # Count common words with higher weight for content words
        common_words = words1.intersection(words2)
        if not common_words:
            return 0.0
        
        # Use semantic importance (simplified - longer words have more meaning)
        semantic_weight = sum(len(w) for w in common_words) / sum(len(w) for w in words1.union(words2))
        
        # Combine with jaccard
        jaccard = len(common_words) / len(words1.union(words2))
        
        # Semantic similarity is weighted combination
        semantic_similarity = 0.6 * jaccard + 0.4 * semantic_weight
        
        return min(1.0, semantic_similarity)
    
    def detect_paraphrase_type(self, original: str, compared: str) -> List[ParaphraseType]:
        """Detect what type of paraphrasing was used"""
        detected_types = []
        original_tokens = set(self.preprocessor.get_tokens(original, use_stemming=False))
        compared_tokens = set(self.preprocessor.get_tokens(compared, use_stemming=False))
        
        # Check for synonym replacement
        synonym_count = 0
        for word in original_tokens:
            if word in self.synonym_dict:
                synonyms = set(self.synonym_dict[word])
                if synonyms.intersection(compared_tokens):
                    synonym_count += 1
        
        if synonym_count >= 2:
            detected_types.append(ParaphraseType.SYNONYM_REPLACEMENT)
        
        # Check for structural changes
        original_words = original.split()
        compared_words = compared.split()
        
        if len(original_words) != len(compared_words):
            detected_types.append(ParaphraseType.LENGTH_MODIFICATION)
        
        # Check for word order changes
        if len(original_tokens) == len(compared_tokens) and original_tokens != compared_tokens:
            if sorted(original_tokens) != sorted(compared_tokens):
                detected_types.append(ParaphraseType.WORD_ORDER_CHANGE)
        
        # Check for voice change
        passive_markers = ['by', 'was', 'were', 'is being', 'are being', 'has been', 'have been']
        original_has_passive = any(marker in original.lower() for marker in passive_markers)
        compared_has_passive = any(marker in compared.lower() for marker in passive_markers)
        
        if original_has_passive != compared_has_passive:
            detected_types.append(ParaphraseType.VOICE_CHANGE)
        
        # If multiple types detected, classify as complex
        if len(detected_types) >= 2:
            detected_types.append(ParaphraseType.COMPLEX_PARAPHRASE)
        
        return detected_types if detected_types else [ParaphraseType.PHRASE_REPLACEMENT]
    
    def analyze_sentences(self, original_text: str, compared_text: str) -> List[SentenceAnalysis]:
        """Analyze text at sentence level"""
        original_sentences = self.preprocessor.get_sentences(original_text)
        compared_sentences = self.preprocessor.get_sentences(compared_text)
        
        sentence_analyses = []
        
        # Compare each sentence
        for i, (orig_sent, comp_sent) in enumerate(zip(original_sentences, compared_sentences)):
            # Compute similarity
            similarity_metrics = self.compute_similarity(orig_sent, comp_sent)
            
            # Determine plagiarism level
            overall_similarity = similarity_metrics['overall_similarity']
            plagiarism_level = self._get_plagiarism_level(overall_similarity)
            
            # Detect paraphrase type
            paraphrase_types = self.detect_paraphrase_type(orig_sent, comp_sent)
            
            # Find matched and unmatched phrases
            orig_phrases = set(orig_sent.split())
            comp_phrases = set(comp_sent.split())
            matched = list(orig_phrases.intersection(comp_phrases))
            unmatched = list(orig_phrases.difference(comp_phrases))
            
            # Generate suggestions
            suggestions = self._generate_suggestions(orig_sent, comp_sent, overall_similarity)
            
            analysis = SentenceAnalysis(
                sentence_id=i,
                original_sentence=orig_sent,
                compared_sentence=comp_sent,
                similarity_score=overall_similarity,
                plagiarism_level=plagiarism_level,
                paraphrase_type=paraphrase_types[0] if paraphrase_types else None,
                matched_phrases=matched[:10],
                unmatched_phrases=unmatched[:10],
                suggestions=suggestions,
                word_overlap=similarity_metrics['word_overlap'],
                structural_similarity=similarity_metrics['structural_similarity'],
                semantic_similarity=similarity_metrics['semantic_similarity']
            )
            
            sentence_analyses.append(analysis)
        
        return sentence_analyses
    
    def _get_plagiarism_level(self, similarity: float) -> PlagiarismLevel:
        """Determine plagiarism level based on similarity score"""
        if similarity < 0.10:
            return PlagiarismLevel.NONE
        elif similarity < 0.30:
            return PlagiarismLevel.LOW
        elif similarity < 0.50:
            return PlagiarismLevel.MODERATE
        elif similarity < 0.70:
            return PlagiarismLevel.HIGH
        elif similarity < 0.90:
            return PlagiarismLevel.SEVERE
        else:
            return PlagiarismLevel.EXTREME
    
    def _generate_suggestions(self, original: str, compared: str, similarity: float) -> List[str]:
        """Generate suggestions for improving originality"""
        suggestions = []
        
        if similarity > 0.7:
            suggestions.append("High similarity detected - consider rewriting this sentence completely")
            suggestions.append("Try using different vocabulary and sentence structure")
        elif similarity > 0.4:
            suggestions.append("Moderate similarity - consider paraphrasing further")
            suggestions.append("Add more original examples or case studies")
        else:
            suggestions.append("Originality level acceptable - continue to ensure proper citation")
        
        # Check for repeated phrases
        original_words = set(original.lower().split())
        compared_words = set(compared.lower().split())
        common = original_words.intersection(compared_words)
        
        if len(common) > len(original_words) * 0.5:
            suggestions.append("Consider replacing common phrases with alternatives")
        
        # Suggest synonym replacements
        for word in original_words:
            if word in self.synonym_dict:
                synonyms = self.synonym_dict[word][:3]
                suggestions.append(f"Consider replacing '{word}' with: {', '.join(synonyms)}")
                break
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def analyze_document(self, original_text: str, compared_text: str, 
                         document_id: Optional[str] = None) -> DocumentAnalysis:
        """Analyze entire document for plagiarism"""
        if not document_id:
            document_id = hashlib.md5(original_text.encode()).hexdigest()[:8]
        
        # Sentence-level analysis
        sentence_analyses = self.analyze_sentences(original_text, compared_text)
        
        # Document-level metrics
        overall_similarity = sum(a.similarity_score for a in sentence_analyses) / len(sentence_analyses) if sentence_analyses else 0
        
        # Identify high-risk sentences (>50% similarity)
        high_risk = [a for a in sentence_analyses if a.similarity_score > 0.5]
        
        # Calculate statistics
        stats = self._calculate_statistics(sentence_analyses)
        
        # Detect paraphrasing
        paraphrasing_detected = any(a.paraphrase_type for a in sentence_analyses)
        
        analysis = DocumentAnalysis(
            document_id=document_id,
            original_text=original_text[:1000] + "..." if len(original_text) > 1000 else original_text,
            compared_text=compared_text[:1000] + "..." if len(compared_text) > 1000 else compared_text,
            overall_similarity=overall_similarity,
            plagiarism_level=self._get_plagiarism_level(overall_similarity),
            sentence_analyses=sentence_analyses,
            high_risk_sentences=high_risk,
            statistics=stats,
            paraphrasing_detected=paraphrasing_detected
        )
        
        # Store document
        self.document_store[document_id] = analysis
        
        return analysis
    
    def _calculate_statistics(self, sentence_analyses: List[SentenceAnalysis]) -> Dict:
        """Calculate comprehensive statistics"""
        if not sentence_analyses:
            return {}
        
        similarities = [a.similarity_score for a in sentence_analyses]
        
        return {
            'total_sentences': len(sentence_analyses),
            'average_similarity': sum(similarities) / len(similarities),
            'max_similarity': max(similarities),
            'min_similarity': min(similarities),
            'std_deviation': math.sqrt(sum((s - sum(similarities)/len(similarities))**2 for s in similarities) / len(similarities)) if len(similarities) > 1 else 0,
            'high_risk_sentences': len([a for a in sentence_analyses if a.similarity_score > 0.5]),
            'medium_risk_sentences': len([a for a in sentence_analyses if 0.3 < a.similarity_score <= 0.5]),
            'low_risk_sentences': len([a for a in sentence_analyses if a.similarity_score <= 0.3]),
            'paraphrase_types': Counter(a.paraphrase_type.value if a.paraphrase_type else 'none' for a in sentence_analyses).most_common()
        }


class IntegratedPlagiarismSystem:
    """Integrated plagiarism detection with sustainability and skill wallet systems"""
    
    def __init__(self, sustainability_system=None, skill_wallet_manager=None):
        self.detector = PlagiarismDetector()
        self.sustainability_system = sustainability_system
        self.skill_wallet_manager = skill_wallet_manager
        
        # Store plagiarism reports
        self.reports: Dict[str, DocumentAnalysis] = {}
        
        # User contribution tracking
        self.user_submissions: Dict[str, List[str]] = defaultdict(list)
        
        # Skill mapping for plagiarism detection
        self._initialize_plagiarism_skills()
    
    def _initialize_plagiarism_skills(self):
        """Initialize skills related to plagiarism detection and academic integrity"""
        if self.skill_wallet_manager:
            # Add plagiarism-related skills
            plagiarism_skills = [
                {
                    'id': 'skill_plag_001',
                    'name': 'Academic Integrity',
                    'description': 'Understanding and practicing academic integrity',
                    'category': SkillCategory.SOFT,
                    'level': SkillLevel.BEGINNER
                },
                {
                    'id': 'skill_plag_002',
                    'name': 'Paraphrasing',
                    'description': 'Ability to paraphrase content effectively',
                    'category': SkillCategory.COMMUNICATION,
                    'level': SkillLevel.BEGINNER
                },
                {
                    'id': 'skill_plag_003',
                    'name': 'Citation Skills',
                    'description': 'Proper citation and referencing techniques',
                    'category': SkillCategory.TECHNICAL,
                    'level': SkillLevel.BEGINNER
                }
            ]
            
            for skill_data in plagiarism_skills:
                skill = Skill(
                    skill_id=skill_data['id'],
                    name=skill_data['name'],
                    description=skill_data['description'],
                    category=skill_data['category'],
                    level=skill_data['level']
                )
                self.skill_wallet_manager.skill_definitions[skill.skill_id] = skill
    
    def check_submission(self, user_id: str, original_text: str, 
                         compared_text: str, document_id: Optional[str] = None) -> DocumentAnalysis:
        """Check a submission for plagiarism and update user stats"""
        # Perform plagiarism detection
        analysis = self.detector.analyze_document(original_text, compared_text, document_id)
        
        # Store report
        self.reports[analysis.document_id] = analysis
        self.user_submissions[user_id].append(analysis.document_id)
        
        # Update sustainability system
        if self.sustainability_system:
            self._update_sustainability_system(user_id, analysis)
        
        # Update skill wallet
        if self.skill_wallet_manager:
            self._update_skill_wallet(user_id, analysis)
        
        return analysis
    
    def _update_sustainability_system(self, user_id: str, analysis: DocumentAnalysis):
        """Update sustainability system based on plagiarism results"""
        user = self.sustainability_system.get_user(user_id)
        if not user:
            return
        
        # Award points based on originality
        originality_score = 1 - analysis.overall_similarity
        
        if originality_score > 0.9:
            points = 30  # Very original
            user.total_points += points
            user.add_xp(points)
            print(f"  🌟 Awarded {points} sustainability points for exceptional originality!")
        elif originality_score > 0.7:
            points = 15  # Moderately original
            user.total_points += points
            user.add_xp(points // 2)
            print(f"  ✅ Awarded {points} sustainability points for good originality!")
        elif originality_score > 0.4:
            points = 5  # Some originality
            user.total_points += points
            user.add_xp(points // 2)
            print(f"  ℹ️ Awarded {points} sustainability points - consider improving originality")
        else:
            # High similarity - penalty
            penalty = 10
            user.total_points = max(0, user.total_points - penalty)
            print(f"  ⚠️ Penalized {penalty} points - submission has high similarity")
    
    def _update_skill_wallet(self, user_id: str, analysis: DocumentAnalysis):
        """Update skill wallet based on plagiarism performance"""
        wallet = self.skill_wallet_manager.get_skill_wallet(user_id)
        if not wallet:
            return
        
        # Award paraphrasing skill based on ability to paraphrase
        originality_score = 1 - analysis.overall_similarity
        
        if originality_score > 0.8 and analysis.paraphrasing_detected:
            # Award or upgrade paraphrasing skill
            skill_id = 'skill_plag_002'
            if skill_id in wallet.skills:
                skill = wallet.skills[skill_id]
                skill.add_experience(20)
                if skill.experience_points > 100:
                    skill.level = SkillLevel.INTERMEDIATE
                    print(f"  📈 Upgraded Paraphrasing skill to INTERMEDIATE!")
            else:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  🎯 Awarded Paraphrasing skill!")
        
        # Award academic integrity skill
        if originality_score > 0.7:
            skill_id = 'skill_plag_001'
            if skill_id not in wallet.skills:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  🎯 Awarded Academic Integrity skill!")
        
        # Award citation skill if references are properly cited (simplified check)
        if 'citation' in analysis.original_text.lower() or 'reference' in analysis.original_text.lower():
            skill_id = 'skill_plag_003'
            if skill_id not in wallet.skills:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  📚 Awarded Citation Skills!")
    
    def get_plagiarism_report(self, document_id: str) -> Optional[DocumentAnalysis]:
        """Get a specific plagiarism report"""
        return self.reports.get(document_id)
    
    def get_user_reports(self, user_id: str) -> List[DocumentAnalysis]:
        """Get all reports for a user"""
        reports = []
        for doc_id in self.user_submissions.get(user_id, []):
            if doc_id in self.reports:
                reports.append(self.reports[doc_id])
        return reports
    
    def get_plagiarism_stats(self) -> Dict:
        """Get overall plagiarism statistics"""
        if not self.reports:
            return {}
        
        similarities = [r.overall_similarity for r in self.reports.values()]
        
        return {
            'total_documents': len(self.reports),
            'average_similarity': sum(similarities) / len(similarities),
            'max_similarity': max(similarities),
            'min_similarity': min(similarities),
            'plagiarism_distribution': {
                level.value: len([r for r in self.reports.values() if r.plagiarism_level == level])
                for level in PlagiarismLevel
            },
            'paraphrasing_detected': len([r for r in self.reports.values() if r.paraphrasing_detected]),
            'high_risk_documents': len([r for r in self.reports.values() if r.overall_similarity > 0.7])
        }
    
    def generate_plagiarism_insights(self, user_id: str) -> Dict:
        """Generate insights and recommendations for a user based on their submissions"""
        reports = self.get_user_reports(user_id)
        
        if not reports:
            return {"message": "No submissions found for this user"}
        
        # Calculate user's average originality
        avg_similarity = sum(r.overall_similarity for r in reports) / len(reports)
        avg_originality = 1 - avg_similarity
        
        # Identify improvement areas
        improvements = []
        if avg_originality < 0.6:
            improvements.append("Focus on improving paraphrasing skills")
            improvements.append("Use synonyms and restructure sentences more effectively")
        
        # Check for repeated issues
        high_risk_count = len([r for r in reports if r.overall_similarity > 0.5])
        if high_risk_count > len(reports) * 0.3:
            improvements.append("Multiple high-similarity submissions detected")
            improvements.append("Consider using more original examples and case studies")
        
        # Skill recommendations
        skill_recommendations = []
        if avg_originality < 0.5:
            skill_recommendations.append({
                'skill': 'Paraphrasing',
                'reason': 'Need to improve ability to express ideas differently',
                'action': 'Practice paraphrasing exercises and use synonym dictionaries'
            })
        
        if high_risk_count > 0:
            skill_recommendations.append({
                'skill': 'Academic Integrity',
                'reason': 'High similarity in multiple submissions',
                'action': 'Review academic integrity guidelines and proper citation methods'
            })
        
        return {
            'user_id': user_id,
            'total_submissions': len(reports),
            'average_similarity': avg_similarity,
            'average_originality': avg_originality,
            'plagiarism_level': self.detector._get_plagiarism_level(avg_similarity).value,
            'improvement_areas': improvements,
            'skill_recommendations': skill_recommendations,
            'best_submission': {
                'document_id': min(reports, key=lambda r: r.overall_similarity).document_id,
                'similarity': min(reports, key=lambda r: r.overall_similarity).overall_similarity
            },
            'worst_submission': {
                'document_id': max(reports, key=lambda r: r.overall_similarity).document_id,
                'similarity': max(reports, key=lambda r: r.overall_similarity).overall_similarity
            }
        }


# Demo function
def demo_plagiarism_system():
    """Demonstrate the plagiarism detection system"""
    print("🔍 PARAPHRASE & PLAGIARISM DETECTION SYSTEM 🔍")
    print("=" * 80)
    
    # Initialize integrated system
    sustainability = SustainabilityGamification()
    skill_manager = SkillWalletProjectManager(sustainability)
    plagiarism_system = IntegratedPlagiarismSystem(sustainability, skill_manager)
    
    # Register a user
    print("\n📝 Registering user...")
    user = sustainability.register_user("user_001", "EcoAlice")
    wallet = skill_manager.create_skill_wallet("user_001")
    print(f"✅ User registered: {user.username}")
    
    # Test texts
    original_text = """
    Climate change is one of the most pressing issues facing humanity today. 
    The rapid increase in greenhouse gas emissions has led to global warming, 
    which in turn causes extreme weather events, rising sea levels, and 
    disruptions to ecosystems. It is imperative that we take immediate action 
    to reduce our carbon footprint and transition to sustainable energy sources. 
    This requires a concerted effort from governments, businesses, and individuals 
    working together to implement effective climate policies.
    """
    
    # Paraphrased version (with some similarity)
    paraphrased_text = """
    One of the most urgent challenges currently confronting humankind is climate change. 
    Global warming, driven by the quick rise in greenhouse gas emissions, results in 
    severe weather phenomena, increasing ocean levels, and ecological imbalances. 
    Taking prompt steps to lower our carbon emissions and shift towards renewable energy 
    is essential. Achieving this demands united action from governing bodies, 
    commercial enterprises, and citizens collaborating to enforce impactful environmental strategies.
    """
    
    # Very similar version (potential plagiarism)
    plagiarized_text = """
    Climate change is one of the most pressing issues facing humanity today. 
    The rapid increase in greenhouse gas emissions has led to global warming, 
    which in turn causes extreme weather events, rising sea levels, and 
    disruptions to ecosystems. We must take immediate action to reduce our 
    carbon footprint and transition to sustainable energy sources.
    """
    
    # Test 1: Check plagiarism detection
    print("\n🔍 Testing Plagiarism Detection...")
    print("-" * 60)
    
    # Check paraphrased version
    print("\n1️⃣ Checking Paraphrased Version:")
    result1 = plagiarism_system.check_submission(
        "user_001", 
        original_text, 
        paraphrased_text, 
        "doc_001"
    )
    
    print(f"\n📊 Results:")
    print(f"  Overall Similarity: {result1.overall_similarity:.2%}")
    print(f"  Plagiarism Level: {result1.plagiarism_level.value.upper()}")
    print(f"  Paraphrasing Detected: {result1.paraphrasing_detected}")
    print(f"  High Risk Sentences: {len(result1.high_risk_sentences)} out of {len(result1.sentence_analyses)}")
    
    # Show sentence-level analysis
    print("\n📝 Sentence-Level Analysis:")
    for i, analysis in enumerate(result1.sentence_analyses[:3], 1):
        print(f"\n  Sentence {i}:")
        print(f"    Original: {analysis.original_sentence[:80]}...")
        print(f"    Compared: {analysis.compared_sentence[:80]}...")
        print(f"    Similarity: {analysis.similarity_score:.2%}")
        if analysis.paraphrase_type:
            print(f"    Paraphrase Type: {analysis.paraphrase_type.value.replace('_', ' ').title()}")
        if analysis.suggestions:
            print(f"    Suggestion: {analysis.suggestions[0]}")
    
    # Test 2: Check plagiarized version
    print("\n\n2️⃣ Checking Plagiarized Version:")
    result2 = plagiarism_system.check_submission(
        "user_001",
        original_text,
        plagiarized_text,
        "doc_002"
    )
    
    print(f"\n📊 Results:")
    print(f"  Overall Similarity: {result2.overall_similarity:.2%}")
    print(f"  Plagiarism Level: {result2.plagiarism_level.value.upper()}")
    print(f"  Paraphrasing Detected: {result2.paraphrasing_detected}")
    print(f"  High Risk Sentences: {len(result2.high_risk_sentences)} out of {len(result2.sentence_analyses)}")
    
    # Show user insights
    print("\n💡 USER INSIGHTS:")
    print("-" * 60)
    insights = plagiarism_system.generate_plagiarism_insights("user_001")
    
    print(f"  User: {insights['user_id']}")
    print(f"  Total Submissions: {insights['total_submissions']}")
    print(f"  Average Originality: {insights['average_originality']:.2%}")
    print(f"  Plagiarism Level: {insights['plagiarism_level'].upper()}")
    
    if insights['improvement_areas']:
        print("\n  Improvement Areas:")
        for area in insights['improvement_areas']:
            print(f"    • {area}")
    
    if insights['skill_recommendations']:
        print("\n  Skill Recommendations:")
        for rec in insights['skill_recommendations']:
            print(f"    • {rec['skill']}: {rec['reason']}")
            print(f"      Action: {rec['action']}")
    
    # Show overall statistics
    print("\n📊 SYSTEM STATISTICS:")
    print("-" * 60)
    stats = plagiarism_system.get_plagiarism_stats()
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Average Similarity: {stats['average_similarity']:.2%}")
    print(f"  Documents with Paraphrasing: {stats['paraphrasing_detected']}")
    print(f"  High Risk Documents: {stats['high_risk_documents']}")
    
    if 'plagiarism_distribution' in stats:
        print("\n  Plagiarism Distribution:")
        for level, count in stats['plagiarism_distribution'].items():
            if count > 0:
                print(f"    • {level.title()}: {count} documents")
    
    # Show skill wallet integration
    print("\n🎯 SKILL WALLET INTEGRATION:")
    print("-" * 60)
    user_wallet = skill_manager.get_skill_wallet("user_001")
    if user_wallet:
        print(f"  Total Skills: {len(user_wallet.skills)}")
        print(f"  Skill Points: {user_wallet.total_skill_points}")
        print("\n  Skills Earned:")
        for skill in user_wallet.skills.values():
            if skill.skill_id.startswith('skill_plag_'):
                print(f"    • {skill.name}: {skill.level.value.title()}")
    
    print("\n✨ Demonstration complete! ✨")


if __name__ == "__main__":
    demo_plagiarism_system()
