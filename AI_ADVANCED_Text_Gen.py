"""
AI-Generated Text Detection Integration
Advanced system for detecting AI-generated content using multiple techniques
Integrated with plagiarism detection, skill wallet, and sustainability systems
"""

import re
import math
import json
import hashlib
import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum
import random
import statistics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

# Download required NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


class AIDetectionLevel(Enum):
    """Level of AI-generated content detection confidence"""
    HUMAN = "human"  # 0-20% AI probability
    LIKELY_HUMAN = "likely_human"  # 20-40% AI probability
    UNCERTAIN = "uncertain"  # 40-60% AI probability
    LIKELY_AI = "likely_ai"  # 60-80% AI probability
    AI = "ai"  # 80-100% AI probability


class AIGenerationModel(Enum):
    """Potential AI models that generated the text"""
    GPT = "gpt"
    CLAUDE = "claude"
    BARD = "bard"
    LLAMA = "llama"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class AITextFeatures:
    """Features extracted from text for AI detection"""
    text_length: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0  # Type-Token Ratio
    burstiness_score: float = 0.0  # Variation in sentence length
    perplexity_score: float = 0.0
    repetition_score: float = 0.0
    punctuation_density: float = 0.0
    capitalization_ratio: float = 0.0
    transition_word_frequency: float = 0.0
    ngram_diversity: float = 0.0
    semantic_coherence: float = 0.0
    sentiment_consistency: float = 0.0
    syntactic_complexity: float = 0.0
    
    # Advanced features
    ai_typical_phrases: int = 0
    human_typical_phrases: int = 0
    fact_density: float = 0.0
    hedge_word_frequency: float = 0.0
    conjunction_frequency: float = 0.0


@dataclass
class AIDetectionResult:
    """Result of AI text detection analysis"""
    document_id: str
    text: str
    ai_probability: float  # 0.0 to 1.0
    detection_level: AIDetectionLevel
    predicted_model: AIGenerationModel
    features: AITextFeatures
    feature_importance: Dict[str, float] = field(default_factory=dict)
    suspicious_patterns: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    human_score: float = 0.0  # 1 - ai_probability


@dataclass
class TrainingData:
    """Training data for AI detection model"""
    human_texts: List[str] = field(default_factory=list)
    ai_texts: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    model: Any = None
    vectorizer: Any = None
    trained: bool = False
    accuracy: float = 0.0


class AIFeatureExtractor:
    """Extract features for AI text detection"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.common_ai_phrases = self._initialize_ai_phrases()
        self.human_typical_phrases = self._initialize_human_phrases()
        self.transition_words = self._initialize_transition_words()
        self.hedge_words = self._initialize_hedge_words()
        self.conjunctions = self._initialize_conjunctions()
    
    def _initialize_ai_phrases(self) -> List[str]:
        """Initialize phrases commonly used by AI"""
        return [
            "in conclusion", "to summarize", "furthermore", "moreover",
            "it is important to note", "it is crucial", "it is essential",
            "in today's world", "in the modern era", "as a result",
            "consequently", "therefore", "thus", "hence",
            "additionally", "in addition", "on the other hand",
            "however", "nevertheless", "nonetheless", "whereas",
            "as previously mentioned", "as discussed above",
            "to begin with", "firstly", "secondly", "finally",
            "not only", "but also", "in fact", "indeed",
            "one might argue", "it could be argued",
            "this suggests", "this indicates", "this demonstrates",
            "overall", "ultimately", "overall",
            "in particular", "specifically", "namely",
            "for instance", "for example", "such as",
            "in other words", "that is", "to put it differently"
        ]
    
    def _initialize_human_phrases(self) -> List[str]:
        """Initialize phrases commonly used by humans (informal)"""
        return [
            "I think", "I believe", "in my opinion", "from my perspective",
            "honestly", "actually", "basically", "literally",
            "you know", "I mean", "sort of", "kind of",
            "I feel", "I guess", "I suppose", "perhaps",
            "maybe", "probably", "seems like", "feels like",
            "I'd say", "I'd argue", "if you ask me",
            "to be honest", "to be fair", "to tell the truth",
            "at the end of the day", "when it comes down to it",
            "no way", "seriously", "for real", "oh well",
            "anyway", "so yeah", "like I said"
        ]
    
    def _initialize_transition_words(self) -> List[str]:
        """Initialize transition words"""
        return [
            "however", "therefore", "moreover", "furthermore",
            "additionally", "consequently", "accordingly", "thus",
            "hence", "nevertheless", "nonetheless", "whereas",
            "meanwhile", "subsequently", "ultimately", "overall",
            "in contrast", "on the contrary", "in addition",
            "as a result", "for example", "for instance"
        ]
    
    def _initialize_hedge_words(self) -> List[str]:
        """Initialize hedge words"""
        return [
            "might", "may", "could", "would", "should",
            "possibly", "probably", "perhaps", "maybe",
            "apparently", "seemingly", "arguably",
            "approximately", "roughly", "around", "about",
            "suggests", "indicates", "implies"
        ]
    
    def _initialize_conjunctions(self) -> List[str]:
        """Initialize conjunctions"""
        return [
            "and", "or", "but", "nor", "for", "yet", "so",
            "although", "because", "since", "while", "whereas",
            "unless", "if", "when", "where", "which", "who", "whom",
            "whose", "that", "as", "than", "whether", "wherever"
        ]
    
    def extract_features(self, text: str) -> AITextFeatures:
        """Extract all features from text for AI detection"""
        features = AITextFeatures()
        
        # Basic text statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features.text_length = len(text)
        
        if words:
            features.avg_word_length = sum(len(w) for w in words) / len(words)
        
        if sentences:
            features.avg_sentence_length = len(words) / len(sentences)
        
        # Vocabulary richness (Type-Token Ratio)
        unique_words = set(words)
        features.vocabulary_richness = len(unique_words) / len(words) if words else 0
        
        # Burstiness score (variation in sentence length)
        if len(sentences) > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            features.burstiness_score = statistics.stdev(sent_lengths) / (statistics.mean(sent_lengths) + 1e-6)
        
        # Repetition score (frequency of repeated words)
        word_freq = Counter(words)
        repeated_count = sum(1 for w, c in word_freq.items() if c > 2)
        features.repetition_score = repeated_count / len(unique_words) if unique_words else 0
        
        # Punctuation density
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        features.punctuation_density = punctuation_count / len(text) if text else 0
        
        # Capitalization ratio
        capital_count = sum(1 for c in text if c.isupper())
        features.capitalization_ratio = capital_count / len(text) if text else 0
        
        # Transition word frequency
        transition_count = sum(1 for w in words if w in self.transition_words)
        features.transition_word_frequency = transition_count / len(words) if words else 0
        
        # N-gram diversity
        if len(words) > 1:
            bigrams = list(ngrams(words, 2))
            unique_bigrams = set(bigrams)
            features.ngram_diversity = len(unique_bigrams) / len(bigrams) if bigrams else 0
        
        # Semantic coherence (simplified - based on topic consistency)
        features.semantic_coherence = self._compute_semantic_coherence(text)
        
        # Sentiment consistency
        features.sentiment_consistency = self._compute_sentiment_consistency(text)
        
        # Syntactic complexity (average clause length)
        features.syntactic_complexity = self._compute_syntactic_complexity(text)
        
        # AI-typical phrase count
        features.ai_typical_phrases = sum(1 for phrase in self.common_ai_phrases if phrase in text.lower())
        
        # Human-typical phrase count
        features.human_typical_phrases = sum(1 for phrase in self.human_typical_phrases if phrase in text.lower())
        
        # Fact density (approximated by number of numbers and dates)
        numbers = sum(1 for w in words if w.isdigit())
        dates = sum(1 for w in words if '/' in w or '-' in w or w.endswith('th'))
        features.fact_density = (numbers + dates) / len(words) if words else 0
        
        # Hedge word frequency
        hedge_count = sum(1 for w in words if w in self.hedge_words)
        features.hedge_word_frequency = hedge_count / len(words) if words else 0
        
        # Conjunction frequency
        conj_count = sum(1 for w in words if w in self.conjunctions)
        features.conjunction_frequency = conj_count / len(words) if words else 0
        
        # Perplexity score (simplified - based on n-gram probability)
        features.perplexity_score = self._compute_perplexity(text)
        
        return features
    
    def _compute_semantic_coherence(self, text: str) -> float:
        """Compute semantic coherence (topic consistency)"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0
        
        # Simplified: compare word overlap between sentences
        word_sets = [set(word_tokenize(s.lower())) for s in sentences]
        
        total_overlap = 0
        for i in range(len(word_sets) - 1):
            overlap = len(word_sets[i].intersection(word_sets[i+1]))
            union = len(word_sets[i].union(word_sets[i+1]))
            total_overlap += overlap / union if union > 0 else 0
        
        return total_overlap / (len(sentences) - 1) if len(sentences) > 1 else 1.0
    
    def _compute_sentiment_consistency(self, text: str) -> float:
        """Compute sentiment consistency (simplified)"""
        # Simple rule: random sentiment variation
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0
        
        # Count positive/negative words (simplified)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'better', 'positive'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'worse', 'negative'}
        
        sentiments = []
        for sent in sentences:
            words = set(word_tokenize(sent.lower()))
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            if pos_count > neg_count:
                sentiments.append(1)
            elif neg_count > pos_count:
                sentiments.append(-1)
            else:
                sentiments.append(0)
        
        # Check consistency (low variance in sentiment)
        if sentiments:
            variance = statistics.variance(sentiments) if len(sentiments) > 1 else 0
            return 1.0 - min(1.0, variance / 2)
        return 1.0
    
    def _compute_syntactic_complexity(self, text: str) -> float:
        """Compute syntactic complexity (average clause length)"""
        # Simplified: count commas, semicolons, and conjunctions as clause markers
        clauses = re.split(r'[,;]| and | or | but ', text)
        if len(clauses) < 2:
            return 1.0
        
        clause_lengths = [len(c.split()) for c in clauses]
        return statistics.mean(clause_lengths) / 10  # Normalized
    
    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity score (simplified)"""
        words = word_tokenize(text.lower())
        if len(words) < 2:
            return 1.0
        
        # Simplified: compute probability distribution of n-grams
        ngram_counts = Counter(zip(words[:-1], words[1:]))
        total_ngrams = len(words) - 1
        
        if total_ngrams == 0:
            return 1.0
        
        # Calculate entropy
        entropy = 0
        for count in ngram_counts.values():
            prob = count / total_ngrams
            entropy += -prob * math.log(prob + 1e-6)
        
        # Perplexity = 2^entropy (simplified)
        perplexity = math.exp(entropy) if entropy > 0 else 1.0
        return min(1.0, perplexity / 100)  # Normalized


class AIDetectionClassifier:
    """Machine learning classifier for AI text detection"""
    
    def __init__(self):
        self.feature_extractor = AIFeatureExtractor()
        self.training_data = TrainingData()
        self.is_trained = False
        
        # Generate synthetic training data for demonstration
        self._generate_synthetic_training_data()
    
    def _generate_synthetic_training_data(self):
        """Generate synthetic training data for demonstration"""
        # Human-written text patterns
        human_samples = [
            "I think climate change is really important. We need to do something about it soon.",
            "In my opinion, renewable energy is the way to go. Solar and wind power are getting cheaper.",
            "I've noticed that plastic pollution is becoming a huge problem. We should reduce our use of single-use plastics.",
            "From my experience, recycling programs work best when communities are involved.",
            "Honestly, I believe that electric vehicles are the future of transportation.",
            "It seems like sustainable farming practices are gaining more attention these days.",
            "I feel that we should be more mindful of our water consumption.",
            "To be honest, I'm not sure if we're doing enough to combat deforestation.",
            "I'd argue that government policies play a crucial role in environmental protection.",
            "Personally, I try to reduce my carbon footprint by walking and biking more."
        ]
        
        # AI-written text patterns
        ai_samples = [
            "Climate change represents one of the most significant challenges confronting contemporary society.",
            "The implementation of renewable energy technologies is essential for sustainable development.",
            "Plastic pollution poses a substantial threat to marine ecosystems and biodiversity.",
            "Recycling initiatives demonstrate considerable efficacy when supported by community engagement.",
            "Electric vehicles constitute a paradigm shift in automotive technology and environmental policy.",
            "Sustainable agricultural practices are increasingly recognized as vital for food security.",
            "Water conservation measures are imperative for addressing global resource scarcity.",
            "Deforestation mitigation strategies must be implemented through international cooperation.",
            "Environmental protection requires comprehensive policy frameworks and regulatory oversight.",
            "Carbon footprint reduction can be achieved through systemic changes in consumption patterns."
        ]
        
        # Add more varied samples
        for _ in range(20):
            # Generate variations of human text
            human_variants = [
                f"I really think that {random.choice(['climate action', 'recycling', 'solar power', 'conservation'])} is {random.choice(['important', 'crucial', 'essential'])}.",
                f"We {random.choice(['should', 'need to', 'must'])} do more about {random.choice(['pollution', 'waste', 'emissions'])}.",
                f"It's {random.choice(['amazing', 'interesting', 'concerning'])} how {random.choice(['fast', 'slowly', 'suddenly'])} things are changing."
            ]
            human_samples.extend(human_variants)
            
            # Generate variations of AI text
            ai_variants = [
                f"The efficacy of {random.choice(['sustainable practices', 'environmental policy', 'green technology'])} is well-documented.",
                f"Comprehensive analysis demonstrates that {random.choice(['renewable energy', 'conservation efforts', 'waste management'])} yields significant benefits.",
                f"Research indicates a correlation between {random.choice(['carbon emissions', 'resource consumption', 'habitat loss'])} and economic development."
            ]
            ai_samples.extend(ai_variants)
        
        self.training_data.human_texts = human_samples
        self.training_data.ai_texts = ai_samples
        
        # Create labels (0 for human, 1 for AI)
        human_labels = [0] * len(human_samples)
        ai_labels = [1] * len(ai_samples)
        
        self.training_data.labels = human_labels + ai_labels
        
        # Extract features
        all_texts = human_samples + ai_samples
        self.training_data.features = self._extract_feature_matrix(all_texts)
        
        # Train the model
        self.train()
    
    def _extract_feature_matrix(self, texts: List[str]) -> np.ndarray:
        """Extract feature matrix from texts"""
        features_list = []
        
        for text in texts:
            features = self.feature_extractor.extract_features(text)
            
            # Convert features to vector
            feature_vector = [
                features.text_length / 1000,  # Normalize
                features.avg_word_length,
                features.avg_sentence_length / 20,  # Normalize
                features.vocabulary_richness,
                features.burstiness_score,
                features.repetition_score,
                features.punctuation_density * 100,
                features.capitalization_ratio * 100,
                features.transition_word_frequency,
                features.ngram_diversity,
                features.semantic_coherence,
                features.sentiment_consistency,
                features.syntactic_complexity,
                features.ai_typical_phrases / 20,  # Normalize
                features.human_typical_phrases / 20,  # Normalize
                features.fact_density * 10,
                features.hedge_word_frequency,
                features.conjunction_frequency,
                features.perplexity_score
            ]
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def train(self):
        """Train the AI detection model"""
        if len(self.training_data.labels) < 10:
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_data.features,
            self.training_data.labels,
            test_size=0.2,
            random_state=42
        )
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.training_data.model = clf
        self.training_data.accuracy = accuracy
        self.training_data.trained = True
        self.is_trained = True
    
    def predict(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Predict if text is AI-generated"""
        if not self.is_trained:
            return 0.5, {}
        
        # Extract features
        features = self.feature_extractor.extract_features(text)
        feature_vector = self._extract_feature_matrix([text])[0]
        
        # Get prediction probability
        prob = self.training_data.model.predict_proba([feature_vector])[0]
        ai_probability = prob[1]  # Probability of being AI
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.training_data.model, 'feature_importances_'):
            feature_names = [
                'text_length', 'avg_word_length', 'avg_sentence_length',
                'vocabulary_richness', 'burstiness_score', 'repetition_score',
                'punctuation_density', 'capitalization_ratio', 'transition_word_frequency',
                'ngram_diversity', 'semantic_coherence', 'sentiment_consistency',
                'syntactic_complexity', 'ai_typical_phrases', 'human_typical_phrases',
                'fact_density', 'hedge_word_frequency', 'conjunction_frequency',
                'perplexity_score'
            ]
            
            for i, importance in enumerate(self.training_data.model.feature_importances_):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = importance
        
        # Identify suspicious patterns
        suspicious_patterns = self._identify_suspicious_patterns(text, features)
        
        return ai_probability, {
            'feature_importance': feature_importance,
            'suspicious_patterns': suspicious_patterns,
            'features': features
        }
    
    def _identify_suspicious_patterns(self, text: str, features: AITextFeatures) -> List[str]:
        """Identify suspicious patterns in text"""
        patterns = []
        
        # Check for excessive transition words
        if features.transition_word_frequency > 0.15:
            patterns.append("Unusually high frequency of transition words (AI characteristic)")
        
        # Check for low vocabulary richness
        if features.vocabulary_richness < 0.4:
            patterns.append("Low vocabulary diversity (AI characteristic)")
        
        # Check for high repetition
        if features.repetition_score > 0.5:
            patterns.append("High word repetition (AI characteristic)")
        
        # Check for AI-typical phrases
        if features.ai_typical_phrases > 3:
            patterns.append(f"Contains {features.ai_typical_phrases} AI-typical phrases")
        
        # Check for high punctuation density
        if features.punctuation_density > 0.15:
            patterns.append("High punctuation density (AI characteristic)")
        
        # Check for high conjunction frequency
        if features.conjunction_frequency > 0.15:
            patterns.append("High conjunction frequency (AI characteristic)")
        
        # Check for low human-typical phrases
        if features.human_typical_phrases < 1:
            patterns.append("Few human-typical phrases detected")
        
        return patterns


class AITextDetectionSystem:
    """Main AI text detection system with integration features"""
    
    def __init__(self, sustainability_system=None, skill_wallet_manager=None, plagiarism_system=None):
        self.classifier = AIDetectionClassifier()
        self.sustainability_system = sustainability_system
        self.skill_wallet_manager = skill_wallet_manager
        self.plagiarism_system = plagiarism_system
        
        # Storage
        self.detection_results: Dict[str, AIDetectionResult] = {}
        self.user_submissions: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.total_analyzed = 0
        self.ai_count = 0
        self.human_count = 0
        
        # Initialize AI detection skills
        self._initialize_ai_detection_skills()
        
        # Generate some sample AI text for demonstration
        self.sample_ai_texts = self._generate_sample_ai_texts()
        self.sample_human_texts = self._generate_sample_human_texts()
    
    def _initialize_ai_detection_skills(self):
        """Initialize skills related to AI detection and content authenticity"""
        if self.skill_wallet_manager:
            ai_skills = [
                {
                    'id': 'skill_ai_001',
                    'name': 'AI Detection',
                    'description': 'Ability to identify AI-generated content',
                    'category': SkillCategory.TECHNICAL,
                    'level': SkillLevel.BEGINNER
                },
                {
                    'id': 'skill_ai_002',
                    'name': 'Content Authenticity',
                    'description': 'Understanding of authentic content creation',
                    'category': SkillCategory.COMMUNICATION,
                    'level': SkillLevel.BEGINNER
                },
                {
                    'id': 'skill_ai_003',
                    'name': 'AI Literacy',
                    'description': 'Understanding of AI capabilities and limitations',
                    'category': SkillCategory.TECHNICAL,
                    'level': SkillLevel.BEGINNER
                }
            ]
            
            for skill_data in ai_skills:
                skill = Skill(
                    skill_id=skill_data['id'],
                    name=skill_data['name'],
                    description=skill_data['description'],
                    category=skill_data['category'],
                    level=skill_data['level']
                )
                self.skill_wallet_manager.skill_definitions[skill.skill_id] = skill
    
    def _generate_sample_ai_texts(self) -> List[str]:
        """Generate sample AI-written texts for demonstration"""
        return [
            "Artificial intelligence represents a paradigm shift in computational "
            "capabilities, offering unprecedented opportunities for automation and "
            "cognitive augmentation across multiple sectors of society.",
            
            "The integration of machine learning algorithms into healthcare systems has "
            "demonstrated substantial improvements in diagnostic accuracy and patient "
            "outcome prediction.",
            
            "Sustainable energy technologies are rapidly evolving, with solar "
            "photovoltaic systems achieving record efficiency levels and wind turbines "
            "scaling to unprecedented capacities.",
            
            "Climate change mitigation strategies require comprehensive policy "
            "frameworks that address both emissions reduction and adaptation measures "
            "for vulnerable communities.",
            
            "Quantum computing promises to revolutionize computational chemistry, "
            "enabling precise simulation of molecular interactions that were previously "
            "computationally intractable."
        ]
    
    def _generate_sample_human_texts(self) -> List[str]:
        """Generate sample human-written texts for demonstration"""
        return [
            "I think AI is really changing how we do things. It's pretty amazing what "
            "computers can do now, but I'm also a bit worried about what it means for "
            "jobs.",
            
            "We've been trying to get solar panels installed at our school. It's been "
            "quite a process dealing with all the paperwork and approvals, but I think "
            "it'll be worth it in the end.",
            
            "Healthcare is getting better with technology, but I still think there's "
            "something special about human doctors and nurses. They understand things "
            "that computers might miss.",
            
            "I've been trying to reduce my carbon footprint lately. It's not easy, but "
            "little things like using reusable bags and walking more actually make a "
            "difference.",
            
            "To be honest, I'm not sure about all this AI stuff. Some of it seems "
            "really cool, but other parts make me a bit uncomfortable. I guess we'll "
            "see how it all plays out."
        ]
    
    def detect_ai_text(self, text: str, user_id: str = None, 
                      document_id: str = None) -> AIDetectionResult:
        """Detect if text is AI-generated"""
        if not document_id:
            document_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        # Get prediction
        ai_probability, prediction_data = self.classifier.predict(text)
        
        # Determine detection level
        detection_level = self._get_detection_level(ai_probability)
        
        # Determine predicted model (simplified)
        predicted_model = self._predict_ai_model(text, ai_probability)
        
        # Create result
        features = prediction_data.get('features', AIFeatureExtractor().extract_features(text))
        feature_importance = prediction_data.get('feature_importance', {})
        suspicious_patterns = prediction_data.get('suspicious_patterns', [])
        
        result = AIDetectionResult(
            document_id=document_id,
            text=text[:1000] + "..." if len(text) > 1000 else text,
            ai_probability=ai_probability,
            detection_level=detection_level,
            predicted_model=predicted_model,
            features=features,
            feature_importance=feature_importance,
            suspicious_patterns=suspicious_patterns,
            confidence_score=self._calculate_confidence(ai_probability, features),
            human_score=1 - ai_probability
        )
        
        # Store result
        self.detection_results[document_id] = result
        
        if user_id:
            self.user_submissions[user_id].append(document_id)
            
            # Update sustainability system
            if self.sustainability_system:
                self._update_sustainability_for_ai_detection(user_id, result)
            
            # Update skill wallet
            if self.skill_wallet_manager:
                self._update_skill_wallet_for_ai_detection(user_id, result)
            
            # Integrate with plagiarism system
            if self.plagiarism_system:
                self._update_plagiarism_for_ai_detection(user_id, result)
        
        # Update statistics
        self.total_analyzed += 1
        if ai_probability > 0.5:
            self.ai_count += 1
        else:
            self.human_count += 1
        
        return result
    
    def _get_detection_level(self, probability: float) -> AIDetectionLevel:
        """Determine detection level based on probability"""
        if probability < 0.20:
            return AIDetectionLevel.HUMAN
        elif probability < 0.40:
            return AIDetectionLevel.LIKELY_HUMAN
        elif probability < 0.60:
            return AIDetectionLevel.UNCERTAIN
        elif probability < 0.80:
            return AIDetectionLevel.LIKELY_AI
        else:
            return AIDetectionLevel.AI
    
    def _predict_ai_model(self, text: str, probability: float) -> AIGenerationModel:
        """Predict which AI model generated the text"""
        if probability < 0.5:
            return AIGenerationModel.UNKNOWN
        
        # Simplified model detection based on style patterns
        text_lower = text.lower()
        
        # GPT patterns
        if any(word in text_lower for word in ['furthermore', 'moreover', 'consequently', 'additionally']):
            return AIGenerationModel.GPT
        
        # Claude patterns (often more conversational)
        if any(word in text_lower for word in ['would', 'could', 'might', 'perhaps', 'maybe']):
            return AIGenerationModel.CLAUDE
        
        # Bard patterns (often more factual and structured)
        if any(word in text_lower for word in ['specifically', 'particularly', 'notably', 'significantly']):
            return AIGenerationModel.BARD
        
        # Llama patterns (often more direct)
        if any(word in text_lower for word in ['clearly', 'obviously', 'definitely', 'certainly']):
            return AIGenerationModel.LLAMA
        
        return AIGenerationModel.OTHER
    
    def _calculate_confidence(self, probability: float, features: AITextFeatures) -> float:
        """Calculate confidence score for the detection"""
        # Higher confidence when probability is at extremes
        distance_from_half = abs(probability - 0.5) * 2
        
        # Adjust based on feature quality
        feature_quality = 0.5  # Base quality
        
        # Good features increase confidence
        if features.vocabulary_richness > 0.3 and features.vocabulary_richness < 0.7:
            feature_quality += 0.2
        
        if features.burstiness_score > 0.3:
            feature_quality += 0.15
        
        if features.ngram_diversity > 0.5:
            feature_quality += 0.15
        
        confidence = min(1.0, distance_from_half * 0.8 + feature_quality * 0.2)
        return min(1.0, max(0.0, confidence))
    
    def _update_sustainability_for_ai_detection(self, user_id: str, result: AIDetectionResult):
        """Update sustainability system based on AI detection"""
        user = self.sustainability_system.get_user(user_id)
        if not user:
            return
        
        # Award points for detecting AI text (ethical AI use)
        if result.detection_level in [AIDetectionLevel.AI, AIDetectionLevel.LIKELY_AI]:
            # User identified AI text - award ethical AI usage points
            points = 20 if result.detection_level == AIDetectionLevel.AI else 10
            user.total_points += points
            user.add_xp(points // 2)
            print(f"  🤖 Awarded {points} sustainability points for AI text detection!")
        
        # Bonus for accurate detection (if probability is high and correct)
        if result.confidence_score > 0.8:
            bonus = 15
            user.total_points += bonus
            user.add_xp(bonus // 2)
            print(f"  ⭐ Bonus {bonus} points for high-confidence detection!")
    
    def _update_skill_wallet_for_ai_detection(self, user_id: str, result: AIDetectionResult):
        """Update skill wallet based on AI detection performance"""
        wallet = self.skill_wallet_manager.get_skill_wallet(user_id)
        if not wallet:
            return
        
        # Award AI detection skill for active detection
        if result.detection_level in [AIDetectionLevel.AI, AIDetectionLevel.LIKELY_AI]:
            skill_id = 'skill_ai_001'
            if skill_id in wallet.skills:
                skill = wallet.skills[skill_id]
                skill.add_experience(15)
                if skill.experience_points > 100:
                    skill.level = SkillLevel.INTERMEDIATE
                    print(f"  📈 Upgraded AI Detection skill to INTERMEDIATE!")
            else:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  🎯 Awarded AI Detection skill!")
        
        # Award AI literacy for consistent engagement
        user_submissions = self.user_submissions.get(user_id, [])
        if len(user_submissions) > 3:
            skill_id = 'skill_ai_003'
            if skill_id not in wallet.skills:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  🧠 Awarded AI Literacy skill!")
    
    def _update_plagiarism_for_ai_detection(self, user_id: str, result: AIDetectionResult):
        """Update plagiarism system with AI detection results"""
        # If text is AI-generated, flag it in plagiarism system
        if result.detection_level in [AIDetectionLevel.AI, AIDetectionLevel.LIKELY_AI]:
            # Store AI detection in plagiarism system
            if hasattr(self.plagiarism_system, 'ai_detections'):
                self.plagiarism_system.ai_detections[result.document_id] = {
                    'user_id': user_id,
                    'ai_probability': result.ai_probability,
                    'detected_at': result.timestamp.isoformat()
                }
    
    def analyze_text_comparison(self, user_id: str, text1: str, text2: str) -> Dict:
        """Compare two texts and analyze AI involvement"""
        result1 = self.detect_ai_text(text1, user_id)
        result2 = self.detect_ai_text(text2, user_id)
        
        # Also check plagiarism if available
        plagiarism_analysis = None
        if self.plagiarism_system:
            doc_analysis = self.plagiarism_system.detector.analyze_document(text1, text2)
            plagiarism_analysis = {
                'similarity': doc_analysis.overall_similarity,
                'plagiarism_level': doc_analysis.plagiarism_level.value,
                'high_risk_sentences': len(doc_analysis.high_risk_sentences)
            }
        
        return {
            'text1_ai_probability': result1.ai_probability,
            'text1_level': result1.detection_level.value,
            'text2_ai_probability': result2.ai_probability,
            'text2_level': result2.detection_level.value,
            'both_ai': result1.ai_probability > 0.5 and result2.ai_probability > 0.5,
            'ai_human': result1.ai_probability > 0.5 and result2.ai_probability <= 0.5,
            'human_ai': result1.ai_probability <= 0.5 and result2.ai_probability > 0.5,
            'plagiarism_analysis': plagiarism_analysis,
            'recommendations': self._generate_comparison_recommendations(result1, result2)
        }
    
    def _generate_comparison_recommendations(self, result1: AIDetectionResult, 
                                            result2: AIDetectionResult) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []
        
        if result1.ai_probability > 0.5 and result2.ai_probability > 0.5:
            recommendations.append("Both texts appear AI-generated - consider adding more personal insights")
            recommendations.append("Humanize your content by adding personal experiences and opinions")
        elif result1.ai_probability > 0.5 and result2.ai_probability <= 0.5:
            recommendations.append("Text1 appears AI-generated while Text2 appears human-written")
            recommendations.append("Try to make AI-generated text more natural by using casual language")
        elif result1.ai_probability <= 0.5 and result2.ai_probability > 0.5:
            recommendations.append("Text1 appears human-written while Text2 appears AI-generated")
            recommendations.append("Compare the differences in style and tone")
        else:
            recommendations.append("Both texts appear human-written - great job!")
            recommendations.append("Keep developing your authentic writing style")
        
        return recommendations
    
    def get_user_ai_stats(self, user_id: str) -> Dict:
        """Get AI detection statistics for a user"""
        submissions = self.user_submissions.get(user_id, [])
        
        if not submissions:
            return {"message": "No submissions found"}
        
        results = [self.detection_results[doc_id] for doc_id in submissions 
                  if doc_id in self.detection_results]
        
        if not results:
            return {"message": "No detection results available"}
        
        # Calculate statistics
        ai_results = [r for r in results if r.ai_probability > 0.5]
        human_results = [r for r in results if r.ai_probability <= 0.5]
        
        avg_ai_prob = sum(r.ai_probability for r in results) / len(results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        
        # Model distribution
        model_dist = Counter(r.predicted_model.value for r in results)
        
        return {
            'user_id': user_id,
            'total_submissions': len(results),
            'ai_detected': len(ai_results),
            'human_detected': len(human_results),
            'ai_percentage': (len(ai_results) / len(results)) * 100,
            'average_ai_probability': avg_ai_prob,
            'average_confidence': avg_confidence,
            'model_distribution': dict(model_dist),
            'suspicious_patterns': list(set(
                pattern for r in results for pattern in r.suspicious_patterns
            ))[:10]
        }
    
    def get_system_stats(self) -> Dict:
        """Get overall system statistics"""
        if self.total_analyzed == 0:
            return {"message": "No analyses performed yet"}
        
        return {
            'total_analyses': self.total_analyzed,
            'ai_count': self.ai_count,
            'human_count': self.human_count,
            'ai_percentage': (self.ai_count / self.total_analyzed) * 100,
            'human_percentage': (self.human_count / self.total_analyzed) * 100,
            'model_accuracy': self.classifier.training_data.accuracy,
            'model_trained': self.classifier.is_trained,
            'users_analyzed': len(self.user_submissions)
        }


# Demo function
def demo_ai_detection_system():
    """Demonstrate the AI detection system"""
    print("🤖 AI-GENERATED TEXT DETECTION SYSTEM 🤖")
    print("=" * 80)
    
    # Initialize integrated systems
    sustainability = SustainabilityGamification()
    skill_manager = SkillWalletProjectManager(sustainability)
    plagiarism_system = IntegratedPlagiarismSystem(sustainability, skill_manager)
    ai_system = AITextDetectionSystem(sustainability, skill_manager, plagiarism_system)
    
    # Register a user
    print("\n📝 Registering user...")
    user = sustainability.register_user("user_001", "EcoAlice")
    wallet = skill_manager.create_skill_wallet("user_001")
    print(f"✅ User registered: {user.username}")
    
    # Demo texts
    human_text = """
    I've been thinking a lot about renewable energy lately. It's really interesting 
    how much solar power has improved over the years. When I look at my own energy 
    consumption, I realize I could probably do more to reduce it. 
    
    I started using LED bulbs and unplugging devices when they're not in use. 
    It's not a huge change, but it feels good to do something, you know? 
    I think if more people made small changes, it would really add up.
    """
    
    ai_text = """
    Renewable energy technologies have experienced exponential growth over the past 
    decade, with solar photovoltaic systems achieving remarkable cost reductions 
    and efficiency improvements. The integration of these technologies into existing 
    power grids presents both opportunities and challenges. 
    
    Policy frameworks that incentivize renewable energy adoption, coupled with 
    technological innovations in energy storage, are essential for facilitating 
    the transition to a sustainable energy future. Furthermore, grid modernization 
    initiatives and smart grid technologies play a crucial role in accommodating 
    distributed energy resources.
    """
    
    mixed_text = """
    I think renewable energy is becoming really important these days. Solar and wind 
    power are getting cheaper and more efficient. The deployment of renewable energy 
    technologies has accelerated substantially, driven by both policy incentives and 
    declining costs.
    
    To be honest, I'm pretty excited about the potential. However, there are still 
    challenges that need to be addressed, such as intermittency and grid integration. 
    The integration of variable renewable energy sources requires careful planning 
    and innovative solutions.
    """
    
    # Test 1: Detect AI-generated text
    print("\n🔍 Testing AI Text Detection...")
    print("-" * 60)
    
    print("\n1️⃣ Human Text Detection:")
    result1 = ai_system.detect_ai_text(human_text, "user_001", "doc_001")
    
    print(f"\n📊 Results:")
    print(f"  AI Probability: {result1.ai_probability:.2%}")
    print(f"  Detection Level: {result1.detection_level.value.upper()}")
    print(f"  Confidence: {result1.confidence_score:.2%}")
    print(f"  Predicted Model: {result1.predicted_model.value.upper()}")
    
    if result1.suspicious_patterns:
        print("\n  Suspicious Patterns:")
        for pattern in result1.suspicious_patterns:
            print(f"    • {pattern}")
    
    print("\n2️⃣ AI Text Detection:")
    result2 = ai_system.detect_ai_text(ai_text, "user_001", "doc_002")
    
    print(f"\n📊 Results:")
    print(f"  AI Probability: {result2.ai_probability:.2%}")
    print(f"  Detection Level: {result2.detection_level.value.upper()}")
    print(f"  Confidence: {result2.confidence_score:.2%}")
    print(f"  Predicted Model: {result2.predicted_model.value.upper()}")
    
    if result2.suspicious_patterns:
        print("\n  Suspicious Patterns:")
        for pattern in result2.suspicious_patterns:
            print(f"    • {pattern}")
    
    print("\n3️⃣ Mixed Text Detection:")
    result3 = ai_system.detect_ai_text(mixed_text, "user_001", "doc_003")
    
    print(f"\n📊 Results:")
    print(f"  AI Probability: {result3.ai_probability:.2%}")
    print(f"  Detection Level: {result3.detection_level.value.upper()}")
    print(f"  Confidence: {result3.confidence_score:.2%}")
    print(f"  Predicted Model: {result3.predicted_model.value.upper()}")
    
    # Test 2: Text comparison
    print("\n🔍 Text Comparison Analysis:")
    print("-" * 60)
    
    comparison = ai_system.analyze_text_comparison("user_001", human_text, ai_text)
    print(f"\n  Text1 AI Probability: {comparison['text1_ai_probability']:.2%}")
    print(f"  Text2 AI Probability: {comparison['text2_ai_probability']:.2%}")
    print(f"  Both AI: {comparison['both_ai']}")
    print(f"  AI-Human: {comparison['ai_human']}")
    
    if comparison['plagiarism_analysis']:
        print(f"\n  Plagiarism Analysis:")
        print(f"    Similarity: {comparison['plagiarism_analysis']['similarity']:.2%}")
        print(f"    Level: {comparison['plagiarism_analysis']['plagiarism_level']}")
    
    print("\n  Recommendations:")
    for rec in comparison['recommendations']:
        print(f"    • {rec}")
    
    # Test 3: User stats
    print("\n📊 User AI Detection Statistics:")
    print("-" * 60)
    user_stats = ai_system.get_user_ai_stats("user_001")
    
    print(f"  User: {user_stats['user_id']}")
    print(f"  Total Submissions: {user_stats['total_submissions']}")
    print(f"  AI Detected: {user_stats['ai_detected']}")
    print(f"  Human Detected: {user_stats['human_detected']}")
    print(f"  AI Percentage: {user_stats['ai_percentage']:.1f}%")
    print(f"  Average AI Probability: {user_stats['average_ai_probability']:.2%}")
    print(f"  Average Confidence: {user_stats['average_confidence']:.2%}")
    
    if user_stats.get('model_distribution'):
        print("\n  Model Distribution:")
        for model, count in user_stats['model_distribution'].items():
            print(f"    • {model.upper()}: {count}")
    
    # Test 4: System stats
    print("\n📊 System Statistics:")
    print("-" * 60)
    sys_stats = ai_system.get_system_stats()
    
    print(f"  Total Analyses: {sys_stats['total_analyses']}")
    print(f"  AI Detected: {sys_stats['ai_count']} ({sys_stats['ai_percentage']:.1f}%)")
    print(f"  Human Detected: {sys_stats['human_count']} ({sys_stats['human_percentage']:.1f}%)")
    print(f"  Model Accuracy: {sys_stats['model_accuracy']:.2%}")
    print(f"  Model Trained: {sys_stats['model_trained']}")
    print(f"  Users Analyzed: {sys_stats['users_analyzed']}")
    
    # Test 5: Feature analysis
    print("\n🔍 Feature Importance Analysis:")
    print("-" * 60)
    
    if result2.feature_importance:
        sorted_features = sorted(
            result2.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print("  Top 5 Most Important Features:")
        for feature, importance in sorted_features:
            print(f"    • {feature}: {importance:.3f}")
    
    # Test 6: Skill wallet integration
    print("\n🎯 Skill Wallet Integration:")
    print("-" * 60)
    user_wallet = skill_manager.get_skill_wallet("user_001")
    
    if user_wallet:
        print(f"  Total Skills: {len(user_wallet.skills)}")
        print(f"  Skill Points: {user_wallet.total_skill_points}")
        
        print("\n  Skills Earned:")
        for skill in user_wallet.skills.values():
            if skill.skill_id.startswith('skill_ai_'):
                print(f"    • {skill.name}: {skill.level.value.title()}")
    
    # Test 7: Feature extraction demo
    print("\n🔬 Feature Extraction Demo:")
    print("-" * 60)
    
    feature_extractor = AIFeatureExtractor()
    features = feature_extractor.extract_features(ai_text)
    
    print("  Key Features:")
    print(f"    Vocabulary Richness: {features.vocabulary_richness:.3f}")
    print(f"    Repetition Score: {features.repetition_score:.3f}")
    print(f"    AI-Typical Phrases: {features.ai_typical_phrases}")
    print(f"    Human-Typical Phrases: {features.human_typical_phrases}")
    print(f"    Transition Word Freq: {features.transition_word_frequency:.3f}")
    print(f"    Hedge Word Freq: {features.hedge_word_frequency:.3f}")
    print(f"    Perplexity Score: {features.perplexity_score:.3f}")
    print(f"    Semantic Coherence: {features.semantic_coherence:.3f}")
    
    print("\n✨ Demonstration complete! ✨")


if __name__ == "__main__":
    demo_ai_detection_system()
