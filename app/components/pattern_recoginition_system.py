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
# ── SECTION: INTELLIGENT PATTERN RECOGNITION SYSTEM (Issue #2001) ───────────
# ───────────────────────────────────────────────────────────────────────────────

import re
import uuid
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class PlagiarismPattern:
    """Represents a detected plagiarism pattern"""

    id: str
    name: str
    pattern_type: str  # 'copy_paste', 'paraphrase', 'structural', 'citation', 'hybrid'
    description: str
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    frequency: int
    first_detected: datetime
    last_detected: datetime
    documents: list[str]
    features: dict[str, Any]
    evolution: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "first_detected": self.first_detected.isoformat(),
            "last_detected": self.last_detected.isoformat(),
        }


@dataclass
class RiskPrediction:
    """Risk prediction for a document or group"""

    id: str
    document_name: str
    predicted_risk_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    contributing_factors: list[str]
    confidence: float
    prediction_date: datetime
    recommendations: list[str]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {**asdict(self), "prediction_date": self.prediction_date.isoformat()}


@dataclass
class PatternTrend:
    """Trend analysis for patterns"""

    pattern_id: str
    pattern_name: str
    time_period: str
    frequency_change: float  # percentage change
    new_documents: int
    risk_trend: str  # 'increasing', 'decreasing', 'stable'
    forecast: dict[str, Any]
    metadata: dict = field(default_factory=dict)


# ── Pattern Recognition Engine ────────────────────────────────────────────


class PatternRecognitionEngine:
    """Main pattern recognition and prediction engine"""

    def __init__(self):
        self.patterns: dict[str, PlagiarismPattern] = {}
        self.predictions: dict[str, RiskPrediction] = {}
        self.historical_data: list[dict] = []
        self.trends: dict[str, PatternTrend] = {}
        self.pattern_counts = defaultdict(int)
        self._init_pattern_types()

    def _init_pattern_types(self):
        """Initialize common pattern types"""
        self.pattern_types = {
            "copy_paste": {
                "name": "Copy-Paste Pattern",
                "description": "Direct text copying without modification",
                "severity": "high",
                "threshold": 0.8,
            },
            "paraphrase": {
                "name": "Paraphrasing Pattern",
                "description": "Text rewritten with changed wording",
                "severity": "medium",
                "threshold": 0.6,
            },
            "structural": {
                "name": "Structural Pattern",
                "description": "Similar document structure and organization",
                "severity": "medium",
                "threshold": 0.5,
            },
            "citation": {
                "name": "Citation Pattern",
                "description": "Unusual citation patterns and references",
                "severity": "medium",
                "threshold": 0.4,
            },
            "hybrid": {
                "name": "Hybrid Pattern",
                "description": "Combination of multiple plagiarism techniques",
                "severity": "critical",
                "threshold": 0.7,
            },
        }

    def detect_patterns(
        self, documents: dict[str, str], similarity_matrix: np.ndarray
    ) -> list[PlagiarismPattern]:
        """Detect patterns in document collection"""
        detected_patterns = []

        # Extract features
        features = self._extract_features(documents, similarity_matrix)

        # Detect copy-paste patterns
        copy_paste = self._detect_copy_paste(documents, similarity_matrix)
        if copy_paste:
            detected_patterns.extend(copy_paste)

        # Detect structural patterns
        structural = self._detect_structural(documents)
        if structural:
            detected_patterns.extend(structural)

        # Detect citation patterns
        citation = self._detect_citation(documents)
        if citation:
            detected_patterns.extend(citation)

        # Detect hybrid patterns
        hybrid = self._detect_hybrid(documents, similarity_matrix)
        if hybrid:
            detected_patterns.extend(hybrid)

        # Store patterns
        for pattern in detected_patterns:
            self.patterns[pattern.id] = pattern
            self.pattern_counts[pattern.pattern_type] += 1

        return detected_patterns

    def _extract_features(
        self, documents: dict[str, str], similarity_matrix: np.ndarray
    ) -> dict:
        """Extract features for pattern detection"""
        features = {
            "word_counts": {},
            "sentence_counts": {},
            "avg_word_length": {},
            "unique_words": {},
            "avg_similarity": {},
            "max_similarity": {},
            "document_lengths": {},
        }

        doc_names = list(documents.keys())

        for i, doc_name in enumerate(doc_names):
            content = documents.get(doc_name, "")

            # Word count
            words = content.split()
            features["word_counts"][doc_name] = len(words)

            # Sentence count
            sentences = re.split(r"[.!?]+", content)
            features["sentence_counts"][doc_name] = len(sentences)

            # Average word length
            if words:
                features["avg_word_length"][doc_name] = sum(
                    len(w) for w in words
                ) / len(words)

            # Unique words
            features["unique_words"][doc_name] = len(set(words))

            # Similarity stats
            if i < len(similarity_matrix):
                row = similarity_matrix[i]
                features["avg_similarity"][doc_name] = np.mean(row)
                features["max_similarity"][doc_name] = np.max(row)

            # Document length
            features["document_lengths"][doc_name] = len(content)

        return features

    def _detect_copy_paste(
        self, documents: dict[str, str], similarity_matrix: np.ndarray
    ) -> list[PlagiarismPattern]:
        """Detect copy-paste patterns"""
        patterns = []
        doc_names = list(documents.keys())
        threshold = 0.7

        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                similarity = similarity_matrix[i][j]
                if similarity > threshold:
                    pattern_id = str(uuid.uuid4())
                    pattern = PlagiarismPattern(
                        id=pattern_id,
                        name=f"Copy-Paste: {doc_names[i]} ↔ {doc_names[j]}",
                        pattern_type="copy_paste",
                        description=f"High similarity ({similarity:.1%}) detected between documents",
                        confidence=similarity,
                        severity="high",
                        frequency=1,
                        first_detected=datetime.now(),
                        last_detected=datetime.now(),
                        documents=[doc_names[i], doc_names[j]],
                        features={
                            "similarity_score": similarity,
                            "text_length": len(documents.get(doc_names[i], "")),
                        },
                    )
                    patterns.append(pattern)
                    break  # Limit patterns per document

        return patterns

    def _detect_structural(self, documents: dict[str, str]) -> list[PlagiarismPattern]:
        """Detect structural patterns"""
        patterns = []

        # Extract paragraph counts and patterns
        doc_patterns = {}
        for doc_name, content in documents.items():
            paragraphs = [p for p in content.split("\n\n") if p.strip()]
            para_lengths = [len(p.split()) for p in paragraphs]

            if para_lengths:
                avg_len = sum(para_lengths) / len(para_lengths)
                doc_patterns[doc_name] = {
                    "paragraph_count": len(paragraphs),
                    "avg_paragraph_length": avg_len,
                    "length_variance": (
                        np.var(para_lengths) if len(para_lengths) > 1 else 0
                    ),
                }

        # Find similar structural patterns
        doc_names = list(doc_patterns.keys())
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                doc1 = doc_names[i]
                doc2 = doc_names[j]

                pattern1 = doc_patterns[doc1]
                pattern2 = doc_patterns[doc2]

                # Compare paragraph count
                para_diff = abs(
                    pattern1["paragraph_count"] - pattern2["paragraph_count"]
                )
                para_sim = 1 - (
                    para_diff
                    / max(pattern1["paragraph_count"], pattern2["paragraph_count"])
                )

                # Compare average length
                len_diff = abs(
                    pattern1["avg_paragraph_length"] - pattern2["avg_paragraph_length"]
                )
                max_len = max(
                    pattern1["avg_paragraph_length"], pattern2["avg_paragraph_length"]
                )
                len_sim = 1 - (len_diff / max_len) if max_len > 0 else 0

                overall_sim = (para_sim + len_sim) / 2

                if overall_sim > 0.6:
                    pattern_id = str(uuid.uuid4())
                    pattern = PlagiarismPattern(
                        id=pattern_id,
                        name=f"Structural: {doc1} ↔ {doc2}",
                        pattern_type="structural",
                        description=f"Similar document structure detected ({overall_sim:.1%})",
                        confidence=overall_sim,
                        severity="medium",
                        frequency=1,
                        first_detected=datetime.now(),
                        last_detected=datetime.now(),
                        documents=[doc1, doc2],
                        features={
                            "paragraph_similarity": para_sim,
                            "length_similarity": len_sim,
                        },
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_citation(self, documents: dict[str, str]) -> list[PlagiarismPattern]:
        """Detect citation patterns"""
        patterns = []

        # Extract citations
        citation_pattern = r"\([A-Z][a-z]+,\s*\d{4}\)"
        doc_citations = {}

        for doc_name, content in documents.items():
            citations = re.findall(citation_pattern, content)
            doc_citations[doc_name] = citations

        # Find common citations
        all_citations = set()
        for citations in doc_citations.values():
            all_citations.update(citations)

        # Detect unusual citation patterns
        doc_names = list(doc_citations.keys())
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                doc1 = doc_names[i]
                doc2 = doc_names[j]

                citations1 = set(doc_citations[doc1])
                citations2 = set(doc_citations[doc2])

                if citations1 and citations2:
                    intersection = len(citations1.intersection(citations2))
                    union = len(citations1.union(citations2))
                    overlap = intersection / union if union > 0 else 0

                    if overlap > 0.5:
                        pattern_id = str(uuid.uuid4())
                        pattern = PlagiarismPattern(
                            id=pattern_id,
                            name=f"Citation: {doc1} ↔ {doc2}",
                            pattern_type="citation",
                            description=f"High citation overlap ({overlap:.1%}) detected",
                            confidence=overlap,
                            severity="medium",
                            frequency=1,
                            first_detected=datetime.now(),
                            last_detected=datetime.now(),
                            documents=[doc1, doc2],
                            features={
                                "citation_overlap": overlap,
                                "shared_citations": list(
                                    citations1.intersection(citations2)
                                ),
                            },
                        )
                        patterns.append(pattern)

        return patterns

    def _detect_hybrid(
        self, documents: dict[str, str], similarity_matrix: np.ndarray
    ) -> list[PlagiarismPattern]:
        """Detect hybrid patterns combining multiple techniques"""
        patterns = []

        # Detect complex patterns
        doc_names = list(documents.keys())

        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                doc1 = doc_names[i]
                doc2 = doc_names[j]

                # Get multiple metrics
                text1 = documents.get(doc1, "")
                text2 = documents.get(doc2, "")

                # Lexical similarity
                words1 = set(text1.split())
                words2 = set(text2.split())
                if words1 and words2:
                    lexical_sim = len(words1.intersection(words2)) / len(
                        words1.union(words2)
                    )
                else:
                    lexical_sim = 0

                # Structural similarity
                paras1 = len([p for p in text1.split("\n\n") if p.strip()])
                paras2 = len([p for p in text2.split("\n\n") if p.strip()])
                if paras1 and paras2:
                    struct_sim = 1 - (abs(paras1 - paras2) / max(paras1, paras2))
                else:
                    struct_sim = 0

                # Semantic similarity (from matrix)
                sem_sim = (
                    similarity_matrix[i][j]
                    if i < len(similarity_matrix) and j < len(similarity_matrix[i])
                    else 0
                )

                # Combined score
                combined = lexical_sim * 0.3 + struct_sim * 0.2 + sem_sim * 0.5

                if combined > 0.6:
                    pattern_id = str(uuid.uuid4())
                    pattern = PlagiarismPattern(
                        id=pattern_id,
                        name=f"Hybrid: {doc1} ↔ {doc2}",
                        pattern_type="hybrid",
                        description=f"Complex plagiarism pattern detected (combined score: {combined:.1%})",
                        confidence=combined,
                        severity="critical",
                        frequency=1,
                        first_detected=datetime.now(),
                        last_detected=datetime.now(),
                        documents=[doc1, doc2],
                        features={
                            "lexical_similarity": lexical_sim,
                            "structural_similarity": struct_sim,
                            "semantic_similarity": sem_sim,
                            "combined_score": combined,
                        },
                    )
                    patterns.append(pattern)

        return patterns


# ── Prediction Engine ─────────────────────────────────────────────────────


class PredictionEngine:
    """Predicts future plagiarism risks"""

    def __init__(self, pattern_engine: PatternRecognitionEngine):
        self.pattern_engine = pattern_engine
        self.prediction_model = None
        self.model_trained = False
        self.scaler = StandardScaler()
        self.historical_data = []

    def train_model(self, training_data: list[dict]) -> bool:
        """Train prediction model on historical data"""
        if len(training_data) < 10:
            return False

        try:
            # Prepare features
            features = []
            labels = []

            for item in training_data:
                features.append(
                    [
                        item.get("similarity_score", 0),
                        item.get("document_length", 0),
                        item.get("word_count", 0),
                        item.get("unique_words_ratio", 0),
                        item.get("complexity_score", 0),
                        item.get("pattern_count", 0),
                        item.get("avg_similarity", 0),
                        item.get("max_similarity", 0),
                    ]
                )
                labels.append(item.get("risk_level", 0))

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Train random forest
            self.prediction_model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
            self.prediction_model.fit(features_scaled, labels)
            self.model_trained = True
            return True

        except Exception as e:
            print(f"Model training failed: {e}")
            return False

    def predict_risk(
        self, document_name: str, document_content: str, similarity_scores: list[float]
    ) -> RiskPrediction:
        """Predict risk for a single document"""
        # Extract features
        words = document_content.split()
        sentences = re.split(r"[.!?]+", document_content)

        features = [
            np.mean(similarity_scores) if similarity_scores else 0,
            len(document_content),
            len(words),
            len(set(words)) / len(words) if words else 0,
            sum(1 for w in words if len(w) > 6) / len(words) if words else 0,
            len(self.pattern_engine.patterns),
            np.mean(similarity_scores) if similarity_scores else 0,
            np.max(similarity_scores) if similarity_scores else 0,
        ]

        # Make prediction
        if self.model_trained:
            features_scaled = self.scaler.transform([features])
            prediction = self.prediction_model.predict(features_scaled)[0]
            confidence = np.max(self.prediction_model.predict_proba(features_scaled)[0])
        else:
            # Fallback to heuristic
            prediction = self._heuristic_risk_score(features)
            confidence = 0.7

        # Determine risk level
        risk_level = self._get_risk_level(prediction)

        # Generate recommendations
        recommendations = self._generate_recommendations(features, risk_level)

        # Create prediction
        prediction_id = str(uuid.uuid4())
        risk_prediction = RiskPrediction(
            id=prediction_id,
            document_name=document_name,
            predicted_risk_score=float(prediction),
            risk_level=risk_level,
            contributing_factors=self._get_contributing_factors(features),
            confidence=confidence,
            prediction_date=datetime.now(),
            recommendations=recommendations,
        )

        self.pattern_engine.predictions[prediction_id] = risk_prediction
        return risk_prediction

    def _heuristic_risk_score(self, features: list[float]) -> float:
        """Calculate risk score using heuristic rules"""
        (
            similarity_score,
            doc_length,
            word_count,
            unique_ratio,
            complexity,
            pattern_count,
            avg_sim,
            max_sim,
        ) = features

        # Weighted scoring
        score = 0

        # Similarity contribution
        if avg_sim > 0.5:
            score += avg_sim * 0.3
        if max_sim > 0.7:
            score += max_sim * 0.2

        # Pattern contribution
        if pattern_count > 0:
            score += min(pattern_count * 0.05, 0.3)

        # Document quality
        if unique_ratio < 0.5:
            score += (1 - unique_ratio) * 0.2

        return min(score, 1.0)

    def _get_risk_level(self, score: float) -> str:
        """Convert score to risk level"""
        if score > 0.8:
            return "critical"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"

    def _get_contributing_factors(self, features: list[float]) -> list[str]:
        """Identify contributing risk factors"""
        factors = []
        (
            similarity_score,
            doc_length,
            word_count,
            unique_ratio,
            complexity,
            pattern_count,
            avg_sim,
            max_sim,
        ) = features

        if avg_sim > 0.5:
            factors.append("High average similarity")
        if max_sim > 0.7:
            factors.append("Very high maximum similarity")
        if pattern_count > 3:
            factors.append("Multiple plagiarism patterns detected")
        if unique_ratio < 0.4:
            factors.append("Low vocabulary diversity")
        if complexity < 0.1:
            factors.append("Very low complexity")

        return factors

    def _generate_recommendations(
        self, features: list[float], risk_level: str
    ) -> list[str]:
        """Generate recommendations based on risk level"""
        recommendations = []

        if risk_level in ["high", "critical"]:
            recommendations.extend(
                [
                    "Conduct immediate document review",
                    "Verify all sources and citations",
                    "Check for matching content in other documents",
                    "Consider academic integrity review",
                ]
            )

        if features[4] < 0.1:  # Low complexity
            recommendations.append("Review for potential copy-paste content")

        if features[2] < 50:  # Very short document
            recommendations.append("Document may be too short for reliable analysis")

        if not recommendations:
            recommendations.append("Document appears low risk. Continue monitoring.")

        return recommendations


# ── Pattern Evolution Tracker ─────────────────────────────────────────────


class PatternEvolutionTracker:
    """Tracks evolution of patterns over time"""

    def __init__(self, pattern_engine: PatternRecognitionEngine):
        self.pattern_engine = pattern_engine
        self.evolution_data: dict[str, list[dict]] = defaultdict(list)
        self.trends: dict[str, PatternTrend] = {}

    def track_evolution(self, pattern_id: str, new_data: dict):
        """Track evolution of a pattern"""
        self.evolution_data[pattern_id].append(
            {"timestamp": datetime.now(), "data": new_data}
        )

        # Update trend
        self._update_trend(pattern_id)

    def _update_trend(self, pattern_id: str):
        """Update trend analysis for a pattern"""
        history = self.evolution_data.get(pattern_id, [])
        if len(history) < 2:
            return

        pattern = self.pattern_engine.patterns.get(pattern_id)
        if not pattern:
            return

        # Calculate frequency change
        recent = history[-5:]
        if len(recent) >= 2:
            freq_change = recent[-1]["data"].get("frequency", 0) - recent[0][
                "data"
            ].get("frequency", 0)

            # Determine trend direction
            if freq_change > 0:
                trend_dir = "increasing"
            elif freq_change < 0:
                trend_dir = "decreasing"
            else:
                trend_dir = "stable"

            # Create trend
            trend = PatternTrend(
                pattern_id=pattern_id,
                pattern_name=pattern.name,
                time_period=f"{history[0]['timestamp'].strftime('%Y-%m-%d')} to {history[-1]['timestamp'].strftime('%Y-%m-%d')}",
                frequency_change=freq_change / len(recent) if recent else 0,
                new_documents=recent[-1]["data"].get("new_documents", 0),
                risk_trend=trend_dir,
                forecast=self._generate_forecast(history),
            )

            self.trends[pattern_id] = trend

    def _generate_forecast(self, history: list[dict]) -> dict:
        """Generate simple forecast"""
        if len(history) < 3:
            return {"predictions": "insufficient_data"}

        # Simple linear trend
        frequencies = [h["data"].get("frequency", 0) for h in history]
        if len(frequencies) > 1:
            trend = (frequencies[-1] - frequencies[0]) / len(frequencies)
            next_freq = frequencies[-1] + trend

            return {
                "next_frequency": next_freq,
                "trend": "up" if trend > 0 else "down" if trend < 0 else "stable",
                "confidence": min(1.0, len(history) / 10),
            }

        return {"predictions": "no_data"}

    def get_evolution_insights(self) -> dict:
        """Get evolution insights"""
        insights = {
            "total_patterns": len(self.evolution_data),
            "trending_up": [],
            "trending_down": [],
            "stable": [],
            "new_patterns": [],
        }

        for pattern_id, trend in self.trends.items():
            if trend.risk_trend == "increasing":
                insights["trending_up"].append(pattern_id)
            elif trend.risk_trend == "decreasing":
                insights["trending_down"].append(pattern_id)
            else:
                insights["stable"].append(pattern_id)

        # New patterns (last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        for pattern_id, history in self.evolution_data.items():
            if history and history[-1]["timestamp"] > cutoff:
                if len(history) == 1:
                    insights["new_patterns"].append(pattern_id)

        return insights


# ── UI Components ──────────────────────────────────────────────────────────


def render_pattern_recognition_ui(pattern_engine: PatternRecognitionEngine):
    """Render pattern recognition UI"""
    st.subheader("🔍 Plagiarism Pattern Recognition")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Detect Patterns", "📈 Pattern Analysis", "🔮 Predictions", "📉 Evolution"]
    )

    with tab1:
        render_detection_tab(pattern_engine)

    with tab2:
        render_pattern_analysis_tab(pattern_engine)

    with tab3:
        render_predictions_tab(pattern_engine)

    with tab4:
        render_evolution_tab(pattern_engine)


def render_detection_tab(pattern_engine: PatternRecognitionEngine):
    """Render pattern detection tab"""
    st.subheader("🔍 Detect Plagiarism Patterns")

    # Check for documents
    documents = st.session_state.get("raw_texts", {})
    similarity_matrix = st.session_state.get("similarity_matrix", None)

    if not documents:
        st.warning("No documents available. Upload documents first.")
        return

    if len(documents) < 2:
        st.warning("Need at least 2 documents for pattern detection.")
        return

    # Pattern detection options
    col1, col2 = st.columns(2)
    with col1:
        detect_copy_paste = st.checkbox("Detect Copy-Paste", value=True)
        detect_structural = st.checkbox("Detect Structural Patterns", value=True)

    with col2:
        detect_citation = st.checkbox("Detect Citation Patterns", value=True)
        detect_hybrid = st.checkbox("Detect Hybrid Patterns", value=True)

    if st.button("🔍 Detect Patterns", type="primary"):
        with st.spinner("Analyzing documents for patterns..."):
            patterns = pattern_engine.detect_patterns(documents, similarity_matrix)

            if patterns:
                st.success(f"✅ Detected {len(patterns)} patterns")

                # Display patterns
                for pattern in patterns:
                    with st.expander(f"📌 {pattern.name}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Type", pattern.pattern_type.title())
                            st.metric("Severity", pattern.severity.upper())
                        with col2:
                            st.metric("Confidence", f"{pattern.confidence:.1%}")
                            st.metric("Frequency", pattern.frequency)
                        with col3:
                            st.metric(
                                "First Detected",
                                pattern.first_detected.strftime("%Y-%m-%d"),
                            )
                            st.metric("Documents", len(pattern.documents))

                        st.markdown(f"**Description:** {pattern.description}")

                        if pattern.features:
                            st.markdown("**Features:**")
                            st.json(pattern.features)

                # Store patterns in session
                st.session_state["detected_patterns"] = patterns

            else:
                st.info("No plagiarism patterns detected.")

    # Show existing patterns
    if pattern_engine.patterns:
        st.subheader("📋 Existing Patterns")
        st.write(f"Total patterns detected: {len(pattern_engine.patterns)}")

        # Pattern summary
        pattern_types = Counter(
            p.pattern_type for p in pattern_engine.patterns.values()
        )
        type_df = pd.DataFrame(
            {
                "Pattern Type": list(pattern_types.keys()),
                "Count": list(pattern_types.values()),
            }
        )
        st.bar_chart(type_df.set_index("Pattern Type"))


def render_pattern_analysis_tab(pattern_engine: PatternRecognitionEngine):
    """Render pattern analysis tab"""
    st.subheader("📈 Pattern Analysis")

    if not pattern_engine.patterns:
        st.info("No patterns detected. Run pattern detection first.")
        return

    # Summary statistics
    patterns = list(pattern_engine.patterns.values())
    total = len(patterns)
    severity_counts = Counter(p.severity for p in patterns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patterns", total)
    col2.metric("Critical", severity_counts.get("critical", 0))
    col3.metric("High", severity_counts.get("high", 0))
    col4.metric("Medium", severity_counts.get("medium", 0))

    # Pattern distribution
    st.subheader("📊 Pattern Distribution")

    # By type
    type_counts = Counter(p.pattern_type for p in patterns)
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            )
        ]
    )
    fig.update_layout(
        title="Pattern Distribution by Type",
        xaxis_title="Pattern Type",
        yaxis_title="Count",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confidence distribution
    st.subheader("📊 Confidence Distribution")
    confidences = [p.confidence for p in patterns]
    fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
    fig.update_layout(
        title="Pattern Confidence Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pattern details
    st.subheader("📋 Pattern Details")
    pattern_data = []
    for p in patterns:
        pattern_data.append(
            {
                "Name": p.name,
                "Type": p.pattern_type,
                "Severity": p.severity,
                "Confidence": f"{p.confidence:.1%}",
                "Documents": len(p.documents),
                "First Detected": p.first_detected.strftime("%Y-%m-%d"),
            }
        )

    df = pd.DataFrame(pattern_data)
    st.dataframe(df, use_container_width=True)


def render_predictions_tab(pattern_engine: PatternRecognitionEngine):
    """Render predictions tab"""
    st.subheader("🔮 Risk Predictions")

    # Initialize prediction engine
    if "prediction_engine" not in st.session_state:
        st.session_state["prediction_engine"] = PredictionEngine(pattern_engine)

    prediction_engine = st.session_state["prediction_engine"]

    # Check if we have documents
    documents = st.session_state.get("raw_texts", {})
    if not documents:
        st.warning("No documents available for prediction.")
        return

    # Train model if enough data
    if len(pattern_engine.patterns) >= 5:
        training_data = []
        for pattern in pattern_engine.patterns.values():
            training_data.append(
                {
                    "similarity_score": pattern.confidence,
                    "document_length": len(" ".join(pattern.documents)),
                    "word_count": (
                        sum(len(d.split()) for d in pattern.documents)
                        / len(pattern.documents)
                        if pattern.documents
                        else 0
                    ),
                    "unique_words_ratio": 0.5,  # Placeholder
                    "complexity_score": 0.3,  # Placeholder
                    "pattern_count": len(pattern_engine.patterns),
                    "avg_similarity": pattern.confidence,
                    "max_similarity": pattern.confidence,
                    "risk_level": 1 if pattern.severity in ["high", "critical"] else 0,
                }
            )

        prediction_engine.train_model(training_data)

    # Predict for each document
    if st.button("🔮 Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            predictions = []
            similarity_matrix = st.session_state.get("similarity_matrix", None)

            for doc_name, content in documents.items():
                # Get similarity scores
                if similarity_matrix is not None:
                    doc_index = list(documents.keys()).index(doc_name)
                    if doc_index < len(similarity_matrix):
                        scores = similarity_matrix[doc_index]
                    else:
                        scores = []
                else:
                    scores = []

                prediction = prediction_engine.predict_risk(doc_name, content, scores)
                predictions.append(prediction)

            if predictions:
                st.success(f"✅ Generated predictions for {len(predictions)} documents")

                # Display predictions
                for pred in predictions:
                    color_map = {
                        "critical": "#D32F2F",
                        "high": "#F44336",
                        "medium": "#FF9800",
                        "low": "#4CAF50",
                    }
                    color = color_map.get(pred.risk_level, "#666")

                    st.markdown(
                        f"""
                    <div style='background:{color}20;padding:15px;border-radius:8px;border-left:4px solid {color};margin:10px 0;'>
                        <h4 style='margin:0;'>{pred.document_name}</h4>
                        <p style='margin:5px 0;'>
                            <strong>Risk Level:</strong> <span style='color:{color};font-weight:bold;'>{pred.risk_level.upper()}</span>
                            | <strong>Score:</strong> {pred.predicted_risk_score:.1%}
                            | <strong>Confidence:</strong> {pred.confidence:.1%}
                        </p>
                        <p style='margin:5px 0;font-size:14px;'><strong>Factors:</strong> {', '.join(pred.contributing_factors)}</p>
                        <p style='margin:5px 0;font-size:14px;'><strong>Recommendations:</strong></p>
                        <ul style='margin:5px 0;'>
                            {''.join(f'<li>{rec}</li>' for rec in pred.recommendations)}
                        </ul>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    # Show existing predictions
    if pattern_engine.predictions:
        st.subheader("📋 Recent Predictions")
        for pred in list(pattern_engine.predictions.values())[-5:]:
            st.markdown(
                f"- **{pred.document_name}**: {pred.risk_level.upper()} ({pred.predicted_risk_score:.1%})"
            )


def render_evolution_tab(pattern_engine: PatternRecognitionEngine):
    """Render evolution tab"""
    st.subheader("📉 Pattern Evolution")

    if not pattern_engine.patterns:
        st.info("No patterns to track evolution.")
        return

    # Initialize tracker
    if "evolution_tracker" not in st.session_state:
        st.session_state["evolution_tracker"] = PatternEvolutionTracker(pattern_engine)

    tracker = st.session_state["evolution_tracker"]

    # Get evolution insights
    insights = tracker.get_evolution_insights()

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patterns", insights["total_patterns"])
    col2.metric("Trending Up", len(insights["trending_up"]))
    col3.metric("New Patterns (7 days)", len(insights["new_patterns"]))

    # Evolution analysis
    if pattern_engine.patterns:
        # Select pattern to analyze
        pattern_options = {p.id: p.name for p in pattern_engine.patterns.values()}
        selected_pattern = st.selectbox(
            "Select Pattern",
            options=list(pattern_options.keys()),
            format_func=lambda x: pattern_options.get(x, x),
        )

        if selected_pattern and selected_pattern in tracker.evolution_data:
            history = tracker.evolution_data[selected_pattern]

            if history:
                # Plot evolution
                timestamps = [h["timestamp"] for h in history]
                frequencies = [h["data"].get("frequency", 0) for h in history]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=frequencies,
                        mode="lines+markers",
                        name="Frequency",
                        line=dict(color="#2196F3", width=2),
                        marker=dict(size=8),
                    )
                )
                fig.update_layout(
                    title="Pattern Evolution Over Time",
                    xaxis_title="Date",
                    yaxis_title="Frequency",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Pattern trends summary
        st.subheader("📊 Pattern Trends")
        if tracker.trends:
            trend_data = []
            for trend in tracker.trends.values():
                trend_data.append(
                    {
                        "Pattern": trend.pattern_name,
                        "Trend": trend.risk_trend.title(),
                        "Change": f"{trend.frequency_change:.1%}",
                        "New Documents": trend.new_documents,
                        "Time Period": trend.time_period,
                    }
                )

            df = pd.DataFrame(trend_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No trend data available. Track patterns over time.")


# ── Integration with Main App ─────────────────────────────────────────────


def integrate_pattern_recognition():
    """Initialize and integrate pattern recognition"""
    if "pattern_engine" not in st.session_state:
        st.session_state["pattern_engine"] = PatternRecognitionEngine()

    # Add pattern recognition tab to main app
    render_pattern_recognition_ui(st.session_state["pattern_engine"])


# ── End of Pattern Recognition System ──────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
