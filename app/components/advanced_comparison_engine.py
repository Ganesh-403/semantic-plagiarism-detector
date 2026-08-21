# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: ADVANCED DOCUMENT COMPARISON ENGINE (Issue #1999) ──────────────
# ───────────────────────────────────────────────────────────────────────────────

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class ComparisonMetric:
    """Represents a single comparison metric"""

    name: str
    score: float
    weight: float
    details: Dict[str, Any] = None
    threshold: float = 0.5

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def is_passed(self) -> bool:
        return self.score >= self.threshold

    def to_dict(self) -> Dict:
        return {**asdict(self), "passed": self.is_passed()}


@dataclass
class DocumentComparisonResult:
    """Complete document comparison result"""

    doc_a: str
    doc_b: str
    timestamp: str
    overall_score: float
    metrics: List[ComparisonMetric]
    highlights: Dict[str, List[Tuple[int, int]]]
    recommendations: List[str]
    risk_level: str  # 'low', 'medium', 'high', 'critical'

    def to_dict(self) -> Dict:
        return {
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "metrics": [m.to_dict() for m in self.metrics],
            "highlights": self.highlights,
            "recommendations": self.recommendations,
            "risk_level": self.risk_level,
        }


# ── Advanced Comparison Engine ─────────────────────────────────────────────


class AdvancedComparisonEngine:
    """Advanced document comparison with multiple metrics"""

    def __init__(self):
        self.metric_weights = {
            "lexical": 0.25,
            "semantic": 0.30,
            "structural": 0.15,
            "stylistic": 0.10,
            "citation": 0.10,
            "paraphrase": 0.10,
        }
        self.thresholds = {
            "lexical": 0.40,
            "semantic": 0.50,
            "structural": 0.30,
            "stylistic": 0.20,
            "citation": 0.40,
            "paraphrase": 0.35,
        }
        self.comparison_history = []

    def compare_documents(
        self,
        text_a: str,
        text_b: str,
        doc_a_name: str = "Document A",
        doc_b_name: str = "Document B",
    ) -> DocumentComparisonResult:
        """Compare two documents using multiple metrics"""
        metrics = []
        highlights = {}
        recommendations = []

        # 1. Lexical Similarity
        lexical_score = self._calculate_lexical_similarity(text_a, text_b)
        lexical_metric = ComparisonMetric(
            name="Lexical Similarity",
            score=lexical_score,
            weight=self.metric_weights["lexical"],
            threshold=self.thresholds["lexical"],
            details={"method": "Jaccard + TF-IDF"},
        )
        metrics.append(lexical_metric)

        # 2. Semantic Similarity
        semantic_score = self._calculate_semantic_similarity(text_a, text_b)
        semantic_metric = ComparisonMetric(
            name="Semantic Similarity",
            score=semantic_score,
            weight=self.metric_weights["semantic"],
            threshold=self.thresholds["semantic"],
            details={"method": "Sentence Embeddings"},
        )
        metrics.append(semantic_metric)

        # 3. Structural Similarity
        structural_score = self._calculate_structural_similarity(text_a, text_b)
        structural_metric = ComparisonMetric(
            name="Structural Similarity",
            score=structural_score,
            weight=self.metric_weights["structural"],
            threshold=self.thresholds["structural"],
            details={"method": "Paragraph Structure Analysis"},
        )
        metrics.append(structural_metric)

        # 4. Stylistic Similarity
        stylistic_score = self._calculate_stylistic_similarity(text_a, text_b)
        stylistic_metric = ComparisonMetric(
            name="Stylistic Similarity",
            score=stylistic_score,
            weight=self.metric_weights["stylistic"],
            threshold=self.thresholds["stylistic"],
            details={"method": "Writing Style Analysis"},
        )
        metrics.append(stylistic_metric)

        # 5. Citation Similarity
        citation_score = self._calculate_citation_similarity(text_a, text_b)
        citation_metric = ComparisonMetric(
            name="Citation Similarity",
            score=citation_score,
            weight=self.metric_weights["citation"],
            threshold=self.thresholds["citation"],
            details={"method": "Citation Analysis"},
        )
        metrics.append(citation_metric)

        # 6. Paraphrase Detection
        paraphrase_score = self._detect_paraphrasing(text_a, text_b)
        paraphrase_metric = ComparisonMetric(
            name="Paraphrase Detection",
            score=paraphrase_score,
            weight=self.metric_weights["paraphrase"],
            threshold=self.thresholds["paraphrase"],
            details={"method": "N-gram + Semantic Alignment"},
        )
        metrics.append(paraphrase_metric)

        # Calculate overall score
        overall_score = sum(m.score * m.weight for m in metrics)

        # Generate highlights
        highlights = self._generate_highlights(text_a, text_b)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        # Determine risk level
        risk_level = self._determine_risk_level(metrics, overall_score)

        # Create result
        result = DocumentComparisonResult(
            doc_a=doc_a_name,
            doc_b=doc_b_name,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            metrics=metrics,
            highlights=highlights,
            recommendations=recommendations,
            risk_level=risk_level,
        )

        # Store in history
        self.comparison_history.append(result)

        return result

    def _calculate_lexical_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate lexical similarity using Jaccard and TF-IDF"""
        # Tokenize
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        # Jaccard similarity
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        jaccard = (
            len(set_a.intersection(set_b)) / len(set_a.union(set_b))
            if set_a.union(set_b)
            else 0.0
        )

        # TF-IDF similarity
        all_tokens = list(set_a.union(set_b))
        tf_a = [tokens_a.count(t) for t in all_tokens]
        tf_b = [tokens_b.count(t) for t in all_tokens]

        # Simple cosine similarity with TF
        dot = sum(a * b for a, b in zip(tf_a, tf_b))
        norm_a = math.sqrt(sum(a * a for a in tf_a))
        norm_b = math.sqrt(sum(b * b for b in tf_b))
        tfidf = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        # Combine scores
        return 0.6 * jaccard + 0.4 * tfidf

    def _calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate semantic similarity using basic embedding approach"""
        # Simple word frequency vector approach
        words_a = self._extract_keywords(text_a)
        words_b = self._extract_keywords(text_b)

        if not words_a or not words_b:
            return 0.0

        # Create feature vectors
        all_words = list(set(words_a.keys()).union(set(words_b.keys())))
        vec_a = [words_a.get(w, 0) for w in all_words]
        vec_b = [words_b.get(w, 0) for w in all_words]

        # Cosine similarity
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def _calculate_structural_similarity(self, text_a: str, text_b: str) -> float:
        """Compare document structure (paragraphs, sections)"""
        # Split into paragraphs
        paras_a = [p for p in text_a.split("\n\n") if p.strip()]
        paras_b = [p for p in text_b.split("\n\n") if p.strip()]

        if not paras_a or not paras_b:
            return 0.0

        # Compare paragraph counts
        count_similarity = 1 - abs(len(paras_a) - len(paras_b)) / max(
            len(paras_a), len(paras_b)
        )

        # Compare paragraph lengths
        lengths_a = [len(p.split()) for p in paras_a]
        lengths_b = [len(p.split()) for p in paras_b]

        if lengths_a and lengths_b:
            avg_a = sum(lengths_a) / len(lengths_a)
            avg_b = sum(lengths_b) / len(lengths_b)
            length_similarity = (
                1 - abs(avg_a - avg_b) / max(avg_a, avg_b)
                if max(avg_a, avg_b) > 0
                else 0
            )
        else:
            length_similarity = 0.0

        return 0.6 * count_similarity + 0.4 * length_similarity

    def _calculate_stylistic_similarity(self, text_a: str, text_b: str) -> float:
        """Analyze writing style similarity"""
        # Extract style features
        features_a = self._extract_style_features(text_a)
        features_b = self._extract_style_features(text_b)

        if not features_a or not features_b:
            return 0.0

        # Compare features
        similarities = []
        for key in features_a.keys():
            if key in features_b:
                if key in [
                    "avg_word_length",
                    "avg_sentence_length",
                    "avg_paragraph_length",
                ]:
                    diff = abs(features_a[key] - features_b[key])
                    max_val = max(features_a[key], features_b[key])
                    sim = 1 - (diff / max_val) if max_val > 0 else 0
                else:
                    sim = 1 if features_a[key] == features_b[key] else 0
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_citation_similarity(self, text_a: str, text_b: str) -> float:
        """Compare citation patterns"""
        citations_a = self._extract_citations(text_a)
        citations_b = self._extract_citations(text_b)

        if not citations_a or not citations_b:
            return 0.0

        # Jaccard similarity on citations
        set_a = set(citations_a)
        set_b = set(citations_b)

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        return intersection / union if union > 0 else 0.0

    def _detect_paraphrasing(self, text_a: str, text_b: str) -> float:
        """Detect paraphrasing using n-gram and semantic analysis"""
        # Extract n-grams
        ngrams_a = self._extract_ngrams(text_a, n=3)
        ngrams_b = self._extract_ngrams(text_b, n=3)

        # Find similar n-grams
        similar = 0
        total = min(len(ngrams_a), len(ngrams_b))

        for ngram_a in ngrams_a[: min(len(ngrams_a), 100)]:
            for ngram_b in ngrams_b[: min(len(ngrams_b), 100)]:
                if self._ngram_similarity(ngram_a, ngram_b) > 0.7:
                    similar += 1
                    break

        return similar / total if total > 0 else 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return [w for w in words if len(w) > 2]

    def _extract_keywords(self, text: str) -> Dict[str, float]:
        """Extract keywords with TF scores"""
        words = self._tokenize(text)
        total = len(words) or 1
        freq = Counter(words)
        return {word: count / total for word, count in freq.items()}

    def _extract_style_features(self, text: str) -> Dict[str, Any]:
        """Extract writing style features"""
        sentences = [s for s in text.split(".") if s.strip()]
        words = self._tokenize(text)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        if not sentences or not words:
            return {}

        features = {
            "avg_sentence_length": sum(len(s.split()) for s in sentences)
            / len(sentences),
            "avg_word_length": sum(len(w) for w in words) / len(words),
            "avg_paragraph_length": sum(len(p.split()) for p in paragraphs)
            / len(paragraphs)
            if paragraphs
            else 0,
            "unique_words_ratio": len(set(words)) / len(words) if words else 0,
            "complexity_score": len([w for w in words if len(w) > 6]) / len(words)
            if words
            else 0,
        }

        return features

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text"""
        # Simple citation extraction (author-year format)
        pattern = r"\(([A-Z][a-z]+,\s*\d{4})\)"
        citations = re.findall(pattern, text)

        # Also check for numbered citations
        pattern2 = r"\[(\d+)\]"
        citations.extend(re.findall(pattern2, text))

        return citations

    def _extract_ngrams(self, text: str, n: int = 3) -> List[Tuple[str, ...]]:
        """Extract n-grams from text"""
        words = self._tokenize(text)
        return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    def _ngram_similarity(
        self, ngram_a: Tuple[str, ...], ngram_b: Tuple[str, ...]
    ) -> float:
        """Calculate similarity between two n-grams"""
        if not ngram_a or not ngram_b:
            return 0.0

        common = len(set(ngram_a).intersection(set(ngram_b)))
        total = len(set(ngram_a).union(set(ngram_b)))

        return common / total if total > 0 else 0.0

    def _generate_highlights(
        self, text_a: str, text_b: str
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Generate highlight positions for similar content"""
        highlights = {"a": [], "b": []}

        # Simple highlighting based on common phrases
        lines_a = text_a.split("\n")
        lines_b = text_b.split("\n")

        for i, line_a in enumerate(lines_a):
            if line_a.strip():
                for j, line_b in enumerate(lines_b):
                    if line_b.strip():
                        # Check if lines are similar
                        ratio = SequenceMatcher(None, line_a, line_b).ratio()
                        if ratio > 0.8:
                            highlights["a"].append((i, j))
                            highlights["b"].append((j, i))

        return highlights

    def _generate_recommendations(self, metrics: List[ComparisonMetric]) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []

        for metric in metrics:
            if metric.score >= 0.8:
                if metric.name == "Lexical Similarity":
                    recommendations.append(
                        "High lexical similarity detected. Review for direct copying."
                    )
                elif metric.name == "Semantic Similarity":
                    recommendations.append(
                        "High semantic similarity detected. Check for paraphrasing."
                    )
                elif metric.name == "Structural Similarity":
                    recommendations.append(
                        "Document structure is very similar. Verify source."
                    )
                elif metric.name == "Stylistic Similarity":
                    recommendations.append(
                        "Writing style is highly similar. Check if same author."
                    )
                elif metric.name == "Citation Similarity":
                    recommendations.append(
                        "Citation patterns match closely. Verify references."
                    )
                elif metric.name == "Paraphrase Detection":
                    recommendations.append(
                        "Significant paraphrasing detected. Deep review needed."
                    )

        if not recommendations:
            recommendations.append(
                "No significant issues detected. Continue monitoring."
            )

        return recommendations

    def _determine_risk_level(
        self, metrics: List[ComparisonMetric], overall_score: float
    ) -> str:
        """Determine risk level based on metrics"""
        high_scores = sum(1 for m in metrics if m.score > 0.7)
        total_metrics = len(metrics)

        if overall_score > 0.8 or high_scores >= 4:
            return "critical"
        elif overall_score > 0.6 or high_scores >= 3:
            return "high"
        elif overall_score > 0.4 or high_scores >= 2:
            return "medium"
        else:
            return "low"

    def get_comparison_history(self, limit: int = 50) -> List[Dict]:
        """Get comparison history"""
        return [r.to_dict() for r in self.comparison_history[-limit:]]

    def get_comparison_stats(self) -> Dict:
        """Get comparison statistics"""
        if not self.comparison_history:
            return {"total": 0}

        results = [r.to_dict() for r in self.comparison_history]

        return {
            "total": len(results),
            "avg_score": sum(r["overall_score"] for r in results) / len(results),
            "risk_distribution": {
                "critical": len([r for r in results if r["risk_level"] == "critical"]),
                "high": len([r for r in results if r["risk_level"] == "high"]),
                "medium": len([r for r in results if r["risk_level"] == "medium"]),
                "low": len([r for r in results if r["risk_level"] == "low"]),
            },
            "recent": results[-5:],
        }


# ── UI Components ──────────────────────────────────────────────────────────


def render_comparison_engine_ui(comparison_engine: AdvancedComparisonEngine):
    """Render comparison engine UI"""
    st.subheader("🔬 Advanced Document Comparison")

    # Document selection
    documents = st.session_state.get("document_names", [])

    if len(documents) < 2:
        st.warning("Need at least 2 documents to compare.")
        return

    col1, col2 = st.columns(2)
    with col1:
        doc_a = st.selectbox("Document A:", documents, key="comp_doc_a")
    with col2:
        doc_b = st.selectbox(
            "Document B:", [d for d in documents if d != doc_a], key="comp_doc_b"
        )

    # Get content
    raw_texts = st.session_state.get("raw_texts", {})
    text_a = raw_texts.get(doc_a, "")
    text_b = raw_texts.get(doc_b, "")

    if st.button("🔍 Compare Documents", type="primary", key="compare_btn"):
        with st.spinner("Analyzing documents..."):
            result = comparison_engine.compare_documents(text_a, text_b, doc_a, doc_b)
            st.session_state["last_comparison"] = result

            # Display results
            display_comparison_results(result)

    # Display previous results
    if "last_comparison" in st.session_state:
        st.divider()
        st.subheader("📋 Previous Comparison")
        display_comparison_results(st.session_state["last_comparison"])

    # Comparison history
    if comparison_engine.comparison_history:
        st.divider()
        with st.expander("📊 Comparison History", expanded=False):
            stats = comparison_engine.get_comparison_stats()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Comparisons", stats["total"])
            col2.metric("Avg Score", f"{stats['avg_score'] * 100:.1f}%")

            risk_counts = stats["risk_distribution"]
            col3.metric("High Risk", risk_counts["critical"] + risk_counts["high"])

            # History table
            history_df = pd.DataFrame(stats["recent"])
            if not history_df.empty:
                st.dataframe(
                    history_df[["doc_a", "doc_b", "overall_score", "risk_level"]],
                    use_container_width=True,
                )


def display_comparison_results(result: DocumentComparisonResult):
    """Display comparison results"""
    # Overall metrics
    risk_colors = {
        "low": "#4CAF50",
        "medium": "#FF9800",
        "high": "#F44336",
        "critical": "#D32F2F",
    }
    color = risk_colors.get(result.risk_level, "#666")

    st.markdown(
        f"""
    <div style='background:{color}20;padding:20px;border-radius:10px;border-left:4px solid {color};'>
        <h3 style='margin:0;color:{color};'>Risk Level: {result.risk_level.upper()}</h3>
        <p style='margin:5px 0;font-size:24px;font-weight:bold;'>{result.overall_score * 100:.1f}%</p>
        <p style='margin:0;font-size:14px;'>Overall Similarity Score</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Metrics breakdown
    st.subheader("📊 Metrics Breakdown")

    metrics_data = []
    for metric in result.metrics:
        metrics_data.append(
            {
                "Metric": metric.name,
                "Score": metric.score,
                "Weight": metric.weight,
                "Threshold": metric.threshold,
                "Status": "✅ Passed" if metric.is_passed() else "⚠️ Needs Review",
            }
        )

    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)

    # Metric visualization
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[m.name for m in result.metrics],
            y=[m.score for m in result.metrics],
            name="Score",
            marker_color=[
                "#4CAF50" if m.is_passed() else "#FF9800" for m in result.metrics
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[m.name for m in result.metrics],
            y=[m.threshold for m in result.metrics],
            name="Threshold",
            mode="lines+markers",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        title="Metric Scores vs Thresholds",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.subheader("💡 Recommendations")
    for rec in result.recommendations:
        st.markdown(f"- {rec}")

    # Highlights
    if result.highlights:
        with st.expander("📝 Document Highlights", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{result.doc_a}**")
                st.markdown(f"Similar sections: {len(result.highlights.get('a', []))}")
            with col2:
                st.markdown(f"**{result.doc_b}**")
                st.markdown(f"Similar sections: {len(result.highlights.get('b', []))}")


# ── Integration with Main App ─────────────────────────────────────────────


def integrate_comparison_engine():
    """Initialize and integrate comparison engine"""
    if "comparison_engine" not in st.session_state:
        st.session_state["comparison_engine"] = AdvancedComparisonEngine()

    # Add comparison tab to main app
    render_comparison_engine_ui(st.session_state["comparison_engine"])


# ── End of Comparison Engine ──────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
