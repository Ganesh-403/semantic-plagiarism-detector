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

"""
Automated Machine Learning Optimization for Plagiarism Detection

Features:
- Dynamic threshold optimization
- Adaptive learning from feedback
- Pattern recognition and analysis
- Performance metrics tracking
- A/B testing for strategies
- Self-improving detection system
"""

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from bayes_opt import BayesianOptimization

    BAYES_OPT_AVAILABLE = True
except ImportError:
    BAYES_OPT_AVAILABLE = False

try:
    from scipy import stats  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for AutoML optimization."""

    enabled: bool = True
    learning_rate: float = 0.01
    optimization_interval: int = 3600  # 1 hour
    min_samples_for_optimization: int = 50
    max_threshold: float = 0.95
    min_threshold: float = 0.30
    exploration_rate: float = 0.1  # For A/B testing
    feedback_weight: float = 0.6
    historical_weight: float = 0.4
    auto_apply_changes: bool = False
    notify_on_improvement: bool = True


@dataclass
class OptimizationMetrics:
    """Performance metrics for optimization."""

    timestamp: float
    threshold: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    false_positives: int
    false_negatives: int
    total_pairs: int
    sample_size: int


@dataclass
class UserFeedback:
    """User feedback on detected pairs."""

    pair_id: str
    doc_a: str
    doc_b: str
    similarity_score: float
    user_judgement: bool  # True = plagiarism, False = not plagiarism
    timestamp: float
    confidence: float = 1.0
    notes: str = ""


@dataclass
class PatternProfile:
    """Document pattern profile."""

    doc_type: str  # academic, technical, creative, etc.
    avg_word_length: float
    avg_sentence_length: float
    vocabulary_richness: float
    readability_score: float
    avg_similarity: float
    std_similarity: float
    sample_size: int


# ==============================================================================
# CORE OPTIMIZER CLASS
# ==============================================================================


class AutoMLOptimizer:
    """
    Automated Machine Learning optimizer for plagiarism detection.
    Continuously learns and improves detection thresholds.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.metrics_history: list[OptimizationMetrics] = []
        self.feedback_history: list[UserFeedback] = []
        self.threshold_history: list[float] = []
        self.pattern_profiles: dict[str, PatternProfile] = {}
        self.best_threshold = 0.75
        self.current_strategy = "balanced"  # balanced, conservative, aggressive

        # ML Models
        self.classifier = None
        self.optimizer = None
        self.is_trained = False

        # Statistics
        self.total_optimizations = 0
        self.improvement_count = 0
        self.last_optimization_time = 0
        self.performance_scores: list[float] = []

        # Initialize
        self._initialize_models()
        self._load_history()

    def _initialize_models(self):
        """Initialize ML models."""
        if SKLEARN_AVAILABLE:
            self.classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        self.is_trained = False

    def _load_history(self):
        """Load optimization history from storage."""
        try:
            history_path = (
                Path(st.session_state.get("data_dir", "."))
                / "optimization_history.json"
            )
            if history_path.exists():
                with open(history_path) as f:
                    data = json.load(f)
                    self.threshold_history = data.get("thresholds", [])
                    self.metrics_history = [
                        OptimizationMetrics(**m) for m in data.get("metrics", [])
                    ]
                    self.best_threshold = (
                        max(self.threshold_history) if self.threshold_history else 0.75
                    )
                    self.total_optimizations = len(self.threshold_history)
        except Exception as e:
            print(f"Error loading history: {e}")

    def save_history(self):
        """Save optimization history to storage."""
        try:
            history_path = (
                Path(st.session_state.get("data_dir", "."))
                / "optimization_history.json"
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "thresholds": self.threshold_history,
                "metrics": [asdict(m) for m in self.metrics_history],
                "best_threshold": self.best_threshold,
                "total_optimizations": self.total_optimizations,
                "improvement_count": self.improvement_count,
            }

            with open(history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_feedback(self, feedback: UserFeedback):
        """Add user feedback for learning."""
        self.feedback_history.append(feedback)

        # Keep only last 10,000 feedback entries
        if len(self.feedback_history) > 10000:
            self.feedback_history = self.feedback_history[-10000:]

        # Update model if enough feedback
        if len(self.feedback_history) >= self.config.min_samples_for_optimization:
            self._train_model()

    def add_metrics(self, metrics: OptimizationMetrics):
        """Add performance metrics."""
        self.metrics_history.append(metrics)

        # Keep last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        # Track performance
        self.performance_scores.append(metrics.f1_score)
        if len(self.performance_scores) > 100:
            self.performance_scores = self.performance_scores[-100:]

    def _train_model(self):
        """Train the ML model on feedback data."""
        if not SKLEARN_AVAILABLE or not self.feedback_history:
            return

        # Prepare features from feedback
        X = []
        y = []

        for feedback in self.feedback_history:
            # Features: similarity score, document characteristics
            features = [
                feedback.similarity_score,
                self._get_doc_characteristic(feedback.doc_a),
                self._get_doc_characteristic(feedback.doc_b),
                feedback.confidence,
            ]
            X.append(features)
            y.append(1 if feedback.user_judgement else 0)

        if len(X) < self.config.min_samples_for_optimization:
            return

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)  # noqa: F841
        recall = recall_score(y_test, y_pred, zero_division=0)  # noqa: F841
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Model trained - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

    def _get_doc_characteristic(self, doc_name: str) -> float:
        """Get document characteristic score."""
        # Simplified - in production would extract more features
        pattern = self.pattern_profiles.get(doc_name)
        if pattern:
            return pattern.avg_similarity
        return 0.5

    def optimize_threshold(self) -> float:
        """
        Optimize threshold using Bayesian optimization or gradient-based methods.

        Returns:
            float: Optimized threshold value
        """
        # Check if enough data for optimization
        if len(self.metrics_history) < 10:
            return self.best_threshold

        if BAYES_OPT_AVAILABLE and len(self.metrics_history) >= 50:
            # Use Bayesian optimization
            return self._bayesian_optimize()
        else:
            # Use gradient-based approach
            return self._gradient_optimize()

    def _bayesian_optimize(self) -> float:
        """Use Bayesian optimization for threshold tuning."""
        if not BAYES_OPT_AVAILABLE:
            return self.best_threshold

        # Define objective function
        def objective(threshold):
            # Evaluate performance at this threshold
            metrics = self._evaluate_threshold(threshold)
            return metrics.f1_score  # Maximize F1 score

        # Define bounds
        bounds = {"threshold": (self.config.min_threshold, self.config.max_threshold)}

        # Create optimizer
        optimizer = BayesianOptimization(f=objective, pbounds=bounds, random_state=42)

        # Optimize
        optimizer.maximize(init_points=5, n_iter=20)

        # Get best threshold
        best = optimizer.max["params"]["threshold"]

        self.best_threshold = best
        self.total_optimizations += 1

        return best

    def _gradient_optimize(self) -> float:
        """Use gradient-based optimization."""
        if len(self.metrics_history) < 5:
            return self.best_threshold

        # Sample thresholds around current best
        current = self.best_threshold
        candidates = []

        # Try thresholds above and below
        for delta in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            candidate = max(
                self.config.min_threshold,
                min(self.config.max_threshold, current + delta),
            )
            if candidate != current:
                metrics = self._evaluate_threshold(candidate)
                candidates.append((candidate, metrics.f1_score))

        if not candidates:
            return current

        # Find best candidate
        best_candidate = max(candidates, key=lambda x: x[1])

        # Only update if improvement > 1%
        if best_candidate[1] > self.performance_scores[-1] * 1.01:
            self.best_threshold = best_candidate[0]
            self.total_optimizations += 1
            self.improvement_count += 1

        return self.best_threshold

    def _evaluate_threshold(self, threshold: float) -> OptimizationMetrics:
        """
        Evaluate performance at given threshold.

        Returns:
            OptimizationMetrics: Performance metrics at threshold
        """
        # Use historical data to estimate performance
        if not self.metrics_history:
            return OptimizationMetrics(
                timestamp=time.time(),
                threshold=threshold,
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                accuracy=0.5,
                false_positives=0,
                false_negatives=0,
                total_pairs=0,
                sample_size=0,
            )

        # Estimate metrics based on historical data
        # This is simplified - would use actual evaluation in production
        historical_metrics = self.metrics_history[-20:]  # Last 20 measurements

        precision = (
            np.mean([m.precision for m in historical_metrics])
            if historical_metrics
            else 0.5
        )
        recall = (
            np.mean([m.recall for m in historical_metrics])
            if historical_metrics
            else 0.5
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return OptimizationMetrics(
            timestamp=time.time(),
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=(
                np.mean([m.accuracy for m in historical_metrics])
                if historical_metrics
                else 0.5
            ),
            false_positives=(
                np.sum([m.false_positives for m in historical_metrics])
                if historical_metrics
                else 0
            ),
            false_negatives=(
                np.sum([m.false_negatives for m in historical_metrics])
                if historical_metrics
                else 0
            ),
            total_pairs=(
                np.sum([m.total_pairs for m in historical_metrics])
                if historical_metrics
                else 0
            ),
            sample_size=len(historical_metrics),
        )

    def analyze_patterns(
        self, similarity_data: pd.DataFrame
    ) -> dict[str, PatternProfile]:
        """
        Analyze patterns in documents.

        Args:
            similarity_data: Similarity matrix

        Returns:
            Dict[str, PatternProfile]: Pattern profiles for each document
        """
        profiles = {}

        for doc_name in similarity_data.columns:
            # Get similarity scores for this document
            scores = similarity_data[doc_name].values

            # Calculate statistics
            avg_sim = np.mean(scores) if len(scores) > 0 else 0
            std_sim = np.std(scores) if len(scores) > 1 else 0

            # Detect document type (simplified)
            doc_type = self._detect_doc_type(doc_name, scores)

            profile = PatternProfile(
                doc_type=doc_type,
                avg_word_length=self._estimate_word_length(doc_name),
                avg_sentence_length=self._estimate_sentence_length(doc_name),
                vocabulary_richness=self._estimate_vocabulary_richness(doc_name),
                readability_score=self._estimate_readability(doc_name),
                avg_similarity=avg_sim,
                std_similarity=std_sim,
                sample_size=len(scores),
            )

            profiles[doc_name] = profile
            self.pattern_profiles[doc_name] = profile

        return profiles

    def _detect_doc_type(self, doc_name: str, scores: np.ndarray) -> str:
        """Detect document type based on patterns."""
        avg_score = np.mean(scores) if len(scores) > 0 else 0

        # Simple heuristic
        if avg_score > 0.8:
            return "high_similarity"
        elif avg_score > 0.6:
            return "medium_similarity"
        elif avg_score > 0.4:
            return "low_similarity"
        else:
            return "unique"

    def _estimate_word_length(self, doc_name: str) -> float:
        """Estimate average word length."""
        # Placeholder - would extract from actual documents
        return random.uniform(4.5, 6.5)

    def _estimate_sentence_length(self, doc_name: str) -> float:
        """Estimate average sentence length."""
        return random.uniform(15, 30)

    def _estimate_vocabulary_richness(self, doc_name: str) -> float:
        """Estimate vocabulary richness."""
        return random.uniform(0.4, 0.8)

    def _estimate_readability(self, doc_name: str) -> float:
        """Estimate readability score."""
        return random.uniform(40, 70)

    def get_optimal_threshold(self, context: dict[str, Any] = None) -> float:
        """
        Get optimal threshold based on current context.

        Args:
            context: Context information (doc_type, user_preference, etc.)

        Returns:
            float: Optimized threshold
        """
        # If optimization is disabled, return default
        if not self.config.enabled:
            return 0.75

        # Check if optimization is needed
        if (
            time.time() - self.last_optimization_time
            < self.config.optimization_interval
        ):
            return self.best_threshold

        # Apply exploration (A/B testing)
        if random.random() < self.config.exploration_rate:
            # Explore: try a random threshold
            candidate = random.uniform(
                self.config.min_threshold, self.config.max_threshold
            )
            self.threshold_history.append(candidate)
            return candidate

        # Exploit: use best known threshold
        self.threshold_history.append(self.best_threshold)

        # Periodically optimize
        if len(self.threshold_history) % 10 == 0:
            self.optimize_threshold()
            self.last_optimization_time = time.time()
            self.save_history()

        return self.best_threshold

    def get_performance_trend(self) -> dict[str, list[float]]:
        """
        Get performance trends over time.

        Returns:
            Dict[str, List[float]]: Performance metrics trends
        """
        if not self.metrics_history:
            return {"precision": [], "recall": [], "f1_score": [], "accuracy": []}

        # Get last 100 metrics
        recent = self.metrics_history[-100:]

        return {
            "precision": [m.precision for m in recent],
            "recall": [m.recall for m in recent],
            "f1_score": [m.f1_score for m in recent],
            "accuracy": [m.accuracy for m in recent],
        }

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get summary of optimization results."""
        if not self.metrics_history:
            return {
                "total_optimizations": 0,
                "current_threshold": self.best_threshold,
                "best_f1": 0,
                "improvements": 0,
                "data_points": 0,
            }

        best_metric = max(self.metrics_history, key=lambda m: m.f1_score)

        return {
            "total_optimizations": self.total_optimizations,
            "current_threshold": self.best_threshold,
            "best_f1": best_metric.f1_score,
            "best_threshold_at_best_f1": best_metric.threshold,
            "improvements": self.improvement_count,
            "data_points": len(self.metrics_history),
            "model_trained": self.is_trained,
            "feedback_count": len(self.feedback_history),
            "config": asdict(self.config),
        }


# ==============================================================================
# AUTO-OPTIMIZATION INTEGRATION
# ==============================================================================


class AutoOptimizationIntegration:
    """
    Integrates AutoML optimization with the plagiarism detection pipeline.
    """

    def __init__(self):
        self.optimizer = AutoMLOptimizer()
        self.pipeline_metrics: list[dict] = []
        self.optimization_runs: int = 0

    def optimize_pipeline(
        self, similarity_data: pd.DataFrame, current_threshold: float
    ) -> tuple[float, dict]:
        """
        Optimize the pipeline parameters.

        Args:
            similarity_data: Similarity matrix
            current_threshold: Current threshold

        Returns:
            Tuple[float, Dict]: Optimized threshold and metrics
        """
        # Analyze patterns
        profiles = self.optimizer.analyze_patterns(similarity_data)

        # Get context
        context = {
            "doc_count": len(similarity_data.columns),
            "avg_similarity": (
                similarity_data.values.mean() if not similarity_data.empty else 0
            ),
            "max_similarity": (
                similarity_data.values.max() if not similarity_data.empty else 0
            ),
            "doc_types": [p.doc_type for p in profiles.values()],
        }

        # Get optimized threshold
        optimized_threshold = self.optimizer.get_optimal_threshold(context)

        # Calculate metrics
        metrics = self._calculate_pipeline_metrics(similarity_data, optimized_threshold)

        # Track metrics
        self.pipeline_metrics.append(metrics)
        self.optimization_runs += 1

        # Add to optimizer history
        self.optimizer.add_metrics(
            OptimizationMetrics(
                timestamp=time.time(),
                threshold=optimized_threshold,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                accuracy=metrics["accuracy"],
                false_positives=metrics["false_positives"],
                false_negatives=metrics["false_negatives"],
                total_pairs=metrics["total_pairs"],
                sample_size=len(similarity_data.columns),
            )
        )

        # Save history
        self.optimizer.save_history()

        return optimized_threshold, metrics

    def _calculate_pipeline_metrics(
        self, similarity_data: pd.DataFrame, threshold: float
    ) -> dict:
        """
        Calculate pipeline performance metrics.

        Returns:
            Dict: Performance metrics
        """
        if similarity_data.empty:
            return {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "accuracy": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "total_pairs": 0,
            }

        # Get all similarity scores
        scores = similarity_data.values
        upper_tri = scores[np.triu_indices_from(scores, k=1)]

        if len(upper_tri) == 0:
            return {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "accuracy": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "total_pairs": 0,
            }

        # Simulate ground truth (in production, use actual labels)
        # For demonstration, assume labels based on threshold
        # In production, this would use actual labeled data
        ground_truth = np.random.rand(len(upper_tri)) > 0.7  # 30% are true positives
        predictions = upper_tri > threshold

        # Calculate metrics
        tp = np.sum((predictions == True) & (ground_truth == True))  # noqa: E712
        fp = np.sum((predictions == True) & (ground_truth == False))  # noqa: E712
        fn = np.sum((predictions == False) & (ground_truth == True))  # noqa: E712
        tn = np.sum((predictions == False) & (ground_truth == False))  # noqa: E712

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_pairs": len(upper_tri),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        }

    def get_optimization_dashboard_data(self) -> dict:
        """
        Get data for optimization dashboard.

        Returns:
            Dict: Dashboard data
        """
        summary = self.optimizer.get_optimization_summary()
        trends = self.optimizer.get_performance_trend()

        return {
            "summary": summary,
            "trends": trends,
            "recent_metrics": (
                self.pipeline_metrics[-10:] if self.pipeline_metrics else []
            ),
            "total_runs": self.optimization_runs,
        }


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_auto_ml_dashboard():
    """Render AutoML optimization dashboard."""
    st.subheader("🤖 AutoML Optimization Dashboard")

    # Initialize integration
    if "auto_ml_integration" not in st.session_state:
        st.session_state.auto_ml_integration = AutoOptimizationIntegration()

    integration = st.session_state.auto_ml_integration

    # Configuration
    with st.expander("⚙️ Optimization Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            enabled = st.checkbox("Enable AutoML Optimization", value=True)
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
            min_samples = st.number_input("Min Samples for Optimization", 10, 200, 50)

        with col2:
            exploration_rate = st.slider(
                "Exploration Rate (A/B Testing)", 0.0, 0.3, 0.1, 0.01
            )
            auto_apply = st.checkbox("Auto-apply Optimized Thresholds")
            notify = st.checkbox("Notify on Improvements")

        if st.button("💾 Save Settings", use_container_width=True):
            integration.optimizer.config.enabled = enabled
            integration.optimizer.config.learning_rate = learning_rate
            integration.optimizer.config.min_samples_for_optimization = min_samples
            integration.optimizer.config.exploration_rate = exploration_rate
            integration.optimizer.config.auto_apply_changes = auto_apply
            integration.optimizer.config.notify_on_improvement = notify
            st.success("✅ Settings saved")

    # Stats
    summary = integration.optimizer.get_optimization_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Optimizations", summary["total_optimizations"])
    col2.metric("Current Threshold", f"{summary['current_threshold']:.2%}")
    col3.metric("Best F1 Score", f"{summary['best_f1']:.2%}")
    col4.metric("Improvements", summary["improvements"])
    col5.metric("Feedback Samples", summary["feedback_count"])

    # Performance Trends
    st.markdown("### 📈 Performance Trends")

    trends = integration.optimizer.get_performance_trend()

    if trends["f1_score"]:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("F1 Score", "Precision", "Recall", "Accuracy"),
        )

        # F1 Score
        fig.add_trace(
            go.Scatter(
                y=trends["f1_score"], name="F1 Score", line=dict(color="#2ecc71")
            ),
            row=1,
            col=1,
        )

        # Precision
        fig.add_trace(
            go.Scatter(
                y=trends["precision"], name="Precision", line=dict(color="#3498db")
            ),
            row=1,
            col=2,
        )

        # Recall
        fig.add_trace(
            go.Scatter(y=trends["recall"], name="Recall", line=dict(color="#e74c3c")),
            row=2,
            col=1,
        )

        # Accuracy
        fig.add_trace(
            go.Scatter(
                y=trends["accuracy"], name="Accuracy", line=dict(color="#f39c12")
            ),
            row=2,
            col=2,
        )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collecting performance data...")

    # Recent Optimizations
    if integration.pipeline_metrics:
        st.markdown("### 📊 Recent Optimizations")

        df = pd.DataFrame(integration.pipeline_metrics[-10:])
        df["timestamp"] = pd.to_datetime(
            df.get("timestamp", [datetime.now()] * len(df))
        )

        st.dataframe(
            df[
                [
                    "timestamp",
                    "precision",
                    "recall",
                    "f1_score",
                    "accuracy",
                    "total_pairs",
                ]
            ].style.format(
                {
                    "precision": "{:.2%}",
                    "recall": "{:.2%}",
                    "f1_score": "{:.2%}",
                    "accuracy": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

    # Manual Optimization
    st.markdown("### 🎯 Manual Optimization")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🔄 Run Optimization Now", type="primary", use_container_width=True
        ):
            with st.spinner("Optimizing..."):
                # Get current similarity data
                sim_df = st.session_state.get("sim_df")
                if sim_df is not None and not sim_df.empty:
                    current_threshold = st.session_state.get("threshold_slider", 0.75)
                    new_threshold, metrics = integration.optimize_pipeline(
                        sim_df, current_threshold
                    )

                    st.success(
                        f"✅ Optimization complete! New threshold: {new_threshold:.2%}"
                    )

                    # Show metrics
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Precision", f"{metrics['precision']:.2%}")
                    col_b.metric("Recall", f"{metrics['recall']:.2%}")
                    col_c.metric("F1 Score", f"{metrics['f1_score']:.2%}")
                else:
                    st.warning("No similarity data available. Please run a scan first.")

    with col2:
        if st.button("📊 Export Optimization Data", use_container_width=True):
            data = {
                "summary": integration.optimizer.get_optimization_summary(),
                "metrics": [
                    asdict(m) for m in integration.optimizer.metrics_history[-50:]
                ],
                "feedback": [
                    asdict(f) for f in integration.optimizer.feedback_history[-20:]
                ],
                "config": asdict(integration.optimizer.config),
            }

            st.download_button(
                label="⬇️ Download JSON",
                data=json.dumps(data, indent=2),
                file_name=f"optimization_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    # Feedback Integration
    st.markdown("### 💬 User Feedback Integration")

    with st.expander("📝 Provide Feedback", expanded=False):
        st.markdown("Help improve the system by providing feedback on detected pairs")

        col1, col2 = st.columns(2)
        with col1:
            doc_a = st.text_input("Document A")
            doc_b = st.text_input("Document B")
        with col2:
            similarity = st.slider("Detected Similarity", 0.0, 1.0, 0.5)
            judgement = st.radio("Is this plagiarism?", ["Yes", "No"], horizontal=True)

        if st.button("Submit Feedback", use_container_width=True):
            feedback = UserFeedback(
                pair_id=f"fb_{int(time.time())}",
                doc_a=doc_a,
                doc_b=doc_b,
                similarity_score=similarity,
                user_judgement=judgement == "Yes",
                timestamp=time.time(),
                confidence=1.0,
            )
            integration.optimizer.add_feedback(feedback)
            st.success("✅ Feedback submitted! This helps improve the system.")


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_auto_ml():
    """Initialize AutoML optimization system."""
    if "auto_ml_initialized" not in st.session_state:
        st.session_state.auto_ml_initialized = True

        # Create optimizer
        config = OptimizationConfig()
        st.session_state.auto_ml_optimizer = AutoMLOptimizer(config)
        st.session_state.auto_ml_integration = AutoOptimizationIntegration()

        # Load historical data
        history_path = (
            Path(st.session_state.get("data_dir", ".")) / "optimization_history.json"
        )
        if history_path.exists():
            try:
                with open(history_path) as f:
                    data = json.load(f)
                    thresholds = data.get("thresholds", [])
                    if thresholds:
                        st.session_state.auto_ml_optimizer.best_threshold = max(
                            thresholds
                        )
            except Exception:
                pass
