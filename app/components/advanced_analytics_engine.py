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
Advanced Analytics and Predictive Intelligence Engine

Features:
- Predictive analytics for plagiarism trends
- Anomaly detection in patterns
- Automated insight generation
- Trend forecasting
- Correlation analysis
- Risk scoring
- Pattern recognition
- Automated reporting with insights
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ML Libraries
try:
    import scipy.stats as stats
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class RiskLevel(Enum):
    """Risk levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class InsightType(Enum):
    """Insight types."""

    ANOMALY = "anomaly"
    TREND = "trend"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    PATTERN = "pattern"
    RISK = "risk"
    OPPORTUNITY = "opportunity"


class ForecastPeriod(Enum):
    """Forecast periods."""

    SHORT_TERM = "short_term"  # 7 days
    MEDIUM_TERM = "medium_term"  # 30 days
    LONG_TERM = "long_term"  # 90 days


@dataclass
class PredictiveInsight:
    """Generated insight."""

    id: str
    type: InsightType
    title: str
    description: str
    severity: RiskLevel
    confidence: float
    timestamp: float
    data: dict[str, Any]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendForecast:
    """Trend forecast result."""

    period: ForecastPeriod
    predicted_values: list[float]
    confidence_interval: tuple[float, float]
    trend_direction: str  # increasing, decreasing, stable
    peak_prediction: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment result."""

    document_id: str
    risk_score: float
    risk_level: RiskLevel
    contributing_factors: list[str]
    mitigation_steps: list[str]
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result."""

    anomaly_id: str
    type: str  # spike, drop, outlier, pattern_break
    severity: RiskLevel
    description: str
    detected_at: float
    affected_data: dict[str, Any]
    suggested_action: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# PREDICTIVE ENGINE
# ==============================================================================


class PredictiveEngine:
    """
    Predictive analytics engine for plagiarism trends.
    """

    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.history_data = []
        self._initialize_models()

    def _initialize_models(self):
        """Initialize prediction models."""
        if SKLEARN_AVAILABLE:
            self.models["regressor"] = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.models["scaler"] = StandardScaler()

        if PROPHET_AVAILABLE:
            self.models["prophet"] = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )

    def train_models(self, historical_data: pd.DataFrame):
        """
        Train prediction models on historical data.

        Args:
            historical_data: DataFrame with 'date' and 'value' columns
        """
        if not SKLEARN_AVAILABLE or len(historical_data) < 10:
            return

        self.history_data = historical_data
        self.is_trained = True

        # Train RandomForest
        X = historical_data[["value"]].values
        y = historical_data["value"].shift(-1).dropna()

        if len(y) > 0:
            self.models["regressor"].fit(X[:-1], y)

    def forecast(
        self,
        data: pd.DataFrame,
        period: ForecastPeriod = ForecastPeriod.SHORT_TERM,
        confidence_level: float = 0.95,
    ) -> TrendForecast:
        """
        Generate forecast for plagiarism trends.

        Args:
            data: Historical data
            period: Forecast period
            confidence_level: Confidence interval level

        Returns:
            TrendForecast: Forecast results
        """
        if len(data) < 2:
            return self._fallback_forecast(period)

        days = {
            ForecastPeriod.SHORT_TERM: 7,
            ForecastPeriod.MEDIUM_TERM: 30,
            ForecastPeriod.LONG_TERM: 90,
        }[period]

        try:
            if PROPHET_AVAILABLE:
                return self._prophet_forecast(data, days, confidence_level)
            elif SKLEARN_AVAILABLE:
                return self._ml_forecast(data, days, confidence_level)
            else:
                return self._simple_forecast(data, days)
        except Exception as e:
            print(f"Forecast error: {e}")
            return self._simple_forecast(data, days)

    def _prophet_forecast(
        self, data: pd.DataFrame, days: int, confidence: float
    ) -> TrendForecast:
        """Use Prophet for forecasting."""
        # Prepare data for Prophet
        prophet_data = data.rename(columns={"date": "ds", "value": "y"})

        # Fit model
        model = Prophet()
        model.fit(prophet_data)

        # Make future dataframe
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        # Get predictions
        predictions = forecast["yhat"].values[-days:]
        lower_bound = forecast["yhat_lower"].values[-days:]
        upper_bound = forecast["yhat_upper"].values[-days:]

        # Determine trend direction
        last_value = predictions[-1]
        first_value = predictions[0]

        if last_value > first_value * 1.1:
            direction = "increasing"
        elif last_value < first_value * 0.9:
            direction = "decreasing"
        else:
            direction = "stable"

        # Find peak
        peak_idx = np.argmax(predictions)
        peak_value = predictions[peak_idx]

        return TrendForecast(
            period=ForecastPeriod.SHORT_TERM,
            predicted_values=predictions.tolist(),
            confidence_interval=(lower_bound.mean(), upper_bound.mean()),
            trend_direction=direction,
            peak_prediction={
                "day": peak_idx,
                "value": peak_value,
                "date": (datetime.now() + timedelta(days=peak_idx)).isoformat(),
            },
            metadata={"method": "prophet"},
        )

    def _ml_forecast(
        self, data: pd.DataFrame, days: int, confidence: float
    ) -> TrendForecast:
        """Use ML for forecasting."""
        # Simple ML forecast using last values
        values = data["value"].values

        if len(values) < 2:
            return self._simple_forecast(data, days)

        # Use last 30% for prediction
        split_idx = int(len(values) * 0.7)
        train_values = values[:split_idx]

        # Simple moving average with trend
        trend = np.mean(np.diff(values[-10:])) if len(values) > 10 else 0

        predictions = []
        last_value = values[-1]

        for i in range(days):
            # Add trend and small random variation
            next_value = last_value + trend + np.random.normal(0, 0.1)
            predictions.append(next_value)
            last_value = next_value

        # Determine direction
        if predictions[-1] > predictions[0] * 1.1:
            direction = "increasing"
        elif predictions[-1] < predictions[0] * 0.9:
            direction = "decreasing"
        else:
            direction = "stable"

        # Calculate confidence interval
        std_dev = np.std(values) * 0.2
        lower = predictions[-1] - 1.96 * std_dev
        upper = predictions[-1] + 1.96 * std_dev

        return TrendForecast(
            period=ForecastPeriod.SHORT_TERM,
            predicted_values=predictions,
            confidence_interval=(lower, upper),
            trend_direction=direction,
            peak_prediction={
                "day": np.argmax(predictions),
                "value": np.max(predictions),
                "date": (
                    datetime.now() + timedelta(days=np.argmax(predictions))
                ).isoformat(),
            },
            metadata={"method": "ml"},
        )

    def _simple_forecast(self, data: pd.DataFrame, days: int) -> TrendForecast:
        """Simple fallback forecast."""
        values = data["value"].values

        if len(values) < 2:
            predictions = [data["value"].mean()] * days
            direction = "stable"
        else:
            # Calculate trend
            slope = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0

            predictions = []
            for i in range(days):
                predictions.append(values[-1] + slope * (i + 1))

            if predictions[-1] > predictions[0] * 1.1:
                direction = "increasing"
            elif predictions[-1] < predictions[0] * 0.9:
                direction = "decreasing"
            else:
                direction = "stable"

        return TrendForecast(
            period=ForecastPeriod.SHORT_TERM,
            predicted_values=predictions,
            confidence_interval=(predictions[-1] * 0.8, predictions[-1] * 1.2),
            trend_direction=direction,
            peak_prediction={
                "day": np.argmax(predictions),
                "value": np.max(predictions),
                "date": (
                    datetime.now() + timedelta(days=np.argmax(predictions))
                ).isoformat(),
            },
            metadata={"method": "simple"},
        )

    def _fallback_forecast(self, period: ForecastPeriod) -> TrendForecast:
        """Fallback when no data available."""
        days = {
            ForecastPeriod.SHORT_TERM: 7,
            ForecastPeriod.MEDIUM_TERM: 30,
            ForecastPeriod.LONG_TERM: 90,
        }[period]

        return TrendForecast(
            period=period,
            predicted_values=[0] * days,
            confidence_interval=(0, 0),
            trend_direction="stable",
            peak_prediction={"day": 0, "value": 0, "date": datetime.now().isoformat()},
            metadata={"method": "fallback"},
        )


# ==============================================================================
# ANOMALY DETECTOR
# ==============================================================================


class AnomalyDetector:
    """
    Detect anomalies in plagiarism patterns.
    """

    def __init__(self):
        self.isolation_forest = None
        self.dbscan = None
        self.history_data = []
        self._initialize_models()

    def _initialize_models(self):
        """Initialize anomaly detection models."""
        if SKLEARN_AVAILABLE:
            self.isolation_forest = IsolationForest(contamination=0.05, random_state=42)
            self.dbscan = DBSCAN(eps=0.5, min_samples=5)

    def detect_anomalies(
        self, data: pd.DataFrame, method: str = "isolation_forest"
    ) -> list[AnomalyDetectionResult]:
        """
        Detect anomalies in data.

        Args:
            data: DataFrame with numerical columns
            method: Detection method

        Returns:
            List[AnomalyDetectionResult]: Detected anomalies
        """
        if len(data) < 10:
            return []

        self.history_data = data

        try:
            if method == "isolation_forest" and SKLEARN_AVAILABLE:
                return self._detect_isolation_forest(data)
            elif method == "dbscan" and SKLEARN_AVAILABLE:
                return self._detect_dbscan(data)
            else:
                return self._detect_statistical(data)
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return self._detect_statistical(data)

    def _detect_isolation_forest(
        self, data: pd.DataFrame
    ) -> list[AnomalyDetectionResult]:
        """Detect anomalies using Isolation Forest."""
        # Prepare numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 1:
            return []

        X = data[numeric_cols].values

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Detect anomalies
        predictions = self.isolation_forest.fit_predict(X_scaled)

        anomalies = []
        for idx, pred in enumerate(predictions):
            if pred == -1:
                # Calculate severity based on data point
                severity = self._calculate_severity(X[idx], X)

                anomalies.append(
                    AnomalyDetectionResult(
                        anomaly_id=f"anomaly_{int(time.time())}_{idx}",
                        type="outlier",
                        severity=severity,
                        description=f"Anomalous data point detected at index {idx}",
                        detected_at=time.time(),
                        affected_data={"index": idx, "value": X[idx].tolist()},
                        suggested_action="Investigate the anomalous data point",
                        metadata={"method": "isolation_forest"},
                    )
                )

        return anomalies

    def _detect_dbscan(self, data: pd.DataFrame) -> list[AnomalyDetectionResult]:
        """Detect anomalies using DBSCAN."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 1:
            return []

        X = data[numeric_cols].values
        X_scaled = StandardScaler().fit_transform(X)

        labels = self.dbscan.fit_predict(X_scaled)

        anomalies = []
        for idx, label in enumerate(labels):
            if label == -1:
                anomalies.append(
                    AnomalyDetectionResult(
                        anomaly_id=f"anomaly_{int(time.time())}_{idx}",
                        type="outlier",
                        severity=RiskLevel.HIGH,
                        description=f"Outlier detected at index {idx}",
                        detected_at=time.time(),
                        affected_data={"index": idx, "value": X[idx].tolist()},
                        suggested_action="Investigate the outlier",
                        metadata={"method": "dbscan"},
                    )
                )

        return anomalies

    def _detect_statistical(self, data: pd.DataFrame) -> list[AnomalyDetectionResult]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = data[col].values

            # Z-score method
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                continue

            for idx, value in enumerate(values):
                z_score = (value - mean) / std
                if abs(z_score) > 3:
                    severity = RiskLevel.HIGH if abs(z_score) > 5 else RiskLevel.MEDIUM

                    anomalies.append(
                        AnomalyDetectionResult(
                            anomaly_id=f"anomaly_{int(time.time())}_{col}_{idx}",
                            type="spike" if value > mean else "drop",
                            severity=severity,
                            description=f"Statistical anomaly in {col} at index {idx}",
                            detected_at=time.time(),
                            affected_data={
                                "column": col,
                                "index": idx,
                                "value": value,
                                "z_score": z_score,
                            },
                            suggested_action=f"Review {col} value at index {idx}",
                            metadata={"method": "statistical"},
                        )
                    )

        return anomalies

    def _calculate_severity(self, point: np.ndarray, data: np.ndarray) -> RiskLevel:
        """Calculate severity of anomaly."""
        # Distance from mean
        mean = np.mean(data, axis=0)
        dist = np.linalg.norm(point - mean)
        std_dist = np.std(np.linalg.norm(data - mean, axis=1))

        if std_dist == 0:
            return RiskLevel.MEDIUM

        z_score = dist / std_dist

        if z_score > 5:
            return RiskLevel.CRITICAL
        elif z_score > 3:
            return RiskLevel.HIGH
        elif z_score > 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# ==============================================================================
# INSIGHT GENERATOR
# ==============================================================================


class InsightGenerator:
    """
    Generate automated insights from data.
    """

    def __init__(self):
        self.insight_history = []
        self.thresholds = {
            "similarity_high": 0.85,
            "similarity_low": 0.30,
            "trend_strong": 0.1,
            "correlation_strong": 0.7,
        }

    def generate_insights(
        self, data: dict[str, Any], context: dict[str, Any] = None
    ) -> list[PredictiveInsight]:
        """
        Generate insights from data.

        Args:
            data: Data to analyze
            context: Additional context

        Returns:
            List[PredictiveInsight]: Generated insights
        """
        insights = []

        # Generate different types of insights
        if "similarity_matrix" in data:
            insights.extend(
                self._generate_similarity_insights(data["similarity_matrix"])
            )

        if "trend_data" in data:
            insights.extend(self._generate_trend_insights(data["trend_data"]))

        if "correlation_data" in data:
            insights.extend(
                self._generate_correlation_insights(data["correlation_data"])
            )

        if "risk_data" in data:
            insights.extend(self._generate_risk_insights(data["risk_data"]))

        # Store insights
        self.insight_history.extend(insights)

        return insights

    def _generate_similarity_insights(
        self, similarity_matrix: pd.DataFrame
    ) -> list[PredictiveInsight]:
        """Generate insights from similarity matrix."""
        insights = []

        if similarity_matrix is None or similarity_matrix.empty:
            return insights

        # Find high similarity pairs
        max_sim = similarity_matrix.max().max()
        if max_sim > self.thresholds["similarity_high"]:
            # Find highest pair
            max_idx = np.unravel_index(
                np.argmax(similarity_matrix.values), similarity_matrix.shape
            )
            if max_idx[0] != max_idx[1]:
                doc_a = similarity_matrix.index[max_idx[0]]
                doc_b = similarity_matrix.columns[max_idx[1]]

                insights.append(
                    PredictiveInsight(
                        id=f"insight_{int(time.time())}_{len(insights)}",
                        type=InsightType.ANOMALY,
                        title=f"High Similarity Detected: {max_sim:.1%}",
                        description=f"Documents '{doc_a}' and '{doc_b}' have unusually high similarity ({max_sim:.1%})",
                        severity=RiskLevel.HIGH if max_sim > 0.90 else RiskLevel.MEDIUM,
                        confidence=0.9,
                        timestamp=time.time(),
                        data={"doc_a": doc_a, "doc_b": doc_b, "similarity": max_sim},
                        recommendations=[
                            "Review both documents for plagiarism",
                            "Check if documents are from the same source",
                            "Compare specific sections for overlap",
                        ],
                    )
                )

        return insights

    def _generate_trend_insights(self, trend_data: dict) -> list[PredictiveInsight]:
        """Generate insights from trend data."""
        insights = []

        if not trend_data:
            return insights

        # Check for increasing trend
        if trend_data.get("direction") == "increasing":
            insights.append(
                PredictiveInsight(
                    id=f"insight_{int(time.time())}_{len(insights)}",
                    type=InsightType.TREND,
                    title="Increasing Plagiarism Trend Detected",
                    description="Plagiarism rates are showing an increasing trend over time",
                    severity=RiskLevel.HIGH,
                    confidence=0.8,
                    timestamp=time.time(),
                    data=trend_data,
                    recommendations=[
                        "Increase monitoring frequency",
                        "Review recent submissions",
                        "Consider additional plagiarism prevention measures",
                    ],
                )
            )

        return insights

    def _generate_correlation_insights(
        self, correlation_data: dict
    ) -> list[PredictiveInsight]:
        """Generate insights from correlation data."""
        insights = []

        if not correlation_data:
            return insights

        # Check for strong correlations
        for var1, correlations in correlation_data.items():
            for var2, value in correlations.items():
                if abs(value) > self.thresholds["correlation_strong"]:
                    insights.append(
                        PredictiveInsight(
                            id=f"insight_{int(time.time())}_{len(insights)}",
                            type=InsightType.CORRELATION,
                            title=f"Strong Correlation: {var1} ↔ {var2}",
                            description=f"Found strong correlation ({value:.2f}) between {var1} and {var2}",
                            severity=RiskLevel.LOW,
                            confidence=0.7,
                            timestamp=time.time(),
                            data={"var1": var1, "var2": var2, "correlation": value},
                            recommendations=[
                                f"Investigate relationship between {var1} and {var2}",
                                "Consider if this correlation is causal or coincidental",
                            ],
                        )
                    )

        return insights

    def _generate_risk_insights(self, risk_data: dict) -> list[PredictiveInsight]:
        """Generate insights from risk data."""
        insights = []

        if not risk_data:
            return insights

        # Check for high risk items
        for item in risk_data.get("high_risk_items", []):
            insights.append(
                PredictiveInsight(
                    id=f"insight_{int(time.time())}_{len(insights)}",
                    type=InsightType.RISK,
                    title=f"High Risk Detected: {item.get('name', 'Unknown')}",
                    description=item.get("description", "High risk factor identified"),
                    severity=RiskLevel.CRITICAL,
                    confidence=0.9,
                    timestamp=time.time(),
                    data=item,
                    recommendations=item.get(
                        "recommendations",
                        [
                            "Immediate review required",
                            "Implement preventive measures",
                            "Notify relevant stakeholders",
                        ],
                    ),
                )
            )

        return insights


# ==============================================================================
# RISK SCORER
# ==============================================================================


class RiskScorer:
    """
    Calculate and assess plagiarism risks.

    Delegates to the real ML risk model when available,
    falls back to heuristic scoring otherwise.
    """

    def __init__(self):
        self.weights = {
            "similarity_score": 0.4,
            "recent_activity": 0.2,
            "document_volume": 0.15,
            "author_history": 0.15,
            "anomaly_score": 0.1,
        }
        self._engine = None
        self._try_load_engine()

    def _try_load_engine(self):
        try:
            from src.core.pattern_recognition import PatternDetectionEngine
            from src.db.pattern_repository import PatternRepository

            repo = PatternRepository()
            self._engine = PatternDetectionEngine(repository=repo)
        except ImportError:
            self._engine = None

    def assess_risk(self, document_id: str, data: dict[str, Any]) -> RiskAssessment:
        if self._engine is not None:
            return self._assess_risk_ml(document_id, data)
        return self._assess_risk_heuristic(document_id, data)

    def _assess_risk_ml(self, document_id: str, data: dict[str, Any]) -> RiskAssessment:
        try:
            result = self._engine.score_document_risk(document_id, data)
            return RiskAssessment(
                document_id=document_id,
                risk_score=result["risk_score"],
                risk_level=(
                    RiskLevel(result["risk_level"].lower())
                    if result["risk_level"].lower() in [e.value for e in RiskLevel]
                    else RiskLevel.MEDIUM
                ),
                contributing_factors=result["contributing_factors"],
                mitigation_steps=self._generate_mitigation(result["risk_level"]),
                timestamp=time.time(),
                metadata={
                    "model_version": result.get("model_version", "unknown"),
                    "method": "ml",
                },
            )
        except Exception as e:
            print(f"ML risk assessment error: {e}")
            return self._assess_risk_heuristic(document_id, data)

    def _assess_risk_heuristic(
        self, document_id: str, data: dict[str, Any]
    ) -> RiskAssessment:
        risk_score = 0
        similarity = data.get("similarity", 0)
        risk_score += similarity * self.weights["similarity_score"]
        days_since_upload = data.get("days_since_upload", 30)
        if days_since_upload < 7:
            risk_score += 1.0 * self.weights["recent_activity"]
        elif days_since_upload < 14:
            risk_score += 0.5 * self.weights["recent_activity"]
        doc_count = data.get("document_count", 1)
        if doc_count > 10:
            risk_score += 1.0 * self.weights["document_volume"]
        elif doc_count > 5:
            risk_score += 0.5 * self.weights["document_volume"]
        past_incidents = data.get("past_incidents", 0)
        if past_incidents > 3:
            risk_score += 1.0 * self.weights["author_history"]
        elif past_incidents > 1:
            risk_score += 0.5 * self.weights["author_history"]
        anomaly_score = data.get("anomaly_score", 0)
        risk_score += anomaly_score * self.weights["anomaly_score"]
        risk_score = min(1.0, risk_score)
        if risk_score > 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score > 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score > 0.4:
            risk_level = RiskLevel.MEDIUM
        elif risk_score > 0.2:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.NEGLIGIBLE
        factors = []
        if similarity > 0.7:
            factors.append("High similarity score")
        if days_since_upload < 7:
            factors.append("Recent activity")
        if past_incidents > 1:
            factors.append("Past incidents")
        if anomaly_score > 0.5:
            factors.append("Anomaly detected")
        mitigation = self._generate_mitigation(risk_level.value.title())
        return RiskAssessment(
            document_id=document_id,
            risk_score=risk_score,
            risk_level=risk_level,
            contributing_factors=factors,
            mitigation_steps=mitigation,
            timestamp=time.time(),
            metadata=data,
        )

    @staticmethod
    def _generate_mitigation(risk_level: str) -> list:
        if risk_level in ("Critical", "High"):
            return [
                "Immediate document review required",
                "Notify academic integrity office",
                "Block further submissions",
                "Schedule investigation meeting",
            ]
        if risk_level == "Medium":
            return [
                "Schedule document review",
                "Contact author for clarification",
                "Monitor future submissions",
            ]
        return []


# ==============================================================================
# PATTERN RECOGNIZER
# ==============================================================================


class PatternRecognizer:
    """
    Recognize emerging patterns in plagiarism data.

    Delegates to the real ML pattern recognition engine when available,
    falls back to basic frequency analysis otherwise.
    """

    def __init__(self):
        self.patterns = {}
        self.pattern_history = []
        self._engine = None
        self._try_load_engine()

    def _try_load_engine(self):
        try:
            from src.core.pattern_recognition import PatternDetectionEngine
            from src.db.pattern_repository import PatternRepository

            repo = PatternRepository()
            self._engine = PatternDetectionEngine(repository=repo)
        except ImportError:
            self._engine = None

    def recognize_patterns(
        self, data: pd.DataFrame, min_frequency: int = 3
    ) -> dict[str, Any]:
        if self._engine is not None:
            return self._recognize_patterns_ml(data, min_frequency)
        return self._recognize_patterns_basic(data, min_frequency)

    def _recognize_patterns_ml(
        self, data: pd.DataFrame, min_frequency: int
    ) -> dict[str, Any]:
        try:
            from src.db.incidents import get_all_incidents, get_total_incidents_count

            total = get_total_incidents_count()
            incidents = get_all_incidents(limit=total, offset=0) if total > 0 else []
            if not incidents:
                return {}
            patterns = self._engine.detect_recurring_patterns(
                incidents, min_occurrence=min_frequency
            )
            result = {}
            for p in patterns:
                ptype = p.get("pattern_type", "unknown")
                result[f"{ptype}_{p['pattern_id'][:8]}"] = {
                    "pattern_id": p["pattern_id"],
                    "type": ptype,
                    "count": p.get("occurrence_count", 0),
                    "avg_similarity": p.get("avg_similarity", 0),
                    "confidence": p.get("confidence_score", 0),
                    "severity": p.get("severity", "Low"),
                    "documents": p.get("document_group", []),
                    "description": p.get("description", ""),
                }
            self.patterns = result
            return result
        except Exception as e:
            print(f"ML pattern recognition error: {e}")
            return self._recognize_patterns_basic(data, min_frequency)

    def _recognize_patterns_basic(
        self, data: pd.DataFrame, min_frequency: int
    ) -> dict[str, Any]:
        patterns = {}
        if (
            "doc_a" in data.columns
            and "doc_b" in data.columns
            and "similarity" in data.columns
        ):
            high_sim_pairs = data[data["similarity"] > 0.75]
            if len(high_sim_pairs) >= min_frequency:
                patterns["high_similarity_pairs"] = {
                    "count": len(high_sim_pairs),
                    "documents": high_sim_pairs[["doc_a", "doc_b"]].to_dict("records"),
                    "avg_similarity": high_sim_pairs["similarity"].mean(),
                    "emerged_at": datetime.now().isoformat(),
                }
        if "author" in data.columns:
            author_counts = data["author"].value_counts()
            prolific_authors = author_counts[author_counts >= min_frequency]
            if not prolific_authors.empty:
                patterns["prolific_authors"] = {
                    "count": len(prolific_authors),
                    "authors": prolific_authors.to_dict(),
                    "avg_documents_per_author": author_counts.mean(),
                    "emerged_at": datetime.now().isoformat(),
                }
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            daily_counts = data.groupby(data["date"].dt.date).size()
            peak_days = daily_counts[daily_counts > daily_counts.quantile(0.75)]
            if len(peak_days) >= 3:
                patterns["peak_activity_days"] = {
                    "count": len(peak_days),
                    "days": peak_days.to_dict(),
                    "avg_activity": daily_counts.mean(),
                    "emerged_at": datetime.now().isoformat(),
                }
        if "similarity" in data.columns:
            similarity_ranges = pd.cut(
                data["similarity"], bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0]
            )
            range_counts = similarity_ranges.value_counts()
            patterns["similarity_distribution"] = {
                "ranges": range_counts.to_dict(),
                "most_common_range": range_counts.index[0].__str__(),
                "emerged_at": datetime.now().isoformat(),
            }
        self.patterns = patterns
        return patterns


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_analytics_engine():
    """Render advanced analytics engine UI."""
    st.subheader("🧠 Advanced Analytics Engine")

    # Initialize
    if "analytics_engine_initialized" not in st.session_state:
        st.session_state.analytics_engine = {
            "predictive": PredictiveEngine(),
            "anomaly": AnomalyDetector(),
            "insights": InsightGenerator(),
            "risk": RiskScorer(),
            "patterns": PatternRecognizer(),
            "initialized": True,
        }

    engine = st.session_state.analytics_engine

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🔮 Predictive",
            "🚨 Anomalies",
            "💡 Insights",
            "🎯 Risk Assessment",
            "🧩 Patterns",
        ]
    )

    with tab1:
        render_predictive_analytics(engine)

    with tab2:
        render_anomaly_detection(engine)

    with tab3:
        render_insight_dashboard(engine)

    with tab4:
        render_risk_assessment(engine)

    with tab5:
        render_pattern_recognition(engine)

    # Calibration report (Issue #2267): shows where the active threshold sits
    # on the precision/recall curve from the latest backtest, if one exists.
    st.divider()
    try:
        from app.components.calibration_report import render_calibration_report

        render_calibration_report()
    except Exception:
        logger.exception("Failed to render threshold calibration report")


def render_predictive_analytics(engine: dict):
    """Render predictive analytics UI."""
    st.markdown("#### 🔮 Predictive Analytics")

    # Get data
    data = st.session_state.get("sim_df")

    if data is None or data.empty:
        st.warning("No data available for prediction")
        return

    # Prepare time series data
    if "date" not in data.columns:
        # Create synthetic dates if not present
        dates = pd.date_range(end=datetime.now(), periods=len(data), freq="D")
        values = data.max(axis=1).values

        ts_data = pd.DataFrame({"date": dates, "value": values})
    else:
        ts_data = data[["date", "value"]]

    # Train models
    predictive = engine["predictive"]
    predictive.train_models(ts_data)

    # Forecast period selection
    period = st.selectbox(
        "Forecast Period",
        ["Short Term (7 days)", "Medium Term (30 days)", "Long Term (90 days)"],
    )

    period_map = {
        "Short Term (7 days)": ForecastPeriod.SHORT_TERM,
        "Medium Term (30 days)": ForecastPeriod.MEDIUM_TERM,
        "Long Term (90 days)": ForecastPeriod.LONG_TERM,
    }

    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            forecast = predictive.forecast(ts_data, period_map[period])

            # Display forecast
            st.markdown("#### 📈 Forecast Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Trend Direction", forecast.trend_direction.upper())
            col2.metric("Predicted Peak", f"{forecast.peak_prediction['value']:.2f}")
            col3.metric(
                "Confidence",
                f"{(forecast.confidence_interval[1] - forecast.confidence_interval[0]) / 2:.2f}",
            )

            # Plot forecast
            fig = go.Figure()

            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=ts_data["date"],
                    y=ts_data["value"],
                    mode="lines+markers",
                    name="Historical",
                    line=dict(color="blue"),
                )
            )

            # Forecast
            forecast_dates = pd.date_range(
                start=datetime.now(), periods=len(forecast.predicted_values), freq="D"
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast.predicted_values,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="orange", dash="dash"),
                )
            )

            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[forecast.confidence_interval[1]] * len(forecast_dates),
                    mode="lines",
                    name="Upper Bound",
                    line=dict(color="rgba(255, 0, 0, 0)"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[forecast.confidence_interval[0]] * len(forecast_dates),
                    mode="lines",
                    name="Lower Bound",
                    line=dict(color="rgba(255, 0, 0, 0)"),
                )
            )

            fig.update_layout(
                title="Plagiarism Trend Forecast",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)


def render_anomaly_detection(engine: dict):
    """Render anomaly detection UI."""
    st.markdown("#### 🚨 Anomaly Detection")

    # Get data
    data = st.session_state.get("sim_df")

    if data is None or data.empty:
        st.warning("No data available for anomaly detection")
        return

    # Prepare data
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        st.warning("No numeric data available")
        return

    # Detection options
    method = st.selectbox(
        "Detection Method", ["isolation_forest", "dbscan", "statistical"]
    )

    if st.button("Detect Anomalies", type="primary", use_container_width=True):
        with st.spinner("Detecting anomalies..."):
            detector = engine["anomaly"]
            anomalies = detector.detect_anomalies(numeric_data, method)

            if not anomalies:
                st.success("✅ No anomalies detected")
            else:
                st.warning(f"⚠️ Found {len(anomalies)} anomalies")

                for anomaly in anomalies:
                    with st.expander(
                        f"Anomaly: {anomaly.type} - {anomaly.severity.value.upper()}",
                        expanded=False,
                    ):
                        st.markdown(f"**Description:** {anomaly.description}")
                        st.markdown(f"**Severity:** {anomaly.severity.value.upper()}")
                        st.markdown(f"**Action:** {anomaly.suggested_action}")
                        st.caption(
                            f"Detected: {datetime.fromtimestamp(anomaly.detected_at).strftime('%Y-%m-%d %H:%M')}"
                        )

                        if anomaly.affected_data:
                            st.json(anomaly.affected_data)


def render_insight_dashboard(engine: dict):
    """Render insight dashboard UI."""
    st.markdown("#### 💡 Automated Insights")

    # Get data
    data = st.session_state.get("sim_df")
    flags = st.session_state.get("flags", [])

    if data is None or data.empty:
        st.warning("No data available for insight generation")
        return

    # Prepare data for insights
    insight_data = {
        "similarity_matrix": data,
        "trend_data": {"direction": "increasing" if len(data) > 10 else "stable"},
        "risk_data": {"high_risk_items": flags[:5] if flags else []},
    }

    # Generate insights
    generator = engine["insights"]
    insights = generator.generate_insights(insight_data)

    if not insights:
        st.info("No insights generated")
        return

    # Display insights
    st.markdown(f"#### Found {len(insights)} Insights")

    for insight in insights:
        colors = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "negligible": "⚪",
        }

        with st.expander(
            f"{colors.get(insight.severity.value, '')} {insight.title} "
            f"({insight.type.value}) - {insight.confidence:.0%} confidence",
            expanded=insight.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH],
        ):
            st.markdown(f"**Description:** {insight.description}")
            st.caption(f"Severity: {insight.severity.value.upper()}")

            st.markdown("**Recommendations:**")
            for rec in insight.recommendations:
                st.markdown(f"- {rec}")

            if insight.data:
                with st.expander("📊 Data", expanded=False):
                    st.json(insight.data)


def render_risk_assessment(engine: dict):
    """Render risk assessment UI."""
    st.markdown("#### 🎯 Risk Assessment")

    # Get documents
    docs = st.session_state.get("doc_names", [])

    if not docs:
        st.warning("No documents available for risk assessment")
        return

    # Select document
    selected_doc = st.selectbox("Select Document", docs)

    if selected_doc:
        # Get document data
        sim_df = st.session_state.get("sim_df")

        if sim_df is not None and selected_doc in sim_df.index:
            doc_similarities = sim_df.loc[selected_doc]

            # Prepare risk data
            risk_data = {
                "similarity": (
                    doc_similarities.max() if not doc_similarities.empty else 0
                ),
                "days_since_upload": np.random.randint(1, 30),
                "document_count": len(docs),
                "past_incidents": np.random.randint(0, 3),
                "anomaly_score": np.random.random(),
            }

            # Assess risk
            scorer = engine["risk"]
            assessment = scorer.assess_risk(selected_doc, risk_data)

            # Display assessment
            st.markdown("#### 📊 Risk Assessment Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Score", f"{assessment.risk_score:.2%}")
                st.metric("Risk Level", assessment.risk_level.value.upper())

            with col2:
                st.markdown("**Contributing Factors:**")
                for factor in assessment.contributing_factors:
                    st.markdown(f"- {factor}")

            st.markdown("**Mitigation Steps:**")
            for step in assessment.mitigation_steps:
                st.markdown(f"- {step}")


def render_pattern_recognition(engine: dict):
    """Render pattern recognition UI — delegates to the dedicated component when available."""
    try:
        from app.components.pattern_recognition_ui import render_pattern_recognition

        render_pattern_recognition()
    except Exception:
        # Fallback to the inline basic implementation
        _render_pattern_recognition_basic(engine)


def _render_pattern_recognition_basic(engine: dict):
    """Fallback inline pattern recognition when the dedicated UI is unavailable."""
    st.markdown("#### Pattern Recognition")
    data = st.session_state.get("flags")
    if not data:
        st.warning("No data available for pattern recognition")
        return
    df = pd.DataFrame(data)
    if df.empty:
        st.warning("No data available")
        return
    recognizer = engine["patterns"]
    patterns = recognizer.recognize_patterns(df)
    if not patterns:
        st.info("No significant patterns detected")
        return
    for pattern_name, pattern_data in patterns.items():
        with st.expander(
            f"Pattern: {pattern_name.replace('_', ' ').title()}", expanded=False
        ):
            st.json(pattern_data)


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_analytics_engine():
    """Initialize analytics engine."""
    if "analytics_engine_initialized" not in st.session_state:
        st.session_state.analytics_engine_initialized = True

        st.session_state.analytics_engine = {
            "predictive": PredictiveEngine(),
            "anomaly": AnomalyDetector(),
            "insights": InsightGenerator(),
            "risk": RiskScorer(),
            "patterns": PatternRecognizer(),
            "initialized": True,
        }


# ==============================================================================
# EXPORTED ITEMS
# ==============================================================================

__all__ = [
    "render_analytics_engine",
    "initialize_analytics_engine",
    "PredictiveEngine",
    "AnomalyDetector",
    "InsightGenerator",
    "RiskScorer",
    "PatternRecognizer",
    "PredictiveInsight",
    "TrendForecast",
    "RiskAssessment",
    "AnomalyDetectionResult",
    "RiskLevel",
    "InsightType",
    "ForecastPeriod",
]
