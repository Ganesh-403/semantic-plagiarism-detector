"""
Plagiarism Anomaly Detection Engine.

Detects unusual patterns in document collections that may indicate
plagiarism, collusion, or academic dishonesty using statistical
anomaly detection and pattern analysis.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected."""

    CLUSTER = "cluster"
    OUTLIER = "outlier"
    PATTERN = "pattern"
    COLLUSION = "collusion"
    TEMPLATE = "template"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"


class AnomalySeverity(Enum):
    """Severity of detected anomalies."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """A detected anomaly."""

    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    title: str
    description: str
    affected_documents: list[str]
    confidence: float
    evidence: dict[str, Any]
    detected_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_documents": self.affected_documents,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "detected_at": self.detected_at,
        }


@dataclass
class AnomalyResult:
    """Complete anomaly detection result."""

    anomalies: List[Anomaly]
    summary: Dict[str, Any]
    statistics: Dict[str, Any]
    recommendations: List[str]
    processing_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomalies": [a.to_dict() for a in self.anomalies],
            "summary": self.summary,
            "statistics": self.statistics,
            "recommendations": self.recommendations,
            "processing_time": self.processing_time,
        }


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""

    z_score_threshold: float = 2.5
    cluster_min_size: int = 3
    cluster_similarity_threshold: float = 0.85
    outlier_percentile: float = 95
    pattern_min_length: int = 20
    collusion_threshold: float = 0.80
    template_threshold: float = 0.75
    enable_statistical: bool = True
    enable_cluster: bool = True
    enable_pattern: bool = True
    enable_collusion: bool = True


class StatisticalAnalyzer:
    """Statistical methods for anomaly detection."""

    def __init__(self, config: AnomalyConfig):
        self.config = config

    def z_score_analysis(
        self, scores: List[float]
    ) -> List[Tuple[int, float, AnomalySeverity]]:
        """Identify outliers using Z-score."""
        if len(scores) < 3:
            return []
        arr = np.array(scores)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return []
        anomalies = []
        for i, score in enumerate(scores):
            z = (score - mean) / std
            if abs(z) >= self.config.z_score_threshold:
                severity = (
                    AnomalySeverity.CRITICAL
                    if z >= 3.5
                    else AnomalySeverity.HIGH
                    if z >= 3.0
                    else AnomalySeverity.MEDIUM
                )
                anomalies.append((i, z, severity))
        return anomalies

    def iqr_analysis(self, scores: list[float]) -> list[tuple[int, float]]:
        """Identify outliers using IQR method."""
        if len(scores) < 4:
            return []
        arr = np.array(scores)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [(i, s) for i, s in enumerate(scores) if s < lower or s > upper]

    def percentile_analysis(self, scores: list[float]) -> dict[str, float]:
        """Compute percentile distribution."""
        if not scores:
            return {}
        arr = np.array(scores)
        return {
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def detect_distribution_anomaly(self, scores: list[float]) -> Optional[Anomaly]:
        """Detect if distribution shape indicates anomalies."""
        if len(scores) < 10:
            return None
        arr = np.array(scores)
        skewness = float(np.mean(((arr - np.mean(arr)) / (np.std(arr) + 1e-10)) ** 3))
        kurtosis = float(
            np.mean(((arr - np.mean(arr)) / (np.std(arr) + 1e-10)) ** 4) - 3
        )
        if abs(skewness) > 2 or kurtosis > 5:
            return Anomaly(
                anomaly_id="STAT-DIST-001",
                anomaly_type=AnomalyType.STATISTICAL,
                severity=AnomalySeverity.MEDIUM,
                title="Unusual Score Distribution",
                description=f"Distribution shows {'heavy right tail' if skewness > 0 else 'heavy left tail'} (skewness={skewness:.2f}) with {'heavy tails' if kurtosis > 0 else 'light tails'} (kurtosis={kurtosis:.2f})",
                affected_documents=[],
                confidence=min(abs(skewness) / 5, 1.0),
                evidence={"skewness": skewness, "kurtosis": kurtosis},
                detected_at=datetime.now().isoformat(),
            )
        return None


class ClusterAnalyzer:
    """Cluster-based anomaly detection."""

    def __init__(self, config: AnomalyConfig):
        self.config = config

    def find_similarity_clusters(
        self, doc_names: List[str], similarity_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Find clusters of highly similar documents."""
        n = len(doc_names)
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if similarity_matrix[i, j] >= self.config.cluster_similarity_threshold:
                    cluster.append(j)
            if len(cluster) >= self.config.cluster_min_size:
                visited.update(cluster)
                clusters.append(
                    {
                        "documents": [doc_names[idx] for idx in cluster],
                        "size": len(cluster),
                        "avg_similarity": float(
                            np.mean(
                                [
                                    similarity_matrix[a, b]
                                    for a in cluster
                                    for b in cluster
                                    if a < b
                                ]
                            )
                        ),
                    }
                )
        return clusters

    def detect_collusion_clusters(
        self, clusters: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect potential collusion from similarity clusters."""
        anomalies = []
        for i, cluster in enumerate(clusters):
            if (
                cluster["size"] >= self.config.cluster_min_size
                and cluster["avg_similarity"] >= self.config.collusion_threshold
            ):
                anomalies.append(
                    Anomaly(
                        anomaly_id=f"COLLUSION-{i + 1:03d}",
                        anomaly_type=AnomalyType.COLLUSION,
                        severity=AnomalySeverity.HIGH,
                        title=f"Potential Collusion Cluster ({cluster['size']} documents)",
                        description=f"Group of {cluster['size']} documents with {cluster['avg_similarity']:.1%} average similarity suggests possible collaboration or shared source.",
                        affected_documents=cluster["documents"],
                        confidence=cluster["avg_similarity"],
                        evidence={
                            "cluster_size": cluster["size"],
                            "avg_similarity": cluster["avg_similarity"],
                        },
                        detected_at=datetime.now().isoformat(),
                    )
                )
        return anomalies


class PatternAnalyzer:
    """Pattern-based anomaly detection."""

    def __init__(self, config: AnomalyConfig):
        self.config = config

    def find_repeated_phrases(
        self, documents: Dict[str, str], min_length: int = 20
    ) -> List[Dict[str, Any]]:
        """Find repeated phrases across documents."""
        phrase_docs: dict[str, set[str]] = defaultdict(set)
        for doc_name, text in documents.items():
            words = text.lower().split()
            for i in range(len(words) - min_length + 1):
                phrase = " ".join(words[i : i + min_length])
                phrase_docs[phrase].add(doc_name)

        repeated = []
        for phrase, doc_set in phrase_docs.items():
            if len(doc_set) >= 2:
                repeated.append(
                    {
                        "phrase": phrase[:100] + "..." if len(phrase) > 100 else phrase,
                        "documents": list(doc_set),
                        "document_count": len(doc_set),
                    }
                )
        return sorted(repeated, key=lambda x: x["document_count"], reverse=True)[:20]

    def detect_template_anomalies(
        self, documents: Dict[str, str], repeated_phrases: List[Dict]
    ) -> List[Anomaly]:
        """Detect template-based plagiarism."""
        anomalies = []
        doc_template_count: dict[str, int] = Counter()
        for phrase_info in repeated_phrases:
            for doc in phrase_info["documents"]:
                doc_template_count[doc] += 1

        for doc, count in doc_template_count.items():
            if count >= 3:
                anomalies.append(
                    Anomaly(
                        anomaly_id=f"TEMPLATE-{doc[:20]}",
                        anomaly_type=AnomalyType.TEMPLATE,
                        severity=AnomalySeverity.MEDIUM,
                        title=f"Template Usage: {doc}",
                        description=f"Document contains {count} repeated phrases found in other documents, suggesting template or shared source usage.",
                        affected_documents=[doc],
                        confidence=min(count / 10, 1.0),
                        evidence={"repeated_phrase_count": count},
                        detected_at=datetime.now().isoformat(),
                    )
                )
        return anomalies

    def detect_copy_patterns(self, documents: dict[str, str]) -> list[Anomaly]:
        """Detect exact copy patterns."""
        anomalies = []
        doc_names = list(documents.keys())
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                text_a = documents[doc_names[i]].lower()
                text_b = documents[doc_names[j]].lower()
                if (
                    len(text_a) < self.config.pattern_min_length
                    or len(text_b) < self.config.pattern_min_length
                ):
                    continue
                words_a = set(text_a.split())
                words_b = set(text_b.split())
                overlap = (
                    len(words_a & words_b) / min(len(words_a), len(words_b))
                    if min(len(words_a), len(words_b)) > 0
                    else 0
                )
                if overlap > self.config.template_threshold:
                    anomalies.append(
                        Anomaly(
                            anomaly_id=f"COPY-{doc_names[i][:10]}-{doc_names[j][:10]}",
                            anomaly_type=AnomalyType.PATTERN,
                            severity=AnomalySeverity.HIGH
                            if overlap > 0.9
                            else AnomalySeverity.MEDIUM,
                            title=f"Copy Pattern: {doc_names[i]} ↔ {doc_names[j]}",
                            description=f"Word overlap of {overlap:.1%} detected between documents, suggesting direct copying.",
                            affected_documents=[doc_names[i], doc_names[j]],
                            confidence=overlap,
                            evidence={"word_overlap": overlap},
                            detected_at=datetime.now().isoformat(),
                        )
                    )
        return anomalies


class AnomalyDetector:
    """
    Main anomaly detection engine.

    Combines statistical, cluster, and pattern analysis to detect
    plagiarism anomalies in document collections.
    """

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.stat_analyzer = StatisticalAnalyzer(self.config)
        self.cluster_analyzer = ClusterAnalyzer(self.config)
        self.pattern_analyzer = PatternAnalyzer(self.config)

    def detect(
        self,
        documents: dict[str, str],
        similarity_matrix: Optional[np.ndarray] = None,
        similarity_scores: Optional[list[float]] = None,
    ) -> AnomalyResult:
        """
        Run full anomaly detection pipeline.

        Args:
            documents: {doc_name: text_content}
            similarity_matrix: Optional similarity matrix between documents
            similarity_scores: Optional list of pairwise similarity scores

        Returns:
            AnomalyResult with all detected anomalies
        """
        start_time = datetime.now()
        anomalies = []
        doc_names = list(documents.keys())

        # Statistical analysis
        if self.config.enable_statistical and similarity_scores:
            z_anomalies = self.stat_analyzer.z_score_analysis(similarity_scores)
            for idx, z_score, severity in z_anomalies:
                anomalies.append(
                    Anomaly(
                        anomaly_id=f"STAT-Z-{idx:03d}",
                        anomaly_type=AnomalyType.STATISTICAL,
                        severity=severity,
                        title=f"Statistical Outlier (Z-score: {z_score:.2f})",
                        description=f"Similarity score at position {idx} is {z_score:.2f} standard deviations from the mean.",
                        affected_documents=[],
                        confidence=min(abs(z_score) / 5, 1.0),
                        evidence={"z_score": z_score, "position": idx},
                        detected_at=datetime.now().isoformat(),
                    )
                )
            dist_anomaly = self.stat_analyzer.detect_distribution_anomaly(
                similarity_scores
            )
            if dist_anomaly:
                anomalies.append(dist_anomaly)

        # Cluster analysis
        if self.config.enable_cluster and similarity_matrix is not None:
            clusters = self.cluster_analyzer.find_similarity_clusters(
                doc_names, similarity_matrix
            )
            collusion = self.cluster_analyzer.detect_collusion_clusters(clusters)
            anomalies.extend(collusion)

        # Pattern analysis
        if self.config.enable_pattern:
            repeated = self.pattern_analyzer.find_repeated_phrases(documents)
            template_anomalies = self.pattern_analyzer.detect_template_anomalies(
                documents, repeated
            )
            anomalies.extend(template_anomalies)
            copy_anomalies = self.pattern_analyzer.detect_copy_patterns(documents)
            anomalies.extend(copy_anomalies)

        anomalies.sort(
            key=lambda a: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}[
                    a.severity.value
                ],
                a.confidence,
            ),
            reverse=True,
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        summary = {
            "total_anomalies": len(anomalies),
            "by_type": dict(Counter(a.anomaly_type.value for a in anomalies)),
            "by_severity": dict(Counter(a.severity.value for a in anomalies)),
            "high_priority_count": sum(
                1 for a in anomalies if a.severity.value in ("high", "critical")
            ),
            "documents_analyzed": len(documents),
        }

        statistics = {}
        if similarity_scores:
            statistics = {
                "score_distribution": self.stat_analyzer.percentile_analysis(
                    similarity_scores
                ),
                "mean_score": float(np.mean(similarity_scores)),
                "std_score": float(np.std(similarity_scores)),
            }

        recommendations = self._generate_recommendations(anomalies, summary)

        return AnomalyResult(
            anomalies=anomalies,
            summary=summary,
            statistics=statistics,
            recommendations=recommendations,
            processing_time=processing_time,
        )

    def _generate_recommendations(
        self, anomalies: List[Anomaly], summary: Dict
    ) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        recs = []
        if summary.get("by_type", {}).get("collusion", 0) > 0:
            recs.append(
                "🔴 Collusion clusters detected. Consider interviewing affected students."
            )
        if summary.get("by_type", {}).get("template", 0) > 0:
            recs.append(
                "🟠 Template usage found. Consider revising assignment prompts."
            )
        if summary.get("by_type", {}).get("pattern", 0) > 0:
            recs.append("🟡 Copy patterns detected. Manual review recommended.")
        if summary.get("high_priority_count", 0) > 5:
            recs.append(
                "⚠️ High volume of anomalies. Consider expanding investigation scope."
            )
        if not recs:
            recs.append("✅ No significant anomalies detected. Continue monitoring.")
        return recs
