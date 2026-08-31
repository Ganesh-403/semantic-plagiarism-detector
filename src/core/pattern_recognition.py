"""Intelligent plagiarism pattern recognition and prediction engine.

Uses ML classifiers (RandomForest, GradientBoosting) trained on historical
incident data to detect recurring patterns, predict document risk scores,
and track technique evolution over time.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── ML availability ──────────────────────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Pattern type constants ───────────────────────────────────────────────
PATTERN_TYPES = (
    "copy_paste",
    "paraphrase",
    "source_sharing",
    "collaborative",
    "template_reuse",
)

RISK_LEVELS = ("Critical", "High", "Medium", "Low", "Negligible")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _make_pattern_id(doc_group: list[str]) -> str:
    sorted_docs = "||".join(sorted(doc_group))
    digest = hashlib.sha256(sorted_docs.encode("utf-8")).hexdigest()
    return f"PAT-{digest[:12].upper()}"


def _make_rec_id(rec_type: str, target: str) -> str:
    digest = hashlib.sha256(f"{rec_type}||{target}".encode("utf-8")).hexdigest()
    return f"REC-{digest[:12].upper()}"


# ═══════════════════════════════════════════════════════════════════════════
#  Pattern Detection Engine
# ═══════════════════════════════════════════════════════════════════════════


class PatternDetectionEngine:
    """Core engine for detecting recurring plagiarism patterns.

    Analyzes historical incident data to find:
    - Repeated document pairs across scans
    - Author clusters with recurring violations
    - Assignment hotspots with high plagiarism rates
    - Temporal patterns (submission timing clustering)
    """

    def __init__(self, repository: Any = None) -> None:
        self._repo = repository
        self._classifier: Any = None
        self._risk_model: Any = None
        self._scaler: Any = StandardScaler() if SKLEARN_AVAILABLE else None
        self._label_encoder: Any = LabelEncoder() if SKLEARN_AVAILABLE else None
        self._model_version = "1.0.0"

    # ── Public API ────────────────────────────────────────────────────────

    def detect_recurring_patterns(
        self,
        incidents: list[dict[str, Any]],
        min_occurrence: int = 2,
    ) -> list[dict[str, Any]]:
        """Mine incident records for recurring plagiarism patterns.

        Args:
            incidents: List of incident dicts from plagiarism_incidents table.
            min_occurrence: Minimum number of times a pattern must appear.

        Returns:
            List of detected pattern dicts ready for DB upsert.
        """
        patterns: list[dict[str, Any]] = []

        # ── 1. Repeat document pairs ──────────────────────────────────────
        pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for inc in incidents:
            doc_a = str(inc.get("document_a", "")).strip()
            doc_b = str(inc.get("document_b", "")).strip()
            if not doc_a or not doc_b or doc_a == doc_b:
                continue
            key = tuple(sorted((doc_a, doc_b)))
            pair_groups[key].append(inc)

        for (doc_a, doc_b), group in pair_groups.items():
            if len(group) < min_occurrence:
                continue
            scores = [float(g.get("similarity_score", 0)) for g in group]
            dates = [g.get("date_flagged", "") or g.get("last_seen", "") for g in group]
            avg_sim = float(np.mean(scores))
            first_seen = min(d for d in dates if d) or _utc_now_iso()
            last_seen = max(d for d in dates if d) or _utc_now_iso()

            confidence = self._compute_confidence(
                occurrence_count=len(group),
                avg_similarity=avg_sim,
                time_span_days=self._days_between(first_seen, last_seen),
            )
            severity = self._severity_from_score(avg_sim)
            ptype = self._classify_pair_type(avg_sim, group)

            pattern_id = _make_pattern_id([doc_a, doc_b])
            patterns.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_type": ptype,
                    "description": f"Recurring similarity between '{doc_a}' and '{doc_b}' "
                    f"detected {len(group)} times",
                    "document_group": [doc_a, doc_b],
                    "author_group": self._extract_authors(group),
                    "assignment_title": self._most_common_field(
                        group, "assignment_title"
                    ),
                    "class_section": self._most_common_field(group, "class_section"),
                    "avg_similarity": avg_sim,
                    "occurrence_count": len(group),
                    "confidence_score": confidence,
                    "severity": severity,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "status": "active",
                }
            )

        # ── 2. Author repeat-offender clusters ────────────────────────────
        author_incidents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for inc in incidents:
            for author_field in ("owner_a", "owner_b"):
                author = str(inc.get(author_field, "")).strip()
                if author:
                    author_incidents[author].append(inc)

        for author, author_incs in author_incidents.items():
            if len(author_incs) < min_occurrence:
                continue
            involved_docs = set()
            for inc in author_incs:
                involved_docs.add(str(inc.get("document_a", "")).strip())
                involved_docs.add(str(inc.get("document_b", "")).strip())
            involved_docs.discard("")
            scores = [float(inc.get("similarity_score", 0)) for inc in author_incs]
            avg_sim = float(np.mean(scores))
            dates = [
                inc.get("date_flagged", "") or inc.get("last_seen", "")
                for inc in author_incs
            ]
            first_seen = min(d for d in dates if d) or _utc_now_iso()
            last_seen = max(d for d in dates if d) or _utc_now_iso()

            confidence = self._compute_confidence(
                occurrence_count=len(author_incs),
                avg_similarity=avg_sim,
                time_span_days=self._days_between(first_seen, last_seen),
            )

            pattern_id = _make_pattern_id([author] + sorted(involved_docs))
            patterns.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_type": "collaborative",
                    "description": f"Author '{author}' involved in {len(author_incs)} plagiarism incidents",
                    "document_group": sorted(involved_docs),
                    "author_group": [author],
                    "assignment_title": None,
                    "class_section": None,
                    "avg_similarity": avg_sim,
                    "occurrence_count": len(author_incs),
                    "confidence_score": confidence,
                    "severity": self._severity_from_score(avg_sim),
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "status": "active",
                }
            )

        # ── 3. Assignment hotspots ────────────────────────────────────────
        assignment_incidents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for inc in incidents:
            for field in (
                "assignment_title_a",
                "assignment_title_b",
                "assignment_title",
            ):
                title = str(inc.get(field, "")).strip()
                if title:
                    assignment_incidents[title].append(inc)
                    break

        for assignment, assign_incs in assignment_incidents.items():
            if len(assign_incs) < min_occurrence:
                continue
            involved_docs = set()
            for inc in assign_incs:
                involved_docs.add(str(inc.get("document_a", "")).strip())
                involved_docs.add(str(inc.get("document_b", "")).strip())
            involved_docs.discard("")
            scores = [float(inc.get("similarity_score", 0)) for inc in assign_incs]
            avg_sim = float(np.mean(scores))
            dates = [
                inc.get("date_flagged", "") or inc.get("last_seen", "")
                for inc in assign_incs
            ]
            first_seen = min(d for d in dates if d) or _utc_now_iso()
            last_seen = max(d for d in dates if d) or _utc_now_iso()

            confidence = self._compute_confidence(
                occurrence_count=len(assign_incs),
                avg_similarity=avg_sim,
                time_span_days=self._days_between(first_seen, last_seen),
            )

            pattern_id = _make_pattern_id([assignment] + sorted(involved_docs))
            patterns.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_type": "source_sharing",
                    "description": f"Assignment '{assignment}' has {len(assign_incs)} plagiarism incidents",
                    "document_group": sorted(involved_docs),
                    "author_group": self._extract_authors(assign_incs),
                    "assignment_title": assignment,
                    "class_section": self._most_common_field(
                        assign_incs, "class_section"
                    ),
                    "avg_similarity": avg_sim,
                    "occurrence_count": len(assign_incs),
                    "confidence_score": confidence,
                    "severity": self._severity_from_score(avg_sim),
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "status": "active",
                }
            )

        # Persist if repository is available
        if self._repo:
            for pattern in patterns:
                try:
                    self._repo.upsert_pattern(**pattern)
                except Exception as exc:
                    logger.warning(
                        "Failed to persist pattern %s: %s", pattern["pattern_id"], exc
                    )

        return patterns

    def detect_technique_evolution(
        self,
        incidents: list[dict[str, Any]],
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Detect shifts in plagiarism technique distributions.

        Compares the current time window against the previous window
        to identify emerging or declining patterns.

        Returns:
            Dict with drift metrics and technique distribution changes.
        """
        now = datetime.now(timezone.utc)
        current_cutoff = now.isoformat()
        previous_cutoff = now.replace(day=max(1, now.day - window_days)).isoformat()

        current_window = [
            inc
            for inc in incidents
            if self._parse_date(inc.get("date_flagged", "")) >= previous_cutoff
        ]
        previous_window = [
            inc
            for inc in incidents
            if previous_cutoff
            > self._parse_date(inc.get("date_flagged", ""))
            >= (now.replace(day=max(1, now.day - 2 * window_days)).isoformat())
        ]

        if not current_window and not previous_window:
            return {
                "drift_score": 0.0,
                "current_distribution": {},
                "previous_distribution": {},
                "emerging_techniques": [],
                "declining_techniques": [],
                "alert": False,
            }

        current_dist = self._technique_distribution(current_window)
        previous_dist = self._technique_distribution(previous_window)

        drift_score = self._jensen_shannon_divergence(current_dist, previous_dist)

        emerging = [
            t
            for t, curr in current_dist.items()
            if curr > previous_dist.get(t, 0) * 1.2 and curr > 0.05
        ]
        declining = [
            t
            for t, prev in previous_dist.items()
            if prev > current_dist.get(t, 0) * 1.2 and prev > 0.05
        ]

        return {
            "drift_score": drift_score,
            "current_distribution": current_dist,
            "previous_distribution": previous_dist,
            "emerging_techniques": emerging,
            "declining_techniques": declining,
            "alert": drift_score > 0.15,
            "current_window_count": len(current_window),
            "previous_window_count": len(previous_window),
        }

    def score_document_risk(
        self,
        document_name: str,
        document_features: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict risk score for a document using ML or heuristic model.

        Args:
            document_name: Filename identifier.
            document_features: Dict with keys:
                - max_similarity: float (highest similarity to any other doc)
                - avg_similarity: float
                - author_incident_count: int (historical incidents for this author)
                - author_avg_similarity: float
                - assignment_incident_rate: float (fraction of submissions flagged)
                - document_length: int (character count)
                - submission_hour: int (0-23, hour of submission)
                - days_until_deadline: float or None
                - class_section_risk: float (fraction of class flagged)

        Returns:
            Dict with risk_score, risk_level, contributing_factors.
        """
        features = self._extract_risk_features(document_features)

        if SKLEARN_AVAILABLE and self._risk_model is not None:
            risk_score = float(self._risk_model.predict(features.reshape(1, -1))[0])
        else:
            risk_score = self._heuristic_risk_score(document_features)

        risk_score = max(0.0, min(1.0, risk_score))
        risk_level = self._score_to_risk_level(risk_score)
        factors = self._identify_risk_factors(document_features, risk_score)

        if self._repo:
            try:
                self._repo.upsert_risk_score(
                    document_name=document_name,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    contributing_factors=factors,
                    model_version=self._model_version,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist risk score for %s: %s", document_name, exc
                )

        return {
            "document_name": document_name,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "contributing_factors": factors,
            "model_version": self._model_version,
        }

    def evaluate_detection_accuracy(
        self,
        predicted_patterns: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Evaluate pattern detection accuracy against ground truth.

        Args:
            predicted_patterns: Patterns returned by detect_recurring_patterns.
            ground_truth: Labeled pattern data with 'pattern_id' and 'label'.

        Returns:
            Dict with precision, recall, f1, accuracy metrics.
        """
        pred_ids = {p["pattern_id"] for p in predicted_patterns}
        gt_map = {g["pattern_id"]: g.get("label", "true") for g in ground_truth}
        gt_ids = set(gt_map.keys())

        tp = len(pred_ids & gt_ids)
        fp = len(pred_ids - gt_ids)
        fn = len(gt_ids - pred_ids)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = tp / len(gt_ids) if gt_ids else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_predicted": len(pred_ids),
            "total_ground_truth": len(gt_ids),
        }

    def train_risk_model(
        self,
        training_data: pd.DataFrame,
        target_column: str = "risk_score",
    ) -> dict[str, float]:
        """Train the risk prediction model on historical data.

        Args:
            training_data: DataFrame with feature columns and target_column.
            target_column: Name of the target variable.

        Returns:
            Dict with training metrics.
        """
        if not SKLEARN_AVAILABLE or len(training_data) < 20:
            return {"status": "insufficient_data", "samples": len(training_data)}

        feature_cols = [c for c in training_data.columns if c != target_column]
        X = training_data[feature_cols].values
        y = (training_data[target_column] >= 0.5).astype(int).values

        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        self._risk_model = GradientBoostingClassifier(
            n_estimators=50, max_depth=4, random_state=42
        )
        self._risk_model.fit(X_scaled, y)

        predictions = self._risk_model.predict(X_scaled)
        accuracy = float(np.mean(predictions == y))

        return {
            "status": "trained",
            "samples": len(training_data),
            "accuracy": accuracy,
            "model_version": self._model_version,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        occurrence_count: int,
        avg_similarity: float,
        time_span_days: float,
    ) -> float:
        """Confidence = f(occurrence, similarity, recency)."""
        occurrence_factor = min(1.0, occurrence_count / 10)
        similarity_factor = avg_similarity
        recency_factor = (
            1.0 / (1.0 + time_span_days / 30) if time_span_days > 0 else 1.0
        )
        return min(
            1.0,
            (0.4 * occurrence_factor + 0.4 * similarity_factor + 0.2 * recency_factor),
        )

    @staticmethod
    def _severity_from_score(score: float) -> str:
        if score >= 0.85:
            return "Critical"
        if score >= 0.70:
            return "High"
        if score >= 0.45:
            return "Medium"
        return "Low"

    @staticmethod
    def _score_to_risk_level(score: float) -> str:
        if score >= 0.8:
            return "Critical"
        if score >= 0.6:
            return "High"
        if score >= 0.4:
            return "Medium"
        if score >= 0.2:
            return "Low"
        return "Negligible"

    @staticmethod
    def _days_between(date_a: str, date_b: str) -> float:
        try:
            dt_a = datetime.fromisoformat(date_a.replace("Z", "+00:00"))
            dt_b = datetime.fromisoformat(date_b.replace("Z", "+00:00"))
            return abs((dt_b - dt_a).total_seconds()) / 86400
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _parse_date(date_str: str) -> str:
        if not date_str:
            return ""
        return date_str

    @staticmethod
    def _extract_authors(incidents: list[dict[str, Any]]) -> list[str]:
        authors = set()
        for inc in incidents:
            for field in ("owner_a", "owner_b"):
                val = str(inc.get(field, "")).strip()
                if val:
                    authors.add(val)
        return sorted(authors)

    @staticmethod
    def _most_common_field(incidents: list[dict[str, Any]], field: str) -> str | None:
        values = [
            str(inc.get(field, "")).strip() for inc in incidents if inc.get(field)
        ]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    @staticmethod
    def _classify_pair_type(avg_similarity: float, group: list[dict[str, Any]]) -> str:
        """Heuristic classification when ML model is unavailable."""
        if avg_similarity >= 0.90:
            return "copy_paste"
        if avg_similarity >= 0.70:
            return "paraphrase"
        if avg_similarity >= 0.50:
            return "source_sharing"
        return "template_reuse"

    def _technique_distribution(
        self, incidents: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute the proportional distribution of plagiarism techniques."""
        if not incidents:
            return {t: 0.0 for t in PATTERN_TYPES}

        type_counts: dict[str, int] = Counter()
        for inc in incidents:
            score = float(inc.get("similarity_score", 0))
            if score >= 0.90:
                type_counts["copy_paste"] += 1
            elif score >= 0.70:
                type_counts["paraphrase"] += 1
            elif score >= 0.50:
                type_counts["source_sharing"] += 1
            else:
                type_counts["template_reuse"] += 1
        total = sum(type_counts.values())
        return {t: type_counts.get(t, 0) / total for t in PATTERN_TYPES}

    @staticmethod
    def _jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
        """Symmetric divergence between two probability distributions."""
        all_keys = set(p.keys()) | set(q.keys())
        if not all_keys:
            return 0.0

        p_vec = np.array([p.get(k, 0.0) for k in all_keys])
        q_vec = np.array([q.get(k, 0.0) for k in all_keys])

        # Normalize
        p_sum = p_vec.sum()
        q_sum = q_vec.sum()
        if p_sum > 0:
            p_vec = p_vec / p_sum
        if q_sum > 0:
            q_vec = q_vec / q_sum

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_vec = p_vec + eps
        q_vec = q_vec + eps
        p_vec = p_vec / p_vec.sum()
        q_vec = q_vec / q_vec.sum()

        m = 0.5 * (p_vec + q_vec)
        js = 0.5 * (
            float(np.sum(p_vec * np.log(p_vec / m)))
            + float(np.sum(q_vec * np.log(q_vec / m)))
        )
        return js

    @staticmethod
    def _extract_risk_features(features: dict[str, Any]) -> np.ndarray:
        """Convert feature dict to numpy array for ML model."""
        return np.array(
            [
                features.get("max_similarity", 0.0),
                features.get("avg_similarity", 0.0),
                min(1.0, features.get("author_incident_count", 0) / 10),
                features.get("author_avg_similarity", 0.0),
                features.get("assignment_incident_rate", 0.0),
                min(1.0, features.get("document_length", 0) / 50000),
                features.get("submission_hour", 12) / 23,
                (features.get("days_until_deadline", 30) or 30) / 30,
                features.get("class_section_risk", 0.0),
            ]
        )

    @staticmethod
    def _heuristic_risk_score(features: dict[str, Any]) -> float:
        """Fallback heuristic when ML model is not trained."""
        weights = {
            "max_similarity": 0.30,
            "author_incident_rate": 0.25,
            "assignment_incident_rate": 0.15,
            "author_avg_similarity": 0.15,
            "class_section_risk": 0.10,
            "recency_bonus": 0.05,
        }

        author_inc_rate = min(1.0, features.get("author_incident_count", 0) / 10)
        recency_bonus = 1.0 if features.get("submission_hour", 12) >= 22 else 0.0

        score = (
            weights["max_similarity"] * features.get("max_similarity", 0)
            + weights["author_incident_rate"] * author_inc_rate
            + weights["assignment_incident_rate"]
            * features.get("assignment_incident_rate", 0)
            + weights["author_avg_similarity"]
            * features.get("author_avg_similarity", 0)
            + weights["class_section_risk"] * features.get("class_section_risk", 0)
            + weights["recency_bonus"] * recency_bonus
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _identify_risk_factors(
        features: dict[str, Any], risk_score: float
    ) -> list[str]:
        factors = []
        if features.get("max_similarity", 0) >= 0.85:
            factors.append("High similarity to existing document")
        if features.get("author_incident_count", 0) >= 3:
            factors.append(
                f"Author has {features['author_incident_count']} prior incidents"
            )
        if features.get("assignment_incident_rate", 0) >= 0.2:
            factors.append("High incident rate for this assignment")
        if features.get("submission_hour", 12) >= 22:
            factors.append("Late-night submission")
        if (
            features.get("days_until_deadline") is not None
            and features["days_until_deadline"] <= 1
        ):
            factors.append("Submitted near deadline")
        if features.get("class_section_risk", 0) >= 0.3:
            factors.append("High-risk class section")
        if not factors:
            factors.append("Low overall risk indicators")
        return factors
