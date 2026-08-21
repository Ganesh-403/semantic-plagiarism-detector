"""Proactive recommendation engine for plagiarism pattern response.

Generates actionable intervention recommendations based on detected patterns,
risk scores, and temporal trends. Integrates with the smart notification
system to deliver alerts.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _make_rec_id(rec_type: str, target: str) -> str:
    digest = hashlib.sha256(f"{rec_type}||{target}".encode("utf-8")).hexdigest()
    return f"REC-{digest[:12].upper()}"


# ── Priority constants ───────────────────────────────────────────────────
PRIORITY_CRITICAL = 1
PRIORITY_HIGH = 2
PRIORITY_MEDIUM = 3
PRIORITY_LOW = 4


class RecommendationEngine:
    """Generates proactive intervention recommendations.

    Rules-based + ML-informed recommendation system that analyzes
    detected patterns and risk scores to produce actionable items.
    """

    def __init__(self, repository: Any = None) -> None:
        self._repo = repository

    def generate_recommendations(
        self,
        patterns: list[dict[str, Any]],
        risk_scores: list[dict[str, Any]],
        trend_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate recommendations from patterns and risk data.

        Args:
            patterns: Detected plagiarism patterns.
            risk_scores: Document risk score results.
            trend_data: Optional trend analysis output.

        Returns:
            List of recommendation dicts ready for DB persistence.
        """
        recommendations: list[dict[str, Any]] = []

        # ── Author-based recommendations ──────────────────────────────────
        recommendations.extend(self._author_recommendations(patterns))

        # ── Assignment-based recommendations ──────────────────────────────
        recommendations.extend(self._assignment_recommendations(patterns))

        # ── Pattern-based recommendations ─────────────────────────────────
        recommendations.extend(self._pattern_recommendations(patterns))

        # ── Risk-based recommendations ────────────────────────────────────
        recommendations.extend(self._risk_recommendations(risk_scores))

        # ── Trend-based recommendations ───────────────────────────────────
        if trend_data:
            recommendations.extend(self._trend_recommendations(trend_data))

        # Persist if repository is available
        if self._repo:
            for rec in recommendations:
                try:
                    self._repo.create_recommendation(**rec)
                except Exception as exc:
                    logger.warning(
                        "Failed to persist recommendation %s: %s",
                        rec.get("recommendation_id"),
                        exc,
                    )

        return recommendations

    def get_pending_actions(self) -> list[dict[str, Any]]:
        """Retrieve unprocessed recommendations."""
        if not self._repo:
            return []
        return self._repo.get_recommendations(status="pending")

    def acknowledge_recommendation(self, recommendation_id: str) -> bool:
        if not self._repo:
            return False
        return self._repo.update_recommendation_status(
            recommendation_id, "acknowledged"
        )

    def dismiss_recommendation(self, recommendation_id: str) -> bool:
        if not self._repo:
            return False
        return self._repo.update_recommendation_status(recommendation_id, "dismissed")

    # ── Rule implementations ──────────────────────────────────────────────

    def _author_recommendations(
        self, patterns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        seen_authors: set[str] = set()

        for pattern in patterns:
            authors = pattern.get("author_group") or []
            for author in authors:
                if author in seen_authors:
                    continue
                seen_authors.add(author)

                occ = pattern.get("occurrence_count", 0)
                if occ >= 5:
                    recs.append(
                        {
                            "recommendation_id": _make_rec_id(
                                "escalate_author", author
                            ),
                            "pattern_id": pattern.get("pattern_id"),
                            "recommendation_type": "escalate",
                            "priority": PRIORITY_CRITICAL,
                            "target": author,
                            "message": f"Author '{author}' has {occ} plagiarism incidents. "
                            "Immediate escalation to academic integrity office recommended.",
                            "action_items": [
                                "Notify academic integrity office",
                                "Schedule investigation meeting",
                                "Review all submissions from this author",
                                "Consider temporary submission hold",
                            ],
                        }
                    )
                elif occ >= 3:
                    recs.append(
                        {
                            "recommendation_id": _make_rec_id("monitor_author", author),
                            "pattern_id": pattern.get("pattern_id"),
                            "recommendation_type": "monitor_author",
                            "priority": PRIORITY_HIGH,
                            "target": author,
                            "message": f"Author '{author}' has {occ} incidents. "
                            "Enhanced monitoring recommended.",
                            "action_items": [
                                "Flag future submissions for priority review",
                                "Review submission patterns and timing",
                                "Consider academic counseling referral",
                            ],
                        }
                    )

        return recs

    def _assignment_recommendations(
        self, patterns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        seen_assignments: set[str] = set()

        for pattern in patterns:
            assignment = pattern.get("assignment_title")
            if not assignment or assignment in seen_assignments:
                continue
            seen_assignments.add(assignment)

            occ = pattern.get("occurrence_count", 0)
            docs = pattern.get("document_group", [])
            doc_count = len(docs) if isinstance(docs, list) else 0

            if doc_count > 0 and occ / max(doc_count, 1) > 0.2:
                recs.append(
                    {
                        "recommendation_id": _make_rec_id(
                            "redesign_assignment", assignment
                        ),
                        "pattern_id": pattern.get("pattern_id"),
                        "recommendation_type": "redesign_assignment",
                        "priority": PRIORITY_HIGH,
                        "target": assignment,
                        "message": f"Assignment '{assignment}' has a high plagiarism rate "
                        f"({occ} incidents across {doc_count} documents). "
                        "Consider redesigning the assignment.",
                        "action_items": [
                            "Review assignment prompt for ambiguity",
                            "Add unique sub-topics per student",
                            "Implement staged submission deadlines",
                            "Require source annotations",
                        ],
                    }
                )

        return recs

    def _pattern_recommendations(
        self, patterns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []

        for pattern in patterns:
            ptype = pattern.get("pattern_type", "")
            confidence = pattern.get("confidence_score", 0)
            severity = pattern.get("severity", "Low")

            if confidence < 0.6:
                continue

            if ptype == "collaborative" and severity in ("Critical", "High"):
                recs.append(
                    {
                        "recommendation_id": _make_rec_id(
                            "investigate_collaboration", pattern.get("pattern_id", "")
                        ),
                        "pattern_id": pattern.get("pattern_id"),
                        "recommendation_type": "policy_review",
                        "priority": PRIORITY_HIGH,
                        "target": pattern.get("pattern_id", ""),
                        "message": f"Collaborative plagiarism pattern detected with "
                        f"{confidence:.0%} confidence. Possible collusion between authors.",
                        "action_items": [
                            "Review submission timestamps for coordination",
                            "Compare document metadata and authorship",
                            "Interview involved students",
                            "Review collaboration policy",
                        ],
                    }
                )

            if ptype == "copy_paste" and severity in ("Critical", "High"):
                recs.append(
                    {
                        "recommendation_id": _make_rec_id(
                            "strengthen_detection", pattern.get("pattern_id", "")
                        ),
                        "pattern_id": pattern.get("pattern_id"),
                        "recommendation_type": "strengthen_detection",
                        "priority": PRIORITY_MEDIUM,
                        "target": pattern.get("pattern_id", ""),
                        "message": "Copy-paste plagiarism detected. "
                        "Consider lowering similarity threshold for this assignment type.",
                        "action_items": [
                            "Lower detection threshold by 5%",
                            "Enable strict text-matching mode",
                            "Review original source materials",
                        ],
                    }
                )

            if ptype == "template_reuse":
                recs.append(
                    {
                        "recommendation_id": _make_rec_id(
                            "update_template", pattern.get("pattern_id", "")
                        ),
                        "pattern_id": pattern.get("pattern_id"),
                        "recommendation_type": "stagger_deadlines",
                        "priority": PRIORITY_LOW,
                        "target": pattern.get("pattern_id", ""),
                        "message": "Template reuse pattern detected. "
                        "Consider providing unique templates or varying assignment structure.",
                        "action_items": [
                            "Distribute unique assignment templates",
                            "Vary question phrasing across sections",
                            "Add randomized component requirements",
                        ],
                    }
                )

        return recs

    def _risk_recommendations(
        self, risk_scores: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []

        critical_docs = [
            r for r in risk_scores if r.get("risk_level") in ("Critical", "High")
        ]

        if len(critical_docs) >= 5:
            recs.append(
                {
                    "recommendation_id": _make_rec_id("batch_review", "critical_batch"),
                    "pattern_id": None,
                    "recommendation_type": "escalate",
                    "priority": PRIORITY_CRITICAL,
                    "target": "batch",
                    "message": f"{len(critical_docs)} documents flagged as critical/high risk. "
                    "Batch review recommended.",
                    "action_items": [
                        "Prioritize review of critical-risk documents",
                        "Assign review team for batch processing",
                        "Generate detailed risk report for stakeholders",
                    ],
                }
            )

        for doc in critical_docs[:3]:
            doc_name = doc.get("document_name", "")
            risk_score = doc.get("risk_score", 0)
            recs.append(
                {
                    "recommendation_id": _make_rec_id("review_document", doc_name),
                    "pattern_id": None,
                    "recommendation_type": "monitor_author",
                    "priority": PRIORITY_HIGH if risk_score >= 0.8 else PRIORITY_MEDIUM,
                    "target": doc_name,
                    "message": f"Document '{doc_name}' has risk score {risk_score:.2%}. "
                    "Priority review recommended.",
                    "action_items": [
                        "Review document against top similar matches",
                        "Check submission metadata and timing",
                        "Contact author for clarification if needed",
                    ],
                }
            )

        return recs

    def _trend_recommendations(
        self, trend_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []

        direction = trend_data.get("drift_direction", "stable")
        drift_score = trend_data.get("drift_score", 0)

        if direction == "increasing" and drift_score > 0.1:
            recs.append(
                {
                    "recommendation_id": _make_rec_id("escalate_trend", "increasing"),
                    "pattern_id": None,
                    "recommendation_type": "escalate",
                    "priority": PRIORITY_HIGH,
                    "target": "department",
                    "message": f"Plagiarism trend is increasing (drift: {drift_score:.3f}). "
                    "Departmental review recommended.",
                    "action_items": [
                        "Prepare trend report for department head",
                        "Review current prevention measures",
                        "Consider additional training sessions",
                        "Evaluate detection threshold adequacy",
                    ],
                }
            )

        emerging = trend_data.get("emerging_techniques", [])
        if emerging:
            recs.append(
                {
                    "recommendation_id": _make_rec_id(
                        "new_technique", "|".join(emerging[:3])
                    ),
                    "pattern_id": None,
                    "recommendation_type": "strengthen_detection",
                    "priority": PRIORITY_MEDIUM,
                    "target": "detection_system",
                    "message": f"Emerging plagiarism techniques detected: {', '.join(emerging)}. "
                    "Detection system review recommended.",
                    "action_items": [
                        "Review and update detection parameters",
                        "Test detection accuracy on new technique samples",
                        "Update training data for ML models",
                    ],
                }
            )

        return recs
