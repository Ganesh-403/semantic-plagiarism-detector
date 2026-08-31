# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: ADVANCED DOCUMENT VERSION CONTROL & CHANGE TRACKING
# ───────────────────────────────────────────────────────────────────────────────

import difflib
import hashlib
import json
import zlib
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Document Version Class ──────────────────────────────────────────────────
class DocumentVersion:
    """Represents a single version of a document"""

    def __init__(self, content: str, doc_name: str, version_id: int):
        self.version_id = version_id
        self.doc_name = doc_name
        self.content = content
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.timestamp = datetime.now()
        self.size = len(content)
        self.word_count = len(content.split())
        self.similarity_score = None
        self.parent_version = None
        self.change_summary = None
        self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert version to dictionary for serialization"""
        return {
            "version_id": self.version_id,
            "doc_name": self.doc_name,
            "content": self.content,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp.isoformat(),
            "size": self.size,
            "word_count": self.word_count,
            "similarity_score": self.similarity_score,
            "parent_version": self.parent_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentVersion":
        """Create version from dictionary"""
        version = cls(data["content"], data["doc_name"], data["version_id"])
        version.content_hash = data["content_hash"]
        version.timestamp = datetime.fromisoformat(data["timestamp"])
        version.size = data["size"]
        version.word_count = data["word_count"]
        version.similarity_score = data.get("similarity_score")
        version.parent_version = data.get("parent_version")
        version.metadata = data.get("metadata", {})
        return version

    def compress(self) -> bytes:
        """Compress version content for storage"""
        return zlib.compress(self.content.encode("utf-8"))

    @staticmethod
    def decompress(data: bytes) -> str:
        """Decompress version content"""
        return zlib.decompress(data).decode("utf-8")


# ── Version Manager ────────────────────────────────────────────────────────
class VersionManager:
    """Manages document versions and history"""

    def __init__(self):
        self.versions = defaultdict(list)  # doc_name -> list of DocumentVersion
        self.current_versions = {}  # doc_name -> version_id
        self.version_index = {}  # doc_name -> {hash: version_id}
        self.metadata_store = {}  # doc_name -> metadata

    def add_version(
        self,
        doc_name: str,
        content: str,
        parent_version: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a new version of a document"""
        version_id = len(self.versions[doc_name]) + 1

        # Check if content already exists
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if doc_name in self.version_index:
            if content_hash in self.version_index[doc_name]:
                return self.version_index[doc_name][content_hash]

        # Create version
        version = DocumentVersion(content, doc_name, version_id)
        version.parent_version = parent_version or self.current_versions.get(doc_name)
        if metadata:
            version.metadata.update(metadata)

        # Store version
        self.versions[doc_name].append(version)
        self.current_versions[doc_name] = version_id

        # Update index
        if doc_name not in self.version_index:
            self.version_index[doc_name] = {}
        self.version_index[doc_name][content_hash] = version_id

        # Detect changes from previous version
        if parent_version or len(self.versions[doc_name]) > 1:
            prev_version = self.get_version(doc_name, version.parent_version)
            if prev_version:
                tracker = ChangeTracker()
                version.change_summary = tracker.detect_changes(
                    prev_version.content, content
                )

        return version_id

    def get_version(self, doc_name: str, version_id: int) -> Optional[DocumentVersion]:
        """Get a specific version of a document"""
        if doc_name in self.versions and version_id <= len(self.versions[doc_name]):
            return self.versions[doc_name][version_id - 1]
        return None

    def get_current_version(self, doc_name: str) -> Optional[DocumentVersion]:
        """Get the current version of a document"""
        version_id = self.current_versions.get(doc_name)
        if version_id:
            return self.get_version(doc_name, version_id)
        return None

    def get_version_history(self, doc_name: str) -> List[DocumentVersion]:
        """Get all versions of a document"""
        return self.versions.get(doc_name, [])

    def get_version_count(self, doc_name: str) -> int:
        """Get number of versions for a document"""
        return len(self.versions.get(doc_name, []))

    def get_all_documents(self) -> List[str]:
        """Get all document names with versions"""
        return list(self.versions.keys())

    def get_version_timeline(self, doc_name: str) -> pd.DataFrame:
        """Get version timeline as DataFrame"""
        versions = self.get_version_history(doc_name)
        if not versions:
            return pd.DataFrame()

        data = []
        for v in versions:
            data.append(
                {
                    "Version": f"v{v.version_id}",
                    "Timestamp": v.timestamp,
                    "Size (bytes)": v.size,
                    "Word Count": v.word_count,
                    "Similarity Score": v.similarity_score if v.similarity_score else 0,
                }
            )
        return pd.DataFrame(data)

    def delete_version(self, doc_name: str, version_id: int) -> bool:
        """Delete a specific version"""
        if doc_name in self.versions:
            versions = self.versions[doc_name]
            if version_id <= len(versions):
                # Remove version
                del versions[version_id - 1]
                # Update version IDs
                for i, v in enumerate(versions, 1):
                    v.version_id = i
                # Update current version if needed
                if self.current_versions.get(doc_name) == version_id:
                    self.current_versions[doc_name] = (
                        len(versions) if versions else None
                    )
                return True
        return False

    def restore_version(self, doc_name: str, version_id: int) -> Optional[str]:
        """Restore a previous version as current version"""
        version = self.get_version(doc_name, version_id)
        if version:
            new_version_id = self.add_version(
                doc_name,
                version.content,
                parent_version=version_id,
                metadata={"restored_from": version_id},
            )
            return version.content
        return None

    def export_history(self, doc_name: str) -> Dict:
        """Export version history as dictionary"""
        versions = self.get_version_history(doc_name)
        return {
            "doc_name": doc_name,
            "total_versions": len(versions),
            "current_version": self.current_versions.get(doc_name),
            "versions": [v.to_dict() for v in versions],
        }

    def import_history(self, data: Dict) -> bool:
        """Import version history from dictionary"""
        try:
            doc_name = data["doc_name"]
            for version_data in data["versions"]:
                version = DocumentVersion.from_dict(version_data)
                self.versions[doc_name].append(version)
                if version_data["version_id"] == data["current_version"]:
                    self.current_versions[doc_name] = version.version_id
            return True
        except Exception:
            return False


# ── Change Tracker ──────────────────────────────────────────────────────────
class ChangeTracker:
    """Detects and tracks changes between document versions"""

    def __init__(self):
        self.change_history = defaultdict(list)
        self.patterns = []

    def detect_changes(self, old_content: str, new_content: str) -> Dict:
        """Detect and classify changes between two document versions"""
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        diff = difflib.SequenceMatcher(None, old_lines, new_lines)
        opcodes = diff.get_opcodes()

        changes = {
            "additions": [],
            "deletions": [],
            "replacements": [],
            "total_added": 0,
            "total_deleted": 0,
            "total_replaced": 0,
            "similarity_ratio": diff.ratio(),
            "edit_distance": self._calculate_edit_distance(old_content, new_content),
            "change_percentage": 0,
        }

        for opcode, i1, i2, j1, j2 in opcodes:
            if opcode == "insert":
                changes["additions"].extend(new_lines[j1:j2])
                changes["total_added"] += j2 - j1
            elif opcode == "delete":
                changes["deletions"].extend(old_lines[i1:i2])
                changes["total_deleted"] += i2 - i1
            elif opcode == "replace":
                changes["replacements"].append(
                    {
                        "old": old_lines[i1:i2],
                        "new": new_lines[j1:j2],
                        "old_count": i2 - i1,
                        "new_count": j2 - j1,
                    }
                )
                changes["total_replaced"] += max(i2 - i1, j2 - j1)

        # Calculate change percentage
        total_lines = max(len(old_lines), len(new_lines))
        if total_lines > 0:
            changes["change_percentage"] = (
                (
                    changes["total_added"]
                    + changes["total_deleted"]
                    + changes["total_replaced"]
                )
                / total_lines
            ) * 100

        return changes

    def _calculate_edit_distance(self, old_content: str, new_content: str) -> int:
        """Calculate Levenshtein edit distance"""
        try:
            import nltk

            return nltk.edit_distance(old_content, new_content)
        except ImportError:
            # Fallback to simple diff length
            return len(difflib.ndiff(old_content, new_content))

    def detect_suspicious_patterns(self, changes: Dict) -> List[str]:
        """Detect suspicious editing patterns"""
        warnings = []

        # Pattern 1: Large additions in single edit
        if changes["total_added"] > 1000:
            warnings.append("Large text addition detected (>1000 chars)")

        # Pattern 2: Massive deletions
        if changes["total_deleted"] > 1000:
            warnings.append("Massive text deletion detected (>1000 chars)")

        # Pattern 3: High change ratio
        total_changes = changes["total_added"] + changes["total_deleted"]
        if total_changes > 500 and changes["similarity_ratio"] < 0.5:
            warnings.append("Major content replacement detected (>50% changes)")

        # Pattern 4: Multiple replacements
        if len(changes["replacements"]) > 3:
            warnings.append("Multiple copy-paste replacements detected")

        # Pattern 5: Very low similarity with high additions
        if changes["similarity_ratio"] < 0.3 and changes["total_added"] > 500:
            warnings.append("Document appears to be replaced with new content")

        return warnings

    def get_change_frequency(self, versions: List[DocumentVersion]) -> Dict:
        """Analyze change frequency across versions"""
        if len(versions) < 2:
            return {"status": "insufficient_data"}

        changes = []
        for i in range(len(versions) - 1):
            change = self.detect_changes(versions[i].content, versions[i + 1].content)
            changes.append(
                {
                    "from_version": versions[i].version_id,
                    "to_version": versions[i + 1].version_id,
                    "similarity_ratio": change["similarity_ratio"],
                    "change_percentage": change["change_percentage"],
                    "total_added": change["total_added"],
                    "total_deleted": change["total_deleted"],
                    "timestamp": versions[i + 1].timestamp,
                }
            )

        # Calculate statistics
        if changes:
            avg_change = np.mean([c["change_percentage"] for c in changes])
            avg_similarity = np.mean([c["similarity_ratio"] for c in changes])

            return {
                "total_changes": len(changes),
                "average_change_percentage": avg_change,
                "average_similarity_ratio": avg_similarity,
                "max_change_percentage": max([c["change_percentage"] for c in changes]),
                "min_change_percentage": min([c["change_percentage"] for c in changes]),
                "changes": changes,
            }

        return {"status": "no_changes"}


# ── Version Diff Generator ──────────────────────────────────────────────────
class VersionDiffGenerator:
    """Generates HTML and visual diffs for version comparison"""

    def __init__(self):
        self.html_template = """
        <div class="diff-container" style="font-family: monospace; font-size: 12px;">
            <h4 style="margin: 10px 0;">Version {old_ver} → {new_ver}</h4>
            <div class="diff-content" style="background: #f5f5f5; padding: 10px; border-radius: 4px;">
                {diff_lines}
            </div>
        </div>
        """
        self.css_styles = """
        <style>
        .diff-container { font-family: monospace; font-size: 12px; }
        .equal { color: #666; padding: 2px 5px; }
        .insert { color: #00aa00; background: #e6ffe6; padding: 2px 5px; }
        .delete { color: #cc0000; background: #ffe6e6; padding: 2px 5px; }
        .replace-old { color: #cc0000; background: #ffe6e6; padding: 2px 5px; text-decoration: line-through; }
        .replace-new { color: #00aa00; background: #e6ffe6; padding: 2px 5px; }
        .diff-line { display: block; margin: 1px 0; }
        </style>
        """

    def generate_html_diff(
        self, old_content: str, new_content: str, old_ver: int = 1, new_ver: int = 2
    ) -> str:
        """Generate HTML diff visualization"""
        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines() if new_content else []

        # Handle empty content
        if not old_lines and not new_lines:
            return "<div>No content to compare</div>"
        if not old_lines:
            return "<div>New document created (no previous version)</div>"
        if not new_lines:
            return "<div>Document deleted (no content remaining)</div>"

        diff = difflib.SequenceMatcher(None, old_lines, new_lines)
        opcodes = diff.get_opcodes()

        diff_lines = []
        for opcode, i1, i2, j1, j2 in opcodes:
            if opcode == "equal":
                for line in old_lines[i1:i2]:
                    diff_lines.append(f'<div class="diff-line equal">{line}</div>')
            elif opcode == "insert":
                for line in new_lines[j1:j2]:
                    diff_lines.append(f'<div class="diff-line insert">+ {line}</div>')
            elif opcode == "delete":
                for line in old_lines[i1:i2]:
                    diff_lines.append(f'<div class="diff-line delete">- {line}</div>')
            elif opcode == "replace":
                # Show old lines as deleted
                for line in old_lines[i1:i2]:
                    diff_lines.append(
                        f'<div class="diff-line replace-old">- {line}</div>'
                    )
                # Show new lines as added
                for line in new_lines[j1:j2]:
                    diff_lines.append(
                        f'<div class="diff-line replace-new">+ {line}</div>'
                    )

        html = self.css_styles + self.html_template.format(
            old_ver=old_ver, new_ver=new_ver, diff_lines="\n".join(diff_lines)
        )
        return html

    def generate_unified_diff(self, old_content: str, new_content: str) -> str:
        """Generate unified diff format"""
        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines() if new_content else []

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old_version",
            tofile="new_version",
            lineterm="",
        )
        return "\n".join(diff)

    def generate_context_diff(
        self, old_content: str, new_content: str, context_lines: int = 3
    ) -> str:
        """Generate context diff format"""
        old_lines = old_content.splitlines() if old_content else []
        new_lines = new_content.splitlines() if new_content else []

        diff = difflib.context_diff(
            old_lines,
            new_lines,
            fromfile="old_version",
            tofile="new_version",
            lineterm="",
            n=context_lines,
        )
        return "\n".join(diff)


# ── Plagiarism Evolution Analyzer ──────────────────────────────────────────
class PlagiarismEvolutionAnalyzer:
    """Analyzes how plagiarism scores evolve across versions"""

    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager
        self.similarity_history = defaultdict(list)
        self.evolution_metrics = {}

    def analyze_plagiarism_evolution(self, doc_name: str) -> Dict:
        """Analyze how plagiarism scores evolve across versions"""
        versions = self.version_manager.get_version_history(doc_name)
        if len(versions) < 2:
            return {"status": "insufficient_data"}

        analysis = {
            "doc_name": doc_name,
            "total_versions": len(versions),
            "versions": [],
            "trend": "stable",
            "significant_changes": [],
            "metrics": {
                "avg_similarity": 0,
                "max_similarity": 0,
                "min_similarity": 0,
                "std_similarity": 0,
                "change_velocity": 0,
            },
        }

        similarities = []
        for i in range(len(versions) - 1):
            old_ver = versions[i]
            new_ver = versions[i + 1]

            old_sim = old_ver.similarity_score or 0
            new_sim = new_ver.similarity_score or 0

            change_data = {
                "from_version": old_ver.version_id,
                "to_version": new_ver.version_id,
                "old_similarity": old_sim,
                "new_similarity": new_sim,
                "similarity_change": new_sim - old_sim,
                "change_percentage": ((new_sim - old_sim) / old_sim * 100)
                if old_sim > 0
                else 0,
                "timestamp": new_ver.timestamp,
            }
            analysis["versions"].append(change_data)
            similarities.append(new_sim)

            # Detect significant changes
            if abs(change_data["similarity_change"]) > 0.2:  # 20% change threshold
                analysis["significant_changes"].append(
                    {**change_data, "significance": "high"}
                )

        # Calculate metrics
        if similarities:
            analysis["metrics"]["avg_similarity"] = np.mean(similarities)
            analysis["metrics"]["max_similarity"] = max(similarities)
            analysis["metrics"]["min_similarity"] = min(similarities)
            analysis["metrics"]["std_similarity"] = np.std(similarities)

            # Calculate change velocity
            if len(analysis["versions"]) > 1:
                changes = [v["similarity_change"] for v in analysis["versions"]]
                analysis["metrics"]["change_velocity"] = np.mean(np.abs(changes))

        # Determine trend
        if analysis["versions"]:
            changes = [v["similarity_change"] for v in analysis["versions"]]
            avg_change = np.mean(changes)
            if avg_change > 0.1:
                analysis["trend"] = "increasing"
            elif avg_change < -0.1:
                analysis["trend"] = "decreasing"
            else:
                analysis["trend"] = "stable"

        # Calculate risk score
        risk_factors = 0
        if analysis["trend"] == "increasing":
            risk_factors += 1
        if analysis["significant_changes"]:
            risk_factors += len(analysis["significant_changes"])
        if analysis["metrics"]["change_velocity"] > 0.1:
            risk_factors += 1
        if analysis["metrics"]["std_similarity"] > 0.15:
            risk_factors += 1

        analysis["risk_score"] = min(
            risk_factors / 5 * 100, 100
        )  # Normalize to percentage

        return analysis

    def generate_evolution_plot(self, doc_name: str) -> go.Figure:
        """Generate plot of plagiarism evolution"""
        analysis = self.analyze_plagiarism_evolution(doc_name)
        if analysis.get("status") == "insufficient_data":
            return None

        fig = go.Figure()

        # Add similarity scores
        versions_data = []
        versions = self.version_manager.get_version_history(doc_name)
        for v in versions:
            if v.similarity_score is not None:
                versions_data.append(
                    {
                        "version": v.version_id,
                        "similarity": v.similarity_score,
                        "timestamp": v.timestamp,
                    }
                )

        if versions_data:
            df = pd.DataFrame(versions_data)

            # Main similarity trace
            fig.add_trace(
                go.Scatter(
                    x=df["version"],
                    y=df["similarity"],
                    mode="lines+markers",
                    name="Similarity Score",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(size=10, symbol="circle"),
                    hovertemplate="Version %{x}: %{y:.1%}<extra></extra>",
                )
            )

            # Add threshold line
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Warning Threshold (50%)",
                annotation_position="bottom right",
            )

            # Add range bands
            if len(df) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=df["version"],
                        y=df["similarity"] + 0.05,
                        mode="lines",
                        name="Upper Band",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["version"],
                        y=df["similarity"] - 0.05,
                        mode="lines",
                        name="Lower Band",
                        fill="tonexty",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )

            # Update layout
            fig.update_layout(
                title=f"Plagiarism Evolution: {doc_name}",
                xaxis_title="Version",
                yaxis_title="Similarity Score",
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1],
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                ),
            )

            # Add annotations for significant changes
            for change in analysis.get("significant_changes", []):
                fig.add_annotation(
                    x=change["to_version"],
                    y=change["new_similarity"],
                    text=f"⚠️ {change['similarity_change'] * 100:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-30,
                    font=dict(color="red"),
                )

        return fig

    def get_similarity_trend_analysis(self, doc_name: str) -> Dict:
        """Get trend analysis with forecasting"""
        analysis = self.analyze_plagiarism_evolution(doc_name)
        if analysis.get("status") == "insufficient_data":
            return {"status": "insufficient_data"}

        versions = self.version_manager.get_version_history(doc_name)
        if not versions:
            return {"status": "no_data"}

        # Simple linear regression for trend
        similarities = []
        timestamps = []
        for v in versions:
            if v.similarity_score is not None:
                similarities.append(v.similarity_score)
                timestamps.append(v.timestamp.timestamp())

        if len(similarities) < 3:
            return {"trend": "insufficient_data"}

        # Calculate trend
        x = np.array(timestamps)
        y = np.array(similarities)
        z = np.polyfit(x, y, 1)
        slope = z[0]

        # Forecast next 3 versions
        last_timestamp = x[-1]
        forecast_timestamps = [
            last_timestamp + (i + 1) * 86400 for i in range(3)
        ]  # 1 day increments
        forecast_similarities = [z[0] * ts + z[1] for ts in forecast_timestamps]

        trend_analysis = {
            "slope": slope,
            "direction": "increasing"
            if slope > 0
            else "decreasing"
            if slope < 0
            else "stable",
            "magnitude": abs(slope),
            "forecast": {
                "timestamps": [
                    datetime.fromtimestamp(ts).isoformat() for ts in forecast_timestamps
                ],
                "similarities": [float(s) for s in forecast_similarities],
            },
            "confidence": min(abs(slope) * 100, 100),  # Rough confidence score
        }

        # Add risk assessment
        if slope > 0.01:
            trend_analysis["risk"] = "high"
            trend_analysis["recommendation"] = (
                "This document shows increasing similarity scores. Consider reviewing for potential plagiarism."
            )
        elif slope < -0.01:
            trend_analysis["risk"] = "low"
            trend_analysis["recommendation"] = (
                "Similarity scores are decreasing, which is generally positive."
            )
        else:
            trend_analysis["risk"] = "medium"
            trend_analysis["recommendation"] = (
                "Similarity scores are stable. Monitor for any sudden changes."
            )

        return trend_analysis


# ── Smart Change Pattern Detector ──────────────────────────────────────────
class SmartChangePatternDetector:
    """Detects intelligent patterns in document changes"""

    def __init__(self):
        self.suspicious_patterns = []
        self.pattern_weights = {
            "large_addition": 0.3,
            "massive_deletion": 0.3,
            "major_replacement": 0.25,
            "multiple_replacements": 0.15,
            "copy_paste_detected": 0.5,
            "unusual_timing": 0.2,
        }

    def detect_changes(self, old_content: str, new_content: str) -> Dict:
        """Detect changes with intelligent pattern recognition"""
        tracker = ChangeTracker()
        changes = tracker.detect_changes(old_content, new_content)

        # Add pattern detection
        patterns = []
        scores = {"suspicious_score": 0}

        # Check each pattern
        if changes["total_added"] > 1000:
            patterns.append("large_addition")
            scores["suspicious_score"] += self.pattern_weights["large_addition"]
            scores["large_addition_detected"] = True

        if changes["total_deleted"] > 1000:
            patterns.append("massive_deletion")
            scores["suspicious_score"] += self.pattern_weights["massive_deletion"]
            scores["massive_deletion_detected"] = True

        if changes["similarity_ratio"] < 0.5 and changes["change_percentage"] > 50:
            patterns.append("major_replacement")
            scores["suspicious_score"] += self.pattern_weights["major_replacement"]
            scores["major_replacement_detected"] = True

        if len(changes["replacements"]) > 3:
            patterns.append("multiple_replacements")
            scores["suspicious_score"] += self.pattern_weights["multiple_replacements"]
            scores["multiple_replacements_detected"] = True

        # Detect copy-paste patterns
        if self._detect_copy_paste_pattern(old_content, new_content):
            patterns.append("copy_paste_detected")
            scores["suspicious_score"] += self.pattern_weights["copy_paste_detected"]
            scores["copy_paste_detected"] = True

        # Detect unusual timing
        scores["unusual_timing_detected"] = self._detect_unusual_timing(
            old_content, new_content
        )
        if scores["unusual_timing_detected"]:
            scores["suspicious_score"] += self.pattern_weights["unusual_timing"]

        # Normalize score
        scores["suspicious_score"] = min(scores["suspicious_score"], 1.0)
        scores["patterns_detected"] = patterns
        scores["risk_level"] = self._get_risk_level(scores["suspicious_score"])
        scores["recommendations"] = self._get_recommendations(patterns)

        return {**changes, **scores}

    def _detect_copy_paste_pattern(self, old_content: str, new_content: str) -> bool:
        """Detect copy-paste patterns in changes"""
        # Check for repeated paragraphs
        old_paragraphs = [p for p in old_content.split("\n\n") if p.strip()]
        new_paragraphs = [p for p in new_content.split("\n\n") if p.strip()]

        # Check if any paragraph appears with slight modification
        for old_p in old_paragraphs:
            for new_p in new_paragraphs:
                if len(old_p) > 100 and len(new_p) > 100:
                    if difflib.SequenceMatcher(None, old_p, new_p).ratio() > 0.8:
                        return True
        return False

    def _detect_unusual_timing(self, old_content: str, new_content: str) -> bool:
        """Detect unusual timing patterns (e.g., late night edits)"""
        # This would typically look at timestamps
        # For now, check if content length changed dramatically
        length_change = abs(len(new_content) - len(old_content))
        return length_change > 5000  # 5KB change threshold

    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on suspicious score"""
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"

    def _get_recommendations(self, patterns: List[str]) -> List[str]:
        """Get recommendations based on detected patterns"""
        recommendations = []

        if "large_addition" in patterns:
            recommendations.append(
                "Large text additions detected. Verify if content is original."
            )

        if "massive_deletion" in patterns:
            recommendations.append(
                "Large text deletions detected. Ensure content wasn't removed to hide plagiarism."
            )

        if "major_replacement" in patterns:
            recommendations.append(
                "Major content replacement detected. Review for potential external copying."
            )

        if "copy_paste_detected" in patterns:
            recommendations.append(
                "Copy-paste pattern detected. Verify source of new content."
            )

        if not recommendations:
            recommendations.append(
                "No suspicious patterns detected. Document changes appear normal."
            )

        return recommendations


# ── Version Storage Manager ─────────────────────────────────────────────────
class VersionStorageManager:
    """Manages persistent storage of versions"""

    def __init__(self, storage_path: str = "version_storage"):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        """Ensure storage directory exists"""
        import os

        os.makedirs(self.storage_path, exist_ok=True)

    def save_version_manager(
        self, version_manager: VersionManager, doc_name: str
    ) -> bool:
        """Save version manager state for a document"""
        try:
            import os

            file_path = os.path.join(self.storage_path, f"{doc_name}.json")

            data = version_manager.export_history(doc_name)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception:
            return False

    def load_version_manager(self, doc_name: str) -> Optional[VersionManager]:
        """Load version manager state for a document"""
        try:
            import os

            file_path = os.path.join(self.storage_path, f"{doc_name}.json")

            if not os.path.exists(file_path):
                return None

            with open(file_path, "r") as f:
                data = json.load(f)

            vm = VersionManager()
            vm.import_history(data)
            return vm
        except Exception:
            return None

    def list_documents(self) -> List[str]:
        """List all documents with stored versions"""
        import os

        files = os.listdir(self.storage_path)
        return [f.replace(".json", "") for f in files if f.endswith(".json")]


# ── UI Components ──────────────────────────────────────────────────────────
def render_version_control_ui(version_manager: VersionManager, doc_name: str):
    """Render version control UI in Streamlit"""
    st.subheader(f"📄 Version History: {doc_name}")

    versions = version_manager.get_version_history(doc_name)
    if not versions:
        st.info("No version history available for this document")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Versions", len(versions))
    col2.metric("Current Version", version_manager.current_versions.get(doc_name, 0))

    # Word counts
    word_counts = [v.word_count for v in versions]
    col3.metric("Average Words", int(np.mean(word_counts)) if word_counts else 0)

    # Similarity evolution
    sim_scores = [
        v.similarity_score for v in versions if v.similarity_score is not None
    ]
    col4.metric(
        "Latest Similarity", f"{sim_scores[-1] * 100:.1f}%" if sim_scores else "N/A"
    )

    # Version timeline
    st.subheader("📊 Version Timeline")
    timeline_df = version_manager.get_version_timeline(doc_name)
    if not timeline_df.empty:
        st.line_chart(
            timeline_df.set_index("Timestamp")[["Word Count", "Similarity Score"]]
        )

    # Version selector
    st.subheader("📑 Version Selector")
    version_options = [
        f"v{v.version_id} - {v.timestamp.strftime('%Y-%m-%d %H:%M')} ({v.word_count} words)"
        for v in versions
    ]
    selected_idx = st.selectbox(
        "Select Version",
        options=range(len(versions)),
        format_func=lambda i: version_options[i],
        index=len(versions) - 1,
        key=f"version_selector_{doc_name}",
    )

    selected_version = versions[selected_idx]

    # Display selected version
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Version {selected_version.version_id}**")
        st.markdown(
            f"- **Created**: {selected_version.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        st.markdown(f"- **Size**: {selected_version.size} bytes")
        st.markdown(f"- **Words**: {selected_version.word_count}")
        if selected_version.similarity_score is not None:
            st.markdown(
                f"- **Similarity Score**: {selected_version.similarity_score * 100:.1f}%"
            )

    with col2:
        if selected_version.parent_version:
            st.markdown(f"**Parent Version**: v{selected_version.parent_version}")
        if selected_version.change_summary:
            st.markdown(
                f"**Change Percentage**: {selected_version.change_summary.get('change_percentage', 0):.1f}%"
            )

    # Show content preview
    with st.expander("📝 View Content", expanded=False):
        st.text_area("Content", selected_version.content, height=300)

    # Diff view
    if selected_idx > 0:
        st.subheader("🔄 Version Diff")
        prev_version = versions[selected_idx - 1]

        diff_generator = VersionDiffGenerator()
        html_diff = diff_generator.generate_html_diff(
            prev_version.content,
            selected_version.content,
            prev_version.version_id,
            selected_version.version_id,
        )

        st.components.v1.html(html_diff, height=400, scrolling=True)

    # Actions
    st.subheader("⚙️ Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Restore This Version", key=f"restore_{doc_name}"):
            content = version_manager.restore_version(
                doc_name, selected_version.version_id
            )
            if content:
                st.success(
                    f"✅ Version {selected_version.version_id} restored successfully!"
                )
                st.rerun()

    with col2:
        if st.button("📤 Export History", key=f"export_{doc_name}"):
            data = version_manager.export_history(doc_name)
            st.download_button(
                label="Download JSON",
                data=json.dumps(data, indent=2, default=str),
                file_name=f"{doc_name}_version_history.json",
                mime="application/json",
            )

    with col3:
        if len(versions) > 1 and st.button(
            "🗑️ Delete This Version", key=f"delete_{doc_name}"
        ):
            if version_manager.delete_version(doc_name, selected_version.version_id):
                st.success(f"✅ Version {selected_version.version_id} deleted")
                st.rerun()


def render_plagiarism_evolution_ui(version_manager: VersionManager, doc_name: str):
    """Render plagiarism evolution dashboard"""
    st.subheader(f"📈 Plagiarism Evolution: {doc_name}")

    analyzer = PlagiarismEvolutionAnalyzer(version_manager)
    analysis = analyzer.analyze_plagiarism_evolution(doc_name)

    if analysis.get("status") == "insufficient_data":
        st.info("Need at least 2 versions for evolution analysis")
        return

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Versions", analysis["total_versions"])
    col2.metric(
        "Trend",
        analysis["trend"].title(),
        delta=f"{analysis['metrics']['change_velocity'] * 100:.1f}%"
        if analysis["metrics"]["change_velocity"] > 0
        else "Stable",
    )
    col3.metric("Risk Score", f"{analysis['risk_score']:.0f}%")
    col4.metric("Significant Changes", len(analysis.get("significant_changes", [])))

    # Plot evolution
    st.subheader("📊 Similarity Evolution")
    fig = analyzer.generate_evolution_plot(doc_name)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Trend analysis
    st.subheader("📈 Trend Analysis")
    trend = analyzer.get_similarity_trend_analysis(doc_name)
    if trend and trend.get("status") != "insufficient_data":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Direction", trend.get("direction", "stable").title())
            st.metric("Confidence", f"{trend.get('confidence', 0):.0f}%")

        with col2:
            risk_color = (
                "🟢"
                if trend.get("risk") == "low"
                else "🟡"
                if trend.get("risk") == "medium"
                else "🔴"
            )
            st.metric(
                "Risk Level", f"{risk_color} {trend.get('risk', 'unknown').title()}"
            )
            st.metric(
                "Recommendation",
                trend.get("recommendation", "No recommendation")[:50] + "...",
            )

        # Forecast
        if "forecast" in trend:
            st.subheader("🔮 Forecast")
            forecast_df = pd.DataFrame(
                {
                    "Version": [f"v{i}" for i in range(1, 4)],
                    "Predicted Similarity": trend["forecast"]["similarities"],
                }
            )
            st.dataframe(forecast_df, use_container_width=True)

    # Significant changes table
    if analysis.get("significant_changes"):
        st.subheader("🚨 Significant Changes")
        changes_df = pd.DataFrame(analysis["significant_changes"])
        changes_df["timestamp"] = pd.to_datetime(changes_df["timestamp"])
        st.dataframe(
            changes_df[
                [
                    "from_version",
                    "to_version",
                    "old_similarity",
                    "new_similarity",
                    "similarity_change",
                    "timestamp",
                ]
            ].style.format(
                {
                    "old_similarity": "{:.1%}",
                    "new_similarity": "{:.1%}",
                    "similarity_change": "{:.1%}",
                }
            ),
            use_container_width=True,
        )


def render_smart_detection_ui(version_manager: VersionManager, doc_name: str):
    """Render smart detection UI"""
    st.subheader(f"🔍 Smart Change Detection: {doc_name}")

    versions = version_manager.get_version_history(doc_name)
    if len(versions) < 2:
        st.info("Need at least 2 versions for change detection")
        return

    detector = SmartChangePatternDetector()
    tracker = ChangeTracker()

    # Analyze changes between consecutive versions
    changes_data = []
    for i in range(len(versions) - 1):
        old_content = versions[i].content
        new_content = versions[i + 1].content

        changes = detector.detect_changes(old_content, new_content)
        changes_data.append(
            {
                "from_version": versions[i].version_id,
                "to_version": versions[i + 1].version_id,
                "similarity_ratio": changes["similarity_ratio"],
                "change_percentage": changes["change_percentage"],
                "suspicious_score": changes["suspicious_score"],
                "risk_level": changes["risk_level"],
                "patterns": changes["patterns_detected"],
                "recommendations": changes["recommendations"],
            }
        )

    # Display summary
    st.subheader("📊 Detection Summary")
    df = pd.DataFrame(changes_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Version Changes", len(changes_data))
    col2.metric("Avg Suspicious Score", f"{df['suspicious_score'].mean():.2f}")
    col3.metric("High Risk Changes", len(df[df["risk_level"] == "high"]))

    # Display patterns
    st.subheader("🔄 Detected Patterns")
    pattern_counts = {}
    for data in changes_data:
        for pattern in data["patterns"]:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    if pattern_counts:
        pattern_df = pd.DataFrame(
            {
                "Pattern": list(pattern_counts.keys()),
                "Count": list(pattern_counts.values()),
            }
        )
        st.bar_chart(pattern_df.set_index("Pattern"))

    # Detailed changes table
    st.subheader("📋 Detailed Change Analysis")
    st.dataframe(
        df[
            [
                "from_version",
                "to_version",
                "similarity_ratio",
                "change_percentage",
                "suspicious_score",
                "risk_level",
            ]
        ].style.format(
            {
                "similarity_ratio": "{:.1%}",
                "change_percentage": "{:.1f}%",
                "suspicious_score": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    # Show recommendations
    st.subheader("💡 Recommendations")
    for idx, row in df.iterrows():
        if row["suspicious_score"] > 0.4:
            with st.expander(
                f"⚠️ Version {row['from_version']} → {row['to_version']} - {row['risk_level'].title()} Risk",
                expanded=True,
            ):
                st.markdown(f"**Suspicious Score**: {row['suspicious_score']:.2f}")
                st.markdown(f"**Patterns**: {', '.join(row['patterns'])}")
                st.markdown(f"**Similarity**: {row['similarity_ratio']:.1%}")
                st.markdown(f"**Change**: {row['change_percentage']:.1f}%")
                for rec in row["recommendations"]:
                    st.markdown(f"📌 {rec}")


def render_global_version_dashboard(version_manager: VersionManager):
    """Render global version control dashboard"""
    st.subheader("📊 Global Version Control Dashboard")

    documents = version_manager.get_all_documents()
    if not documents:
        st.info("No documents with version history found")
        return

    # Summary statistics
    total_versions = sum(version_manager.get_version_count(doc) for doc in documents)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", len(documents))
    col2.metric("Total Versions", total_versions)
    col3.metric(
        "Average Versions per Document", f"{total_versions / len(documents):.1f}"
    )

    # Document list with version counts
    st.subheader("📁 Documents")
    doc_data = []
    for doc in documents:
        versions = version_manager.get_version_history(doc)
        current = version_manager.get_current_version(doc)
        doc_data.append(
            {
                "Document": doc,
                "Versions": len(versions),
                "Current Version": current.version_id if current else 0,
                "Last Updated": versions[-1].timestamp.strftime("%Y-%m-%d %H:%M")
                if versions
                else "Never",
                "Words": current.word_count if current else 0,
            }
        )

    doc_df = pd.DataFrame(doc_data)
    st.dataframe(doc_df, use_container_width=True)

    # Document selection for detailed view
    selected_doc = st.selectbox(
        "Select Document for Detailed View",
        options=documents,
        key="global_doc_selector",
    )

    if selected_doc:
        render_version_control_ui(version_manager, selected_doc)


def initialize_version_control():
    """Initialize version control system in session state"""
    if "version_manager" not in st.session_state:
        st.session_state["version_manager"] = VersionManager()

    if "version_storage" not in st.session_state:
        st.session_state["version_storage"] = VersionStorageManager()

    if "version_tabs" not in st.session_state:
        st.session_state["version_tabs"] = {"current_tab": "Version History"}


def render_version_control_dashboard():
    """Render the main version control dashboard"""
    initialize_version_control()

    version_manager = st.session_state["version_manager"]

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📄 Version History",
            "📈 Plagiarism Evolution",
            "🔍 Smart Detection",
            "📊 Global Dashboard",
        ]
    )

    with tab1:
        documents = version_manager.get_all_documents()
        if documents:
            doc_name = st.selectbox(
                "Select Document", options=documents, key="version_history_doc_selector"
            )
            render_version_control_ui(version_manager, doc_name)
        else:
            st.info(
                "No documents with version history. Upload documents to start tracking versions."
            )

    with tab2:
        documents = version_manager.get_all_documents()
        if documents:
            doc_name = st.selectbox(
                "Select Document", options=documents, key="evolution_doc_selector"
            )
            render_plagiarism_evolution_ui(version_manager, doc_name)
        else:
            st.info("No documents with version history.")

    with tab3:
        documents = version_manager.get_all_documents()
        if documents and len(version_manager.get_version_history(documents[0])) >= 2:
            doc_name = st.selectbox(
                "Select Document", options=documents, key="smart_detection_doc_selector"
            )
            render_smart_detection_ui(version_manager, doc_name)
        else:
            st.info("Need documents with at least 2 versions for smart detection.")

    with tab4:
        render_global_version_dashboard(version_manager)


# ── Integration with Main Application ──────────────────────────────────────
def integrate_version_control_with_analysis(
    version_manager: VersionManager,
    similarity_matrix: np.ndarray,
    document_names: List[str],
    raw_texts: Dict[str, str],
) -> Dict:
    """Integrate version control with plagiarism analysis"""
    results = {}

    # Update version scores
    for doc_name in document_names:
        if doc_name in raw_texts:
            versions = version_manager.get_version_history(doc_name)
            if versions:
                # Find similarity score for current version
                doc_index = document_names.index(doc_name)
                if doc_index < len(similarity_matrix):
                    avg_similarity = np.mean(similarity_matrix[doc_index])
                    versions[-1].similarity_score = avg_similarity
                    results[doc_name] = {
                        "current_version": versions[-1].version_id,
                        "similarity_score": avg_similarity,
                    }

    return results


def migrate_existing_documents_to_version_control(
    version_manager: VersionManager, existing_documents: List[Dict]
) -> Dict:
    """Migrate existing documents to version control system"""
    migrated = 0
    for doc in existing_documents:
        doc_name = doc.get("filename", f"unknown_{hash(str(doc))}")
        content = doc.get("content", "")
        if content:
            version_manager.add_version(doc_name, content)
            migrated += 1

    return {
        "migrated_documents": migrated,
        "total_documents": len(existing_documents),
        "migration_date": datetime.now(),
    }


# ── End of Version Control Section ─────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
