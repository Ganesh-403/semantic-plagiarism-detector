"""
Document Version Control and Change Tracking

Features:
- Version history tracking
- Diff visualization
- Change detection
- Rollback capability
- Document evolution tracking
- Audit trail
- Smart merging
- Version comparison
"""

import difflib
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta  # noqa: F401
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class DocumentVersion:
    """Document version record."""

    version_id: str
    document_id: str
    document_name: str
    content: str
    timestamp: float
    author: str
    comment: str
    version_number: int
    hash: str
    size: int
    word_count: int
    change_type: str  # create, update, delete, rollback
    parent_version_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionDiff:
    """Diff between two versions."""

    version_id_a: str
    version_id_b: str
    document_id: str
    added_lines: List[str]
    removed_lines: List[str]
    modified_lines: List[Tuple[str, str]]
    unchanged_lines: List[str]
    summary: str
    change_percentage: float


@dataclass
class ChangeSummary:
    """Summary of changes for a version."""

    additions: int
    deletions: int
    modifications: int
    word_change: int
    char_change: int
    file_size_change: int
    sentiment_change: float


# ==============================================================================
# VERSION MANAGER
# ==============================================================================


class VersionManager:
    """
    Document version control and change tracking.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.versions: Dict[str, List[DocumentVersion]] = defaultdict(list)
        self.current_versions: Dict[str, str] = {}  # doc_id -> version_id
        self._load_versions()

    def _load_versions(self):
        """Load versions from storage."""
        try:
            version_path = self.storage_path / "versions.json"
            if version_path.exists():
                with open(version_path, "r") as f:
                    data = json.load(f)

                    for doc_id, versions in data.items():
                        self.versions[doc_id] = [DocumentVersion(**v) for v in versions]

                        if versions:
                            # Find current version (latest)
                            latest = max(versions, key=lambda x: x["version_number"])
                            self.current_versions[doc_id] = latest["version_id"]
        except Exception as e:
            print(f"Error loading versions: {e}")

    def _save_versions(self):
        """Save versions to storage."""
        try:
            version_path = self.storage_path / "versions.json"
            version_path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for doc_id, versions in self.versions.items():
                data[doc_id] = [asdict(v) for v in versions]

            with open(version_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving versions: {e}")

    def create_version(
        self,
        document_id: str,
        document_name: str,
        content: str,
        author: str,
        comment: str = "",
        change_type: str = "update",
    ) -> DocumentVersion:
        """
        Create a new version of a document.

        Args:
            document_id: Document ID
            document_name: Document name
            content: Document content
            author: Author name
            comment: Version comment
            change_type: Type of change

        Returns:
            DocumentVersion: New version
        """
        # Get existing versions
        existing = self.versions.get(document_id, [])

        # Calculate version number
        version_number = len(existing) + 1

        # Calculate hash
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if content actually changed
        if existing:
            last_version = existing[-1]
            if last_version.hash == content_hash:
                # No change, return last version
                return last_version

        # Determine change type if auto-detected
        if change_type == "auto":
            change_type = self._detect_change_type(content, existing)

        # Create version
        version = DocumentVersion(
            version_id=f"v_{document_id}_{version_number}_{int(time.time())}",
            document_id=document_id,
            document_name=document_name,
            content=content,
            timestamp=time.time(),
            author=author,
            comment=comment,
            version_number=version_number,
            hash=content_hash,
            size=len(content),
            word_count=len(content.split()),
            change_type=change_type,
            parent_version_id=existing[-1].version_id if existing else None,
            metadata={
                "characters": len(content),
                "lines": content.count("\n") + 1,
                "paragraphs": content.count("\n\n") + 1,
            },
        )

        # Save version
        self.versions[document_id].append(version)
        self.current_versions[document_id] = version.version_id
        self._save_versions()

        return version

    def _detect_change_type(self, content: str, existing: List[DocumentVersion]) -> str:
        """Auto-detect change type."""
        if not existing:
            return "create"

        last = existing[-1]
        if content == last.content:
            return "no_change"

        # Calculate diff
        diff = list(difflib.ndiff(last.content.splitlines(), content.splitlines()))
        additions = sum(1 for line in diff if line.startswith("+ "))
        deletions = sum(1 for line in diff if line.startswith("- "))

        if additions == 0 and deletions == 0:
            return "no_change"
        elif additions > deletions:
            return "addition"
        elif deletions > additions:
            return "deletion"
        else:
            return "modification"

    def get_version(
        self, document_id: str, version_id: str
    ) -> Optional[DocumentVersion]:
        """Get a specific version."""
        versions = self.versions.get(document_id, [])
        for version in versions:
            if version.version_id == version_id:
                return version
        return None

    def get_latest_version(self, document_id: str) -> Optional[DocumentVersion]:
        """Get the latest version."""
        versions = self.versions.get(document_id, [])
        if versions:
            return versions[-1]
        return None

    def get_all_versions(self, document_id: str) -> List[DocumentVersion]:
        """Get all versions of a document."""
        return self.versions.get(document_id, [])

    def get_version_history(self, document_id: str, limit: int = 50) -> List[Dict]:
        """Get version history."""
        versions = self.versions.get(document_id, [])

        history = []
        for version in versions[-limit:]:
            history.append(
                {
                    "version_number": version.version_number,
                    "version_id": version.version_id[:8],
                    "timestamp": datetime.fromtimestamp(version.timestamp).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "author": version.author,
                    "comment": version.comment or version.change_type,
                    "change_type": version.change_type,
                    "size": version.size,
                    "word_count": version.word_count,
                }
            )

        return history

    def rollback(
        self, document_id: str, version_id: str, author: str
    ) -> Optional[DocumentVersion]:
        """Rollback to a specific version."""
        target_version = self.get_version(document_id, version_id)
        if not target_version:
            return None

        # Create new version with content from target
        return self.create_version(
            document_id=document_id,
            document_name=target_version.document_name,
            content=target_version.content,
            author=author,
            comment=f"Rollback to version {target_version.version_number}",
            change_type="rollback",
        )

    def compare_versions(
        self, document_id: str, version_id_a: str, version_id_b: str
    ) -> Optional[VersionDiff]:
        """Compare two versions."""
        version_a = self.get_version(document_id, version_id_a)
        version_b = self.get_version(document_id, version_id_b)

        if not version_a or not version_b:
            return None

        # Get content
        lines_a = version_a.content.splitlines()
        lines_b = version_b.content.splitlines()

        # Compute diff
        diff = list(difflib.ndiff(lines_a, lines_b))

        added = []
        removed = []
        modified = []  # noqa: F841
        unchanged = []

        for line in diff:
            if line.startswith("+ "):
                added.append(line[2:])
            elif line.startswith("- "):
                removed.append(line[2:])
            elif line.startswith("? "):
                continue
            else:
                unchanged.append(line[2:])

        # Generate summary
        total_changes = len(added) + len(removed)
        total_lines = max(len(lines_a), len(lines_b))
        change_percentage = (
            (total_changes / total_lines * 100) if total_lines > 0 else 0
        )

        summary = f"{len(added)} additions, {len(removed)} deletions, {change_percentage:.1f}% changed"

        return VersionDiff(
            version_id_a=version_id_a,
            version_id_b=version_id_b,
            document_id=document_id,
            added_lines=added[:50],  # Limit for display
            removed_lines=removed[:50],
            modified_lines=[],  # Would need more sophisticated diff
            unchanged_lines=unchanged[:50],
            summary=summary,
            change_percentage=change_percentage,
        )

    def get_change_summary(self, document_id: str, version_id: str) -> ChangeSummary:
        """Get summary of changes in a version."""
        version = self.get_version(document_id, version_id)
        if not version:
            return ChangeSummary(0, 0, 0, 0, 0, 0, 0)

        # Get previous version
        versions = self.versions.get(document_id, [])
        idx = next(
            (i for i, v in enumerate(versions) if v.version_id == version_id), -1
        )

        if idx <= 0:
            return ChangeSummary(0, 0, 0, 0, 0, 0, 0)

        prev_version = versions[idx - 1]

        # Calculate diff
        diff = list(
            difflib.ndiff(
                prev_version.content.splitlines(), version.content.splitlines()
            )
        )

        additions = sum(1 for line in diff if line.startswith("+ "))
        deletions = sum(1 for line in diff if line.startswith("- "))

        return ChangeSummary(
            additions=additions,
            deletions=deletions,
            modifications=0,  # Would need more sophisticated analysis
            word_change=abs(
                len(version.content.split()) - len(prev_version.content.split())
            ),
            char_change=abs(len(version.content) - len(prev_version.content)),
            file_size_change=version.size - prev_version.size,
            sentiment_change=0.0,
        )

    def get_evolution_timeline(self, document_id: str) -> Dict[str, Any]:
        """Get document evolution timeline."""
        versions = self.versions.get(document_id, [])

        if not versions:
            return {
                "versions": [],
                "total_versions": 0,
                "timespan": 0,
                "growth_rate": 0,
                "authors": [],
            }

        # Extract timeline data
        timeline = []
        authors = set()

        for version in versions:
            timeline.append(
                {
                    "version": version.version_number,
                    "date": datetime.fromtimestamp(version.timestamp).strftime(
                        "%Y-%m-%d"
                    ),
                    "size": version.size,
                    "words": version.word_count,
                    "author": version.author,
                    "change_type": version.change_type,
                }
            )
            authors.add(version.author)

        # Calculate statistics
        total_versions = len(versions)
        timespan = versions[-1].timestamp - versions[0].timestamp

        return {
            "versions": timeline,
            "total_versions": total_versions,
            "timespan": timespan / 86400,  # Days
            "growth_rate": (timeline[-1]["size"] - timeline[0]["size"])
            / max(timespan, 1),
            "authors": list(authors),
        }


# ==============================================================================
# CHANGE TRACKER
# ==============================================================================


class ChangeTracker:
    """
    Track and analyze document changes over time.
    """

    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager

    def detect_active_periods(self, document_id: str, threshold: int = 3) -> List[Dict]:
        """Detect periods of high activity."""
        versions = self.version_manager.get_all_versions(document_id)

        if len(versions) < 2:
            return []

        # Find active periods (consecutive versions with short time gaps)
        active_periods = []
        current_period = {
            "start": versions[0].timestamp,
            "end": versions[0].timestamp,
            "count": 1,
            "versions": [versions[0]],
        }

        for i in range(1, len(versions)):
            gap = versions[i].timestamp - versions[i - 1].timestamp

            if gap < 300:  # Within 5 minutes
                current_period["end"] = versions[i].timestamp
                current_period["count"] += 1
                current_period["versions"].append(versions[i])
            else:
                if current_period["count"] >= threshold:
                    active_periods.append(current_period)
                current_period = {
                    "start": versions[i].timestamp,
                    "end": versions[i].timestamp,
                    "count": 1,
                    "versions": [versions[i]],
                }

        if current_period["count"] >= threshold:
            active_periods.append(current_period)

        return active_periods

    def analyze_editing_patterns(self, document_id: str) -> Dict[str, Any]:
        """Analyze editing patterns."""
        versions = self.version_manager.get_all_versions(document_id)

        if len(versions) < 2:
            return {"pattern": "insufficient_data"}

        # Calculate time gaps
        gaps = []
        for i in range(1, len(versions)):
            gap = versions[i].timestamp - versions[i - 1].timestamp
            gaps.append(gap)

        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        std_gap = (
            (sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 if gaps else 0
        )

        # Detect pattern
        if avg_gap < 60:
            pattern = "continuous"
        elif avg_gap < 600:
            pattern = "regular"
        elif avg_gap < 3600:
            pattern = "sporadic"
        else:
            pattern = "infrequent"

        return {
            "pattern": pattern,
            "avg_time_between_edits": avg_gap,
            "std_time_between_edits": std_gap,
            "total_edits": len(versions) - 1,
            "edits_per_day": (len(versions) - 1)
            / ((versions[-1].timestamp - versions[0].timestamp) / 86400)
            if versions
            else 0,
        }


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_version_control_ui():
    """Render version control UI."""
    st.subheader("📝 Document Version Control")

    # Initialize
    if "version_manager" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        st.session_state.version_manager = VersionManager(data_dir / "versions")

    version_manager = st.session_state.version_manager

    # Document selector
    docs = st.session_state.get("doc_names", [])
    if not docs:
        st.info("No documents available for version control")
        return

    selected_doc = st.selectbox("Select Document", docs, key="version_doc_select")

    if selected_doc:
        # Get document ID (using name as ID for simplicity)
        doc_id = selected_doc

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 History", "🔍 Compare", "📊 Evolution", "⚙️ Rollback"]
        )

        with tab1:
            render_version_history(version_manager, doc_id, selected_doc)

        with tab2:
            render_version_comparison(version_manager, doc_id)

        with tab3:
            render_evolution_analytics(version_manager, doc_id)

        with tab4:
            render_rollback_ui(version_manager, doc_id, selected_doc)


def render_version_history(version_manager: VersionManager, doc_id: str, doc_name: str):
    """Render version history."""
    st.markdown("#### 📋 Version History")

    versions = version_manager.get_all_versions(doc_id)

    if not versions:
        st.info("No versions found for this document")
        return

    # Create current version if not exists
    current = version_manager.get_latest_version(doc_id)
    if not current:
        # Create initial version from current document content
        content = st.session_state.get(f"doc_content_{doc_id}", "")
        if content:
            version_manager.create_version(
                doc_id, doc_name, content, "system", "Initial version", "create"
            )
            versions = version_manager.get_all_versions(doc_id)

    # Display versions
    st.caption(f"Total versions: {len(versions)}")

    # Create version table
    version_data = []
    for version in reversed(versions):
        version_data.append(
            {
                "Version": f"v{version.version_number}",
                "Date": datetime.fromtimestamp(version.timestamp).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Author": version.author,
                "Change Type": version.change_type,
                "Words": f"{version.word_count:,}",
                "Size": f"{version.size:,}",
                "Comment": version.comment[:50] + "..."
                if len(version.comment) > 50
                else version.comment,
            }
        )

    df = pd.DataFrame(version_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Version details
    st.markdown("#### 📄 Version Details")
    selected_version = st.selectbox(
        "Select Version to View",
        [
            f"v{v.version_number} - {v.author} - {datetime.fromtimestamp(v.timestamp).strftime('%Y-%m-%d %H:%M')}"
            for v in reversed(versions)
        ],
        key="version_select_detail",
    )

    if selected_version:
        version_num = int(selected_version.split()[0][1:])
        selected = next((v for v in versions if v.version_number == version_num), None)

        if selected:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text_area(
                    "Content",
                    selected.content[:500] + "..."
                    if len(selected.content) > 500
                    else selected.content,
                    height=200,
                    disabled=True,
                )
            with col2:
                st.metric("Word Count", selected.word_count)
                st.metric("Size", f"{selected.size:,} chars")
                st.caption(f"Hash: {selected.hash[:8]}...")

                # Change summary
                summary = version_manager.get_change_summary(
                    doc_id, selected.version_id
                )
                if summary:
                    st.metric("Additions", summary.additions)
                    st.metric("Deletions", summary.deletions)


def render_version_comparison(version_manager: VersionManager, doc_id: str):
    """Render version comparison UI."""
    st.markdown("#### 🔍 Compare Versions")

    versions = version_manager.get_all_versions(doc_id)

    if len(versions) < 2:
        st.info("Need at least 2 versions to compare")
        return

    # Select versions to compare
    version_options = [f"v{v.version_number} - {v.author}" for v in versions]

    col1, col2 = st.columns(2)
    with col1:
        version_a = st.selectbox(
            "Version A (Older)", version_options, index=0, key="compare_a"
        )
    with col2:
        version_b = st.selectbox(
            "Version B (Newer)",
            version_options,
            index=len(version_options) - 1,
            key="compare_b",
        )

    if version_a and version_b:
        # Get version objects
        v_a_num = int(version_a.split()[0][1:])
        v_b_num = int(version_b.split()[0][1:])

        v_a = next((v for v in versions if v.version_number == v_a_num), None)
        v_b = next((v for v in versions if v.version_number == v_b_num), None)

        if v_a and v_b:
            # Compare
            diff = version_manager.compare_versions(
                doc_id, v_a.version_id, v_b.version_id
            )

            if diff:
                # Summary
                st.markdown(f"**{diff.summary}**")
                st.progress(min(diff.change_percentage / 100, 1.0))

                # Side-by-side diff
                st.markdown("#### 📝 Side-by-Side Comparison")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Version A (v{v_a_num})**")
                    lines_a = v_a.content.splitlines()[:50]
                    if len(lines_a) > 50:
                        lines_a.append("...")
                    st.text_area(
                        "", "\n".join(lines_a), height=300, disabled=True, key="diff_a"
                    )

                with col2:
                    st.markdown(f"**Version B (v{v_b_num})**")
                    lines_b = v_b.content.splitlines()[:50]
                    if len(lines_b) > 50:
                        lines_b.append("...")
                    st.text_area(
                        "", "\n".join(lines_b), height=300, disabled=True, key="diff_b"
                    )

                # Change details
                st.markdown("#### 📊 Change Details")
                col1, col2, col3 = st.columns(3)
                col1.metric("Added Lines", len(diff.added_lines))
                col2.metric("Removed Lines", len(diff.removed_lines))
                col3.metric("Changed", f"{diff.change_percentage:.1f}%")


def render_evolution_analytics(version_manager: VersionManager, doc_id: str):
    """Render document evolution analytics."""
    st.markdown("#### 📊 Document Evolution")

    # Get evolution data
    timeline_data = version_manager.get_evolution_timeline(doc_id)

    if not timeline_data["versions"]:
        st.info("No evolution data available")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Versions", timeline_data["total_versions"])
    col2.metric("Timespan", f"{timeline_data['timespan']:.1f} days")
    col3.metric("Authors", len(timeline_data["authors"]))
    col4.metric("Growth Rate", f"{timeline_data['growth_rate']:.1f} bytes/day")

    # Evolution chart
    df = pd.DataFrame(timeline_data["versions"])
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Size Evolution",
            "Word Count",
            "Edit Frequency",
            "Author Contributions",
        ),
    )

    # Size evolution
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["size"], mode="lines+markers", name="Size"),
        row=1,
        col=1,
    )

    # Word count
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["words"], mode="lines+markers", name="Words"),
        row=1,
        col=2,
    )

    # Edit frequency (versions per day)
    if len(df) > 1:
        freq = df.groupby("date").size()
        fig.add_trace(go.Bar(x=freq.index, y=freq.values, name="Edits"), row=2, col=1)

    # Author contributions
    author_counts = df["author"].value_counts()
    fig.add_trace(
        go.Pie(labels=author_counts.index, values=author_counts.values), row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Activity analysis
    st.markdown("#### 🔥 Activity Analysis")
    tracker = ChangeTracker(version_manager)
    active_periods = tracker.detect_active_periods(doc_id)

    if active_periods:
        st.success(f"Detected {len(active_periods)} active editing periods")
        for period in active_periods:
            st.caption(
                f"• {period['count']} edits in {((period['end'] - period['start']) / 60):.1f} minutes"
            )
    else:
        st.info("No significant active periods detected")


def render_rollback_ui(version_manager: VersionManager, doc_id: str, doc_name: str):
    """Render rollback UI."""
    st.markdown("#### ⏪ Rollback to Previous Version")

    versions = version_manager.get_all_versions(doc_id)

    if len(versions) < 2:
        st.info("Need at least 2 versions to rollback")
        return

    # Select version to rollback to
    version_options = [
        f"v{v.version_number} - {v.author} - {datetime.fromtimestamp(v.timestamp).strftime('%Y-%m-%d %H:%M')}"
        for v in reversed(versions)
    ]

    selected = st.selectbox(
        "Select Version to Rollback To",
        version_options,
        index=1,  # Skip latest
        help="This will create a new version with the content from the selected version",
    )

    if selected:
        version_num = int(selected.split()[0][1:])
        target = next((v for v in versions if v.version_number == version_num), None)

        if target:
            # Show preview
            st.markdown("#### 📄 Preview")
            st.text_area(
                "Content to rollback to",
                target.content[:500] + "..."
                if len(target.content) > 500
                else target.content,
                height=150,
                disabled=True,
            )

            # Rollback
            author = st.session_state.get("username", "system")
            comment = st.text_area(
                "Rollback Comment", f"Rollback to version {version_num}"
            )  # noqa: F841

            if st.button(
                "⏪ Execute Rollback", type="primary", use_container_width=True
            ):
                with st.spinner("Rolling back..."):
                    new_version = version_manager.rollback(
                        doc_id, target.version_id, author
                    )

                    if new_version:
                        # Update document content in session state
                        st.session_state[f"doc_content_{doc_id}"] = target.content

                        st.success(
                            f"✅ Successfully rolled back to version {version_num}"
                        )
                        st.info(f"Created new version v{new_version.version_number}")
                        st.rerun()
                    else:
                        st.error("❌ Rollback failed")


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_version_control():
    """Initialize version control system."""
    if "version_control_initialized" not in st.session_state:
        st.session_state.version_control_initialized = True

        # Create version manager
        data_dir = Path(st.session_state.get("data_dir", "."))
        version_manager = VersionManager(data_dir / "versions")
        st.session_state.version_manager = version_manager
