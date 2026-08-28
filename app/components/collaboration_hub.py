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
Collaboration Hub for Plagiarism Review Teams

Features:
- Team workspaces and project organization
- Case assignment and tracking
- Discussion threads per case
- Review workflow management
- Shared annotations
- Team analytics and performance metrics
- Review queues and prioritization
- Real-time activity feed
"""

import json
import time
import uuid  # noqa: F401
from collections import Counter, defaultdict  # noqa: F401
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta  # noqa: F401
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple  # noqa: F401

import pandas as pd  # noqa: F401
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class CaseStatus(Enum):
    """Case workflow status."""

    NEW = "new"
    OPEN = "open"
    IN_REVIEW = "in_review"
    UNDER_DISCUSSION = "under_discussion"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REJECTED = "rejected"


class CasePriority(Enum):
    """Case priority levels."""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class ReviewAction(Enum):
    """Review actions."""

    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    ASSIGN = "assign"


@dataclass
class TeamWorkspace:
    """Team workspace definition."""

    id: str
    name: str
    description: str
    members: list[str]
    created_at: float
    created_by: str
    projects: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlagiarismCase:
    """Plagiarism case record."""

    id: str
    title: str
    description: str
    doc_a: str
    doc_b: str
    similarity_score: float
    status: CaseStatus
    priority: CasePriority
    assigned_to: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    created_by: str = ""
    comments: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscussionThread:
    """Discussion thread."""

    id: str
    case_id: str
    messages: list[dict[str, Any]]
    created_at: float
    updated_at: float
    participants: list[str]
    resolved: bool = False


@dataclass
class ReviewQueue:
    """Review queue."""

    id: str
    name: str
    case_ids: list[str]
    priority: CasePriority
    assigned_to: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# COLLABORATION HUB MANAGER
# ==============================================================================


class CollaborationHub:
    """
    Main collaboration hub manager.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.workspaces: dict[str, TeamWorkspace] = {}
        self.cases: dict[str, PlagiarismCase] = {}
        self.discussions: dict[str, DiscussionThread] = {}
        self.queues: dict[str, ReviewQueue] = {}
        self.activity_feed: list[dict] = []
        self._load_data()

    def _load_data(self):
        """Load data from storage."""
        try:
            data_path = self.storage_path / "collaboration_data.json"
            if data_path.exists():
                with open(data_path, "r") as f:
                    data = json.load(f)

                    self.workspaces = {
                        k: TeamWorkspace(**v)
                        for k, v in data.get("workspaces", {}).items()
                    }

                    self.cases = {
                        k: PlagiarismCase(**v) for k, v in data.get("cases", {}).items()
                    }

                    self.discussions = {
                        k: DiscussionThread(**v)
                        for k, v in data.get("discussions", {}).items()
                    }

                    self.queues = {
                        k: ReviewQueue(**v) for k, v in data.get("queues", {}).items()
                    }

                    self.activity_feed = data.get("activity_feed", [])
        except Exception as e:
            print(f"Error loading collaboration data: {e}")

    def _save_data(self):
        """Save data to storage."""
        try:
            data_path = self.storage_path / "collaboration_data.json"
            data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "workspaces": {k: asdict(v) for k, v in self.workspaces.items()},
                "cases": {k: asdict(v) for k, v in self.cases.items()},
                "discussions": {k: asdict(v) for k, v in self.discussions.items()},
                "queues": {k: asdict(v) for k, v in self.queues.items()},
                "activity_feed": self.activity_feed[-100:],  # Keep last 100
            }

            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving collaboration data: {e}")

    def create_workspace(
        self, name: str, description: str, created_by: str, members: list[str] = None
    ) -> TeamWorkspace:
        """Create a new workspace."""
        workspace = TeamWorkspace(
            id=f"ws_{int(time.time())}",
            name=name,
            description=description,
            members=members or [created_by],
            created_at=time.time(),
            created_by=created_by,
        )
        self.workspaces[workspace.id] = workspace
        self._save_data()
        self._add_activity(f"Workspace '{name}' created by {created_by}")
        return workspace

    def create_case(
        self,
        title: str,
        description: str,
        doc_a: str,
        doc_b: str,
        similarity_score: float,
        created_by: str,
        priority: CasePriority = CasePriority.MEDIUM,
    ) -> PlagiarismCase:
        """Create a new plagiarism case."""
        case = PlagiarismCase(
            id=f"case_{int(time.time())}",
            title=title,
            description=description,
            doc_a=doc_a,
            doc_b=doc_b,
            similarity_score=similarity_score,
            status=CaseStatus.NEW,
            priority=priority,
            created_by=created_by,
        )
        self.cases[case.id] = case
        self._save_data()
        self._add_activity(f"Case '{title}' created by {created_by}")
        return case

    def assign_case(self, case_id: str, assignee: str):
        """Assign a case to a reviewer."""
        case = self.cases.get(case_id)
        if case:
            case.assigned_to = assignee
            case.status = CaseStatus.OPEN
            case.updated_at = time.time()
            self._add_activity(f"Case '{case.title}' assigned to {assignee}")
            self._save_data()

    def update_case_status(self, case_id: str, status: CaseStatus, comment: str = ""):
        """Update case status."""
        case = self.cases.get(case_id)
        if case:
            old_status = case.status
            case.status = status
            case.updated_at = time.time()

            # Add to history
            case.history.append(
                {
                    "timestamp": time.time(),
                    "user": st.session_state.get("username", "system"),
                    "action": f"Status changed from {old_status.value} to {status.value}",
                    "comment": comment,
                }
            )

            self._add_activity(f"Case '{case.title}' status changed to {status.value}")
            self._save_data()

    def add_comment(self, case_id: str, comment: str, user: str):
        """Add comment to a case."""
        case = self.cases.get(case_id)
        if case:
            case.comments.append(
                {"timestamp": time.time(), "user": user, "comment": comment}
            )
            case.updated_at = time.time()
            self._save_data()
            self._add_activity(f"New comment on case '{case.title}' by {user}")

    def add_annotation(self, case_id: str, annotation: dict[str, Any], user: str):
        """Add annotation to a case."""
        case = self.cases.get(case_id)
        if case:
            annotation["timestamp"] = time.time()
            annotation["user"] = user
            case.annotations.append(annotation)
            case.updated_at = time.time()
            self._save_data()

    def create_discussion(
        self, case_id: str, initial_message: str, user: str
    ) -> DiscussionThread:
        """Create a discussion thread for a case."""
        discussion = DiscussionThread(
            id=f"disc_{int(time.time())}",
            case_id=case_id,
            messages=[
                {"timestamp": time.time(), "user": user, "message": initial_message}
            ],
            created_at=time.time(),
            updated_at=time.time(),
            participants=[user],
        )
        self.discussions[discussion.id] = discussion

        # Update case status
        case = self.cases.get(case_id)
        if case and case.status == CaseStatus.OPEN:
            case.status = CaseStatus.UNDER_DISCUSSION

        self._save_data()
        self._add_activity(
            f"Discussion started on case '{case.title if case else case_id}'"
        )
        return discussion

    def add_message_to_discussion(self, discussion_id: str, message: str, user: str):
        """Add message to a discussion thread."""
        discussion = self.discussions.get(discussion_id)
        if discussion:
            discussion.messages.append(
                {"timestamp": time.time(), "user": user, "message": message}
            )
            discussion.updated_at = time.time()
            if user not in discussion.participants:
                discussion.participants.append(user)
            self._save_data()

    def create_queue(
        self, name: str, priority: CasePriority = CasePriority.MEDIUM
    ) -> ReviewQueue:
        """Create a review queue."""
        queue = ReviewQueue(
            id=f"queue_{int(time.time())}", name=name, case_ids=[], priority=priority
        )
        self.queues[queue.id] = queue
        self._save_data()
        return queue

    def add_to_queue(self, queue_id: str, case_id: str):
        """Add case to queue."""
        queue = self.queues.get(queue_id)
        if queue and case_id not in queue.case_ids:
            queue.case_ids.append(case_id)
            self._save_data()

    def remove_from_queue(self, queue_id: str, case_id: str):
        """Remove case from queue."""
        queue = self.queues.get(queue_id)
        if queue and case_id in queue.case_ids:
            queue.case_ids.remove(case_id)
            self._save_data()

    def _add_activity(self, message: str):
        """Add to activity feed."""
        self.activity_feed.append(
            {
                "timestamp": time.time(),
                "message": message,
                "user": st.session_state.get("username", "system"),
            }
        )
        # Keep last 1000 activities
        if len(self.activity_feed) > 1000:
            self.activity_feed = self.activity_feed[-1000:]
        self._save_data()

    def get_cases_by_status(self, status: CaseStatus) -> list[PlagiarismCase]:
        """Get cases by status."""
        return [c for c in self.cases.values() if c.status == status]

    def get_cases_by_assignee(self, assignee: str) -> list[PlagiarismCase]:
        """Get cases by assignee."""
        return [c for c in self.cases.values() if c.assigned_to == assignee]

    def get_team_metrics(self) -> dict[str, Any]:
        """Get team performance metrics."""
        total_cases = len(self.cases)
        resolved = len(
            [
                c
                for c in self.cases.values()
                if c.status in [CaseStatus.RESOLVED, CaseStatus.CLOSED]
            ]
        )

        # Cases by status
        status_counts = Counter([c.status.value for c in self.cases.values()])

        # Cases by priority
        priority_counts = Counter([c.priority.name for c in self.cases.values()])

        # Average resolution time
        avg_resolution_time = 0
        if resolved > 0:
            total_time = 0
            resolved_cases = [
                c
                for c in self.cases.values()
                if c.status in [CaseStatus.RESOLVED, CaseStatus.CLOSED]
            ]
            for case in resolved_cases:
                created = datetime.fromtimestamp(case.created_at)
                updated = datetime.fromtimestamp(case.updated_at)
                total_time += (updated - created).total_seconds()
            avg_resolution_time = total_time / resolved

        # Workload by assignee
        assignee_counts = Counter(
            [c.assigned_to for c in self.cases.values() if c.assigned_to]
        )

        return {
            "total_cases": total_cases,
            "resolved": resolved,
            "completion_rate": resolved / total_cases if total_cases > 0 else 0,
            "status_counts": dict(status_counts),
            "priority_counts": dict(priority_counts),
            "avg_resolution_time": avg_resolution_time,
            "assignee_counts": dict(assignee_counts),
            "open_cases": len(
                [
                    c
                    for c in self.cases.values()
                    if c.status
                    not in [CaseStatus.RESOLVED, CaseStatus.CLOSED, CaseStatus.REJECTED]
                ]
            ),
        }


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_collaboration_hub():
    """Render collaboration hub UI."""
    st.subheader("👥 Collaboration Hub")

    # Initialize
    if "collaboration_hub" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        st.session_state.collaboration_hub = CollaborationHub(
            data_dir / "collaboration"
        )

    hub = st.session_state.collaboration_hub
    user = st.session_state.get("username", "anonymous")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Cases", "💬 Discussions", "📊 Analytics", "📌 Queues", "⚙️ Workspace"]
    )

    with tab1:
        render_case_management(hub, user)

    with tab2:
        render_discussion_forum(hub, user)

    with tab3:
        render_collaboration_analytics(hub)

    with tab4:
        render_queue_management(hub)

    with tab5:
        render_workspace_management(hub, user)


def render_case_management(hub: CollaborationHub, user: str):
    """Render case management UI."""
    st.markdown("#### 📋 Case Management")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status", ["All"] + [s.value for s in CaseStatus]
        )
    with col2:
        priority_filter = st.selectbox(
            "Filter by Priority", ["All"] + [p.name for p in CasePriority]
        )
    with col3:
        assignee_filter = st.selectbox(
            "Filter by Assignee",
            ["All", "Unassigned", user]
            + list(set([c.assigned_to for c in hub.cases.values() if c.assigned_to])),
        )

    # Get filtered cases
    cases = list(hub.cases.values())

    if status_filter != "All":
        cases = [c for c in cases if c.status.value == status_filter]
    if priority_filter != "All":
        cases = [c for c in cases if c.priority.name == priority_filter]
    if assignee_filter == "Unassigned":
        cases = [c for c in cases if c.assigned_to is None]
    elif assignee_filter != "All":
        cases = [c for c in cases if c.assigned_to == assignee_filter]

    # Create new case
    with st.expander("➕ Create New Case", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            case_title = st.text_input("Case Title")
            doc_a = st.text_input("Document A")
        with col2:
            case_priority = st.selectbox("Priority", [p.name for p in CasePriority])
            doc_b = st.text_input("Document B")

        case_description = st.text_area("Description")
        similarity = st.slider("Similarity Score", 0.0, 1.0, 0.75)

        if st.button("Create Case", use_container_width=True):
            if case_title and doc_a and doc_b:
                case = hub.create_case(
                    title=case_title,
                    description=case_description,
                    doc_a=doc_a,
                    doc_b=doc_b,
                    similarity_score=similarity,
                    created_by=user,
                    priority=CasePriority[case_priority],
                )
                st.success(f"✅ Case created: {case.id}")
                st.rerun()

    # Display cases
    if not cases:
        st.info("No cases found")
        return

    # Summary
    st.caption(f"Showing {len(cases)} cases")

    # Case list
    for case in cases:
        # Priority colors
        priority_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🔵", "LOW": "⚪"}

        with st.expander(
            f"{priority_colors.get(case.priority.name, '')} {case.title} - {case.status.value} "
            f"({case.similarity_score:.1%} similarity)",
            expanded=case.status in [CaseStatus.NEW, CaseStatus.OPEN],
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {case.description}")
                st.markdown(f"**Documents:** {case.doc_a} ↔ {case.doc_b}")
                st.markdown(f"**Similarity:** {case.similarity_score:.1%}")
                st.caption(
                    f"Created: {datetime.fromtimestamp(case.created_at).strftime('%Y-%m-%d %H:%M')}"
                )
                if case.assigned_to:
                    st.caption(f"Assigned to: {case.assigned_to}")

            with col2:
                # Actions
                if case.assigned_to is None:
                    if st.button(f"📋 Assign to me", key=f"assign_{case.id}"):  # noqa: F541
                        hub.assign_case(case.id, user)
                        st.rerun()

                if case.status not in [
                    CaseStatus.RESOLVED,
                    CaseStatus.CLOSED,
                    CaseStatus.REJECTED,
                ]:
                    status_options = [
                        s.value
                        for s in CaseStatus
                        if s not in [CaseStatus.NEW, CaseStatus.CLOSED]
                    ]
                    new_status = st.selectbox(
                        "Update Status", status_options, key=f"status_{case.id}"
                    )
                    if st.button(f"Update", key=f"update_{case.id}"):  # noqa: F541
                        hub.update_case_status(case.id, CaseStatus(new_status))
                        st.rerun()

                # Add comment
                comment = st.text_area(
                    "Add Comment",
                    key=f"comment_{case.id}",
                    placeholder="Enter comment...",
                )
                if st.button(f"💬 Add Comment", key=f"add_comment_{case.id}"):  # noqa: F541
                    if comment:
                        hub.add_comment(case.id, comment, user)
                        st.rerun()

            # Comments
            if case.comments:
                st.markdown("#### 💬 Comments")
                for comment in case.comments[-5:]:
                    st.markdown(
                        f"**{comment['user']}** ({datetime.fromtimestamp(comment['timestamp']).strftime('%H:%M')}): {comment['comment']}"
                    )


def render_discussion_forum(hub: CollaborationHub, user: str):
    """Render discussion forum UI."""
    st.markdown("#### 💬 Discussion Forum")

    # Show discussions
    if not hub.discussions:
        st.info("No active discussions")
        return

    for disc_id, discussion in hub.discussions.items():
        case = hub.cases.get(discussion.case_id)
        case_title = case.title if case else discussion.case_id

        with st.expander(
            f"💬 {case_title} - {len(discussion.messages)} messages", expanded=False
        ):
            # Messages
            for msg in discussion.messages[-10:]:
                st.markdown(
                    f"**{msg['user']}** ({datetime.fromtimestamp(msg['timestamp']).strftime('%H:%M')}): {msg['message']}"
                )

            # Add message
            new_message = st.text_area(
                "Type your message",
                key=f"msg_{disc_id}",
                placeholder="Enter message...",
            )
            if st.button(f"📤 Send", key=f"send_{disc_id}"):  # noqa: F541
                if new_message:
                    hub.add_message_to_discussion(disc_id, new_message, user)
                    st.rerun()

            # Show participants
            st.caption(f"Participants: {', '.join(discussion.participants)}")


def render_collaboration_analytics(hub: CollaborationHub):
    """Render collaboration analytics."""
    st.markdown("#### 📊 Collaboration Analytics")

    metrics = hub.get_team_metrics()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", metrics["total_cases"])
    col2.metric("Resolved", metrics["resolved"])
    col3.metric("Completion Rate", f"{metrics['completion_rate']:.1%}")
    col4.metric("Open Cases", metrics["open_cases"])

    # Charts
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cases by Status",
            "Cases by Priority",
            "Workload by Assignee",
            "Resolution Trend",
        ),
    )

    # Status distribution
    if metrics["status_counts"]:
        statuses = list(metrics["status_counts"].keys())
        counts = list(metrics["status_counts"].values())
        fig.add_trace(go.Pie(labels=statuses, values=counts), row=1, col=1)

    # Priority distribution
    if metrics["priority_counts"]:
        priorities = list(metrics["priority_counts"].keys())
        counts = list(metrics["priority_counts"].values())
        fig.add_trace(go.Pie(labels=priorities, values=counts), row=1, col=2)

    # Workload by assignee
    if metrics["assignee_counts"]:
        assignees = list(metrics["assignee_counts"].keys())
        counts = list(metrics["assignee_counts"].values())
        fig.add_trace(go.Bar(x=assignees, y=counts), row=2, col=1)

    # Resolution time
    if metrics["avg_resolution_time"] > 0:
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics["avg_resolution_time"] / 3600,
                title={"text": "Avg Resolution (hours)"},
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Activity feed
    st.markdown("#### 🔔 Activity Feed")
    if hub.activity_feed:
        for activity in hub.activity_feed[-20:]:
            st.caption(
                f"{datetime.fromtimestamp(activity['timestamp']).strftime('%H:%M')} - {activity['user']}: {activity['message']}"
            )
    else:
        st.info("No activity yet")


def render_queue_management(hub: CollaborationHub):
    """Render queue management UI."""
    st.markdown("#### 📌 Queue Management")

    # Create queue
    with st.expander("➕ Create Queue", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            queue_name = st.text_input("Queue Name")
        with col2:
            queue_priority = st.selectbox("Priority", [p.name for p in CasePriority])

        if st.button("Create Queue", use_container_width=True):
            if queue_name:
                hub.create_queue(queue_name, CasePriority[queue_priority])
                st.success("✅ Queue created")
                st.rerun()

    # Display queues
    if not hub.queues:
        st.info("No queues created")
        return

    for queue in hub.queues.values():
        with st.expander(
            f"📌 {queue.name} ({len(queue.case_ids)} cases)", expanded=False
        ):
            # Cases in queue
            for case_id in queue.case_ids:
                case = hub.cases.get(case_id)
                if case:
                    st.markdown(
                        f"• {case.title} ({case.status.value}) - {case.similarity_score:.1%}"
                    )

            # Add case to queue
            available_cases = [
                c for c in hub.cases.values() if c.id not in queue.case_ids
            ]
            if available_cases:
                case_options = [f"{c.title} ({c.id})" for c in available_cases]
                selected = st.selectbox(
                    "Add case to queue", case_options, key=f"add_to_{queue.id}"
                )
                if st.button(f"➕ Add", key=f"add_{queue.id}"):  # noqa: F541
                    case_id = selected.split("(")[-1].replace(")", "")
                    hub.add_to_queue(queue.id, case_id)
                    st.rerun()


def render_workspace_management(hub: CollaborationHub, user: str):
    """Render workspace management UI."""
    st.markdown("#### ⚙️ Workspace Management")

    # Create workspace
    with st.expander("➕ Create Workspace", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            ws_name = st.text_input("Workspace Name")
        with col2:
            ws_description = st.text_input("Description")

        members = st.text_input(
            "Members (comma separated)", placeholder="user1, user2, user3"
        )

        if st.button("Create Workspace", use_container_width=True):
            if ws_name:
                member_list = [m.strip() for m in members.split(",") if m.strip()]
                hub.create_workspace(ws_name, ws_description, user, member_list)
                st.success("✅ Workspace created")
                st.rerun()

    # Display workspaces
    if not hub.workspaces:
        st.info("No workspaces created")
        return

    for workspace in hub.workspaces.values():
        with st.expander(f"🏢 {workspace.name}", expanded=False):
            st.markdown(f"**Description:** {workspace.description}")
            st.markdown(f"**Created by:** {workspace.created_by}")
            st.markdown(f"**Members:** {', '.join(workspace.members)}")
            st.markdown(
                f"**Projects:** {', '.join(workspace.projects) if workspace.projects else 'None'}"
            )


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_collaboration_hub():
    """Initialize collaboration hub."""
    if "collaboration_hub_initialized" not in st.session_state:
        st.session_state.collaboration_hub_initialized = True

        # Create collaboration hub
        data_dir = Path(st.session_state.get("data_dir", "."))
        hub = CollaborationHub(data_dir / "collaboration")
        st.session_state.collaboration_hub = hub
