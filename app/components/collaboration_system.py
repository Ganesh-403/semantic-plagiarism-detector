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
# ── SECTION: REAL-TIME COLLABORATION & REVIEW SYSTEM (Issue #1986) ──────────
# ───────────────────────────────────────────────────────────────────────────────

import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class DocumentAnnotation:
    """Represents a user annotation on a document"""

    id: str
    doc_name: str
    user_id: str
    annotation_type: str  # 'comment', 'highlight', 'note', 'suggestion'
    content: str
    created_at: datetime
    updated_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    replies: list[dict] = None
    metadata: dict = None

    def __post_init__(self):
        if self.replies is None:
            self.replies = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }

    def add_reply(self, user_id: str, content: str):
        reply = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "content": content,
            "created_at": datetime.now().isoformat(),
        }
        self.replies.append(reply)
        self.updated_at = datetime.now()

    def resolve(self, user_id: str):
        self.resolved = True
        self.resolved_at = datetime.now()
        self.resolved_by = user_id


@dataclass
class ReviewWorkflow:
    """Represents a review workflow for a document"""

    id: str
    doc_name: str
    created_by: str
    created_at: datetime
    status: str  # 'pending', 'in_review', 'approved', 'rejected', 'needs_changes'
    reviewers: list[str]
    decisions: dict[str, dict]  # reviewer_id -> {decision, timestamp, comments}
    current_reviewer_index: int = 0
    timeline: list[dict] = None
    metadata: dict = None

    def __post_init__(self):
        if self.timeline is None:
            self.timeline = []
        if self.metadata is None:
            self.metadata = {}

    def add_timeline_event(self, event_type: str, user_id: str, details: dict):
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
        }
        self.timeline.append(event)

    def submit_review(self, reviewer_id: str, decision: str, comments: str = ""):
        """Submit a review decision"""
        self.decisions[reviewer_id] = {
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
            "comments": comments,
        }
        self.add_timeline_event(
            "review_submitted",
            reviewer_id,
            {"decision": decision, "comments": comments},
        )

        # Update workflow status based on decisions
        self._update_status()

    def _update_status(self):
        """Update workflow status based on review decisions"""
        if not self.decisions:
            return

        decisions = [d["decision"] for d in self.decisions.values()]

        if "rejected" in decisions:
            self.status = "rejected"
        elif "needs_changes" in decisions:
            self.status = "needs_changes"
        elif len(decisions) >= len(self.reviewers) and all(
            d == "approved" for d in decisions
        ):
            self.status = "approved"
        else:
            self.status = "in_review"

        self.add_timeline_event("status_updated", "system", {"status": self.status})

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "decisions": self.decisions,
            "timeline": self.timeline,
        }


@dataclass
class UserSession:
    """Represents a user's active session"""

    user_id: str
    username: str
    doc_name: Optional[str] = None
    is_active: bool = True
    last_activity: datetime = None
    current_action: str = "idle"  # 'viewing', 'editing', 'reviewing', 'commenting'

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()

    def update_activity(self, doc_name: str, action: str):
        self.doc_name = doc_name
        self.current_action = action
        self.last_activity = datetime.now()
        self.is_active = True

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "doc_name": self.doc_name,
            "is_active": self.is_active,
            "last_activity": self.last_activity.isoformat(),
            "current_action": self.current_action,
        }


# ── Managers ────────────────────────────────────────────────────────────────


class AnnotationManager:
    """Manages document annotations and comments"""

    def __init__(self):
        self.annotations: dict[str, list[DocumentAnnotation]] = defaultdict(list)
        self.annotation_count = 0

    def add_annotation(
        self,
        doc_name: str,
        user_id: str,
        content: str,
        annotation_type: str = "comment",
    ) -> DocumentAnnotation:
        """Add a new annotation to a document"""
        annotation = DocumentAnnotation(
            id=str(uuid.uuid4()),
            doc_name=doc_name,
            user_id=user_id,
            annotation_type=annotation_type,
            content=content,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.annotations[doc_name].append(annotation)
        self.annotation_count += 1
        return annotation

    def get_annotations(self, doc_name: str) -> list[DocumentAnnotation]:
        """Get all annotations for a document"""
        return self.annotations.get(doc_name, [])

    def get_annotation(self, annotation_id: str) -> Optional[DocumentAnnotation]:
        """Get a specific annotation by ID"""
        for doc_annotations in self.annotations.values():
            for ann in doc_annotations:
                if ann.id == annotation_id:
                    return ann
        return None

    def update_annotation(self, annotation_id: str, content: str) -> bool:
        """Update an annotation's content"""
        annotation = self.get_annotation(annotation_id)
        if annotation:
            annotation.content = content
            annotation.updated_at = datetime.now()
            return True
        return False

    def resolve_annotation(self, annotation_id: str, user_id: str) -> bool:
        """Mark an annotation as resolved"""
        annotation = self.get_annotation(annotation_id)
        if annotation:
            annotation.resolve(user_id)
            return True
        return False

    def add_reply(self, annotation_id: str, user_id: str, content: str) -> bool:
        """Add a reply to an annotation"""
        annotation = self.get_annotation(annotation_id)
        if annotation:
            annotation.add_reply(user_id, content)
            return True
        return False

    def get_annotation_summary(self, doc_name: str) -> dict:
        """Get summary statistics for annotations on a document"""
        annotations = self.get_annotations(doc_name)
        return {
            "total": len(annotations),
            "resolved": sum(1 for a in annotations if a.resolved),
            "unresolved": sum(1 for a in annotations if not a.resolved),
            "by_type": {
                atype: sum(1 for a in annotations if a.annotation_type == atype)
                for atype in set(a.annotation_type for a in annotations)
            },
            "by_user": {
                user_id: sum(1 for a in annotations if a.user_id == user_id)
                for user_id in set(a.user_id for a in annotations)
            },
        }


class WorkflowManager:
    """Manages review workflows for documents"""

    def __init__(self):
        self.workflows: dict[str, ReviewWorkflow] = {}
        self.workflow_count = 0

    def create_workflow(
        self, doc_name: str, created_by: str, reviewers: list[str]
    ) -> ReviewWorkflow:
        """Create a new review workflow for a document"""
        workflow = ReviewWorkflow(
            id=str(uuid.uuid4()),
            doc_name=doc_name,
            created_by=created_by,
            created_at=datetime.now(),
            status="pending",
            reviewers=reviewers,
            decisions={},
        )
        workflow.add_timeline_event("created", created_by, {"reviewers": reviewers})
        self.workflows[workflow.id] = workflow
        self.workflow_count += 1
        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[ReviewWorkflow]:
        """Get a workflow by ID"""
        return self.workflows.get(workflow_id)

    def get_workflows_for_document(self, doc_name: str) -> list[ReviewWorkflow]:
        """Get all workflows for a document"""
        return [w for w in self.workflows.values() if w.doc_name == doc_name]

    def get_workflows_for_user(self, user_id: str) -> list[ReviewWorkflow]:
        """Get all workflows assigned to a user"""
        return [w for w in self.workflows.values() if user_id in w.reviewers]

    def get_pending_workflows(self) -> list[ReviewWorkflow]:
        """Get all pending workflows"""
        return [w for w in self.workflows.values() if w.status == "pending"]

    def get_workflow_status_summary(self) -> dict:
        """Get summary statistics of all workflows"""
        return {
            "total": len(self.workflows),
            "pending": len(
                [w for w in self.workflows.values() if w.status == "pending"]
            ),
            "in_review": len(
                [w for w in self.workflows.values() if w.status == "in_review"]
            ),
            "approved": len(
                [w for w in self.workflows.values() if w.status == "approved"]
            ),
            "rejected": len(
                [w for w in self.workflows.values() if w.status == "rejected"]
            ),
            "needs_changes": len(
                [w for w in self.workflows.values() if w.status == "needs_changes"]
            ),
        }


class ActivityManager:
    """Manages user sessions and real-time activity tracking"""

    def __init__(self):
        self.sessions: dict[str, UserSession] = {}
        self.activity_log: list[dict] = []
        self.active_timeout_seconds = 300  # 5 minutes

    def create_session(self, user_id: str, username: str) -> UserSession:
        """Create a new user session"""
        session = UserSession(user_id, username)
        self.sessions[user_id] = session
        return session

    def update_activity(self, user_id: str, doc_name: str, action: str = "viewing"):
        """Update user activity"""
        if user_id in self.sessions:
            session = self.sessions[user_id]
            session.update_activity(doc_name, action)
        else:
            session = self.create_session(user_id, user_id)
            session.update_activity(doc_name, action)

    def get_active_users(self, doc_name: str = None) -> list[UserSession]:
        """Get active users, optionally filtered by document"""
        current_time = datetime.now()
        active_sessions = []

        for session in self.sessions.values():
            if session.is_active:
                time_diff = (current_time - session.last_activity).total_seconds()
                if time_diff < self.active_timeout_seconds:
                    if doc_name is None or session.doc_name == doc_name:
                        active_sessions.append(session)

        return active_sessions

    def get_session(self, user_id: str) -> Optional[UserSession]:
        """Get a user's session"""
        return self.sessions.get(user_id)

    def get_user_activity(self, user_id: str) -> dict:
        """Get activity statistics for a user"""
        session = self.get_session(user_id)
        if not session:
            return {"is_active": False}

        return {
            "is_active": session.is_active,
            "current_doc": session.doc_name,
            "current_action": session.current_action,
            "last_activity": session.last_activity.isoformat(),
        }

    def get_document_activity(self, doc_name: str) -> dict:
        """Get activity statistics for a document"""
        active_users = self.get_active_users(doc_name)
        return {
            "active_users": len(active_users),
            "users": [
                {"username": s.username, "action": s.current_action}
                for s in active_users
            ],
            "viewing": len([u for u in active_users if u.current_action == "viewing"]),
            "editing": len([u for u in active_users if u.current_action == "editing"]),
            "reviewing": len(
                [u for u in active_users if u.current_action == "reviewing"]
            ),
            "commenting": len(
                [u for u in active_users if u.current_action == "commenting"]
            ),
        }


class ReviewSystem:
    """Main review system coordinating all components"""

    def __init__(self):
        self.annotation_manager = AnnotationManager()
        self.workflow_manager = WorkflowManager()
        self.activity_manager = ActivityManager()
        self.decision_history: list[dict] = []

    def submit_review_decision(
        self, workflow_id: str, reviewer_id: str, decision: str, comments: str = ""
    ):
        """Submit a review decision with tracking"""
        workflow = self.workflow_manager.get_workflow(workflow_id)
        if workflow:
            workflow.submit_review(reviewer_id, decision, comments)
            self.decision_history.append(
                {
                    "workflow_id": workflow_id,
                    "reviewer_id": reviewer_id,
                    "decision": decision,
                    "comments": comments,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return True
        return False

    def get_review_summary(self, doc_name: str) -> dict:
        """Get comprehensive review summary for a document"""
        workflows = self.workflow_manager.get_workflows_for_document(doc_name)
        annotations = self.annotation_manager.get_annotations(doc_name)
        activity = self.activity_manager.get_document_activity(doc_name)

        return {
            "workflows": len(workflows),
            "active_workflow": workflows[-1].to_dict() if workflows else None,
            "annotations": len(annotations),
            "annotation_summary": self.annotation_manager.get_annotation_summary(
                doc_name
            ),
            "active_users": activity,
            "decision_count": len(
                [
                    d
                    for d in self.decision_history
                    if d.get("workflow_id") in [w.id for w in workflows]
                ]
            ),
        }

    def get_user_review_dashboard(self, user_id: str) -> dict:
        """Get review dashboard for a specific user"""
        pending_workflows = [
            w
            for w in self.workflow_manager.get_workflows_for_user(user_id)
            if w.status in ["pending", "in_review"]
        ]

        my_decisions = [
            d for d in self.decision_history if d.get("reviewer_id") == user_id
        ]

        return {
            "pending_reviews": len(pending_workflows),
            "my_decisions": len(my_decisions),
            "pending_workflows": [w.to_dict() for w in pending_workflows[:5]],
            "recent_decisions": my_decisions[:5],
        }


# ── UI Components ──────────────────────────────────────────────────────────


def render_annotation_ui(review_system: ReviewSystem, doc_name: str, user_id: str):
    """Render document annotation interface"""
    st.subheader(f"📝 Annotations: {doc_name}")

    # Get annotations
    annotations = review_system.annotation_manager.get_annotations(doc_name)

    # Add new annotation
    with st.expander("✏️ Add Comment", expanded=False):
        comment_text = st.text_area("Your comment:", key=f"new_comment_{doc_name}")
        col1, col2 = st.columns(2)
        with col1:
            annotation_type = st.selectbox(
                "Type:", ["comment", "suggestion", "note"], key=f"ann_type_{doc_name}"
            )
        with col2:
            if st.button("Submit Comment", key=f"submit_ann_{doc_name}"):
                if comment_text.strip():
                    review_system.annotation_manager.add_annotation(
                        doc_name, user_id, comment_text, annotation_type
                    )
                    st.success("✅ Comment added!")
                    st.rerun()
                else:
                    st.warning("Please enter some text.")

    # Display annotations
    st.subheader(f"📋 Comments ({len(annotations)})")

    # Filter controls
    show_resolved = st.toggle("Show Resolved", key=f"show_resolved_{doc_name}")
    filter_type = st.selectbox(
        "Filter by Type:",
        ["All", "comment", "suggestion", "note"],
        key=f"filter_type_{doc_name}",
    )

    filtered_annotations = annotations
    if not show_resolved:
        filtered_annotations = [a for a in filtered_annotations if not a.resolved]
    if filter_type != "All":
        filtered_annotations = [
            a for a in filtered_annotations if a.annotation_type == filter_type
        ]

    # Display each annotation
    for ann in filtered_annotations:
        with st.expander(
            f"{ann.annotation_type.capitalize()} by {ann.user_id} - "
            f"{ann.created_at.strftime('%Y-%m-%d %H:%M')} "
            f"{'✅ Resolved' if ann.resolved else '🔄 Open'}",
            expanded=not ann.resolved,
        ):
            st.markdown(f"**Content:** {ann.content}")

            if ann.metadata:
                st.caption(f"Metadata: {ann.metadata}")

            # Show replies
            if ann.replies:
                st.markdown("**Replies:**")
                for reply in ann.replies:
                    st.markdown(
                        f"- {reply['user_id']}: {reply['content']} "
                        f"({reply['created_at']})"
                    )

            # Reply form
            reply_text = st.text_input("Add reply:", key=f"reply_{ann.id}_{doc_name}")
            if st.button("Reply", key=f"reply_btn_{ann.id}_{doc_name}"):
                if reply_text.strip():
                    review_system.annotation_manager.add_reply(
                        ann.id, user_id, reply_text
                    )
                    st.success("Reply added!")
                    st.rerun()

            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if not ann.resolved:
                    if st.button("✅ Resolve", key=f"resolve_{ann.id}_{doc_name}"):
                        review_system.annotation_manager.resolve_annotation(
                            ann.id, user_id
                        )
                        st.rerun()
            with col2:
                if user_id == ann.user_id:
                    if st.button("🗑️ Delete", key=f"delete_{ann.id}_{doc_name}"):
                        # Remove annotation from list
                        st.warning("Delete functionality pending")

    # Annotation statistics
    summary = review_system.annotation_manager.get_annotation_summary(doc_name)
    st.caption(
        f"Total: {summary['total']} | "
        f"Resolved: {summary['resolved']} | "
        f"Unresolved: {summary['unresolved']}"
    )


def render_workflow_ui(review_system: ReviewSystem, doc_name: str, user_id: str):
    """Render review workflow interface"""
    st.subheader(f"🔄 Review Workflow: {doc_name}")

    # Get existing workflows
    workflows = review_system.workflow_manager.get_workflows_for_document(doc_name)

    # Create new workflow
    if not workflows:
        with st.expander("📋 Create Review Workflow", expanded=True):
            st.info("No review workflow exists for this document.")

            all_users = ["admin1", "reviewer1", "reviewer2"]  # Should fetch from system

            selected_reviewers = st.multiselect(
                "Select Reviewers:",
                options=all_users,
                default=all_users[1:] if len(all_users) > 1 else [],
            )

            if st.button("Create Workflow", key=f"create_wf_{doc_name}"):
                if selected_reviewers:
                    workflow = review_system.workflow_manager.create_workflow(
                        doc_name, user_id, selected_reviewers
                    )
                    st.success(
                        f"✅ Workflow created with {len(selected_reviewers)} reviewers!"
                    )
                    st.rerun()
                else:
                    st.warning("Please select at least one reviewer.")

    # Display workflows
    if workflows:
        for workflow in workflows:
            with st.expander(
                f"Workflow #{workflow.id[:8]} - {workflow.status.upper()} "
                f"({len(workflow.decisions)}/{len(workflow.reviewers)} reviews)",
                expanded=True,
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", workflow.status.title())
                with col2:
                    st.metric(
                        "Reviewers",
                        f"{len(workflow.decisions)}/{len(workflow.reviewers)}",
                    )
                with col3:
                    st.metric("Created", workflow.created_at.strftime("%Y-%m-%d %H:%M"))

                # Show reviewers and their decisions
                st.markdown("**Reviewers:**")
                for reviewer in workflow.reviewers:
                    status = "⏳ Pending"
                    if reviewer in workflow.decisions:
                        decision = workflow.decisions[reviewer]["decision"]
                        status = (
                            f"✅ {decision.title()}"
                            if decision == "approved"
                            else f"❌ {decision.title()}"
                        )
                    st.markdown(f"- {reviewer}: {status}")

                # If user is a reviewer and hasn't reviewed
                if user_id in workflow.reviewers and user_id not in workflow.decisions:
                    st.markdown("---")
                    st.markdown("**Submit Your Review:**")

                    decision = st.radio(
                        "Decision:",
                        ["approved", "rejected", "needs_changes"],
                        key=f"decision_{workflow.id}",
                    )

                    comments = st.text_area(
                        "Comments:",
                        key=f"comments_{workflow.id}",
                        placeholder="Provide feedback...",
                    )

                    if st.button("Submit Review", key=f"submit_{workflow.id}"):
                        review_system.submit_review_decision(
                            workflow.id, user_id, decision, comments
                        )
                        st.success("✅ Review submitted!")
                        st.rerun()

                # Show timeline
                if workflow.timeline:
                    with st.expander("📋 Timeline", expanded=False):
                        for event in workflow.timeline:
                            st.markdown(
                                f"- {event['timestamp']}: {event['event_type']} "
                                f"by {event['user_id']} - {event['details']}"
                            )


def render_review_dashboard(review_system: ReviewSystem, user_id: str):
    """Render the main review dashboard"""
    st.subheader("📊 Review Dashboard")

    # Get user dashboard data
    dashboard = review_system.get_user_review_dashboard(user_id)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Pending Reviews", dashboard["pending_reviews"])
    col2.metric("Your Decisions", dashboard["my_decisions"])
    col3.metric("Total Workflows", review_system.workflow_manager.workflow_count)

    # Summary statistics
    summary = review_system.workflow_manager.get_workflow_status_summary()
    st.subheader("📈 Workflow Status Summary")

    status_data = {
        "Status": ["Pending", "In Review", "Approved", "Rejected", "Needs Changes"],
        "Count": [
            summary["pending"],
            summary["in_review"],
            summary["approved"],
            summary["rejected"],
            summary["needs_changes"],
        ],
    }
    status_df = pd.DataFrame(status_data)
    st.bar_chart(status_df.set_index("Status"))

    # Pending workflows
    if dashboard["pending_workflows"]:
        st.subheader("⏳ Pending Reviews")
        for wf in dashboard["pending_workflows"]:
            with st.expander(f"📄 {wf['doc_name']} - {wf['status'].upper()}"):
                st.markdown(f"**Created:** {wf['created_at']}")
                st.markdown(f"**Reviewers:** {', '.join(wf['reviewers'])}")
                st.markdown(
                    f"**Decisions:** {len(wf['decisions'])}/{len(wf['reviewers'])}"
                )

                if st.button("Review Now", key=f"review_now_{wf['id']}"):
                    st.session_state["selected_workflow"] = wf["id"]
                    st.rerun()

    # Recent decisions
    if dashboard["recent_decisions"]:
        st.subheader("📋 Your Recent Decisions")
        for decision in dashboard["recent_decisions"]:
            st.markdown(
                f"- {decision['timestamp']}: {decision['decision'].upper()} "
                f"- {decision.get('comments', 'No comments')}"
            )


def render_activity_ui(review_system: ReviewSystem, doc_name: str):
    """Render real-time activity UI"""
    st.subheader("👥 Real-time Activity")

    activity = review_system.activity_manager.get_document_activity(doc_name)

    # Activity metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Users", activity["active_users"])
    col2.metric("Viewing", activity["viewing"])
    col3.metric("Editing", activity["editing"])
    col4.metric("Reviewing", activity["reviewing"])

    # Active users
    if activity["users"]:
        st.markdown("**Active Users:**")
        for user in activity["users"]:
            icon = (
                "👁️"
                if user["action"] == "viewing"
                else "✏️"
                if user["action"] == "editing"
                else "📝"
            )
            st.markdown(f"- {icon} {user['username']} ({user['action']})")


def render_decision_history(review_system: ReviewSystem, doc_name: str):
    """Render decision history for a document"""
    st.subheader("📋 Decision History")

    # Get decisions for this document
    workflows = review_system.workflow_manager.get_workflows_for_document(doc_name)

    decisions = []
    for wf in workflows:
        for reviewer, decision in wf.decisions.items():
            decisions.append(
                {
                    "workflow": wf.id[:8],
                    "reviewer": reviewer,
                    "decision": decision["decision"],
                    "comments": decision.get("comments", ""),
                    "timestamp": decision["timestamp"],
                }
            )

    if decisions:
        df = pd.DataFrame(decisions)
        st.dataframe(df, use_container_width=True)

        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export Decisions as CSV",
            data=csv,
            file_name=f"{doc_name}_decisions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No decisions recorded for this document.")


def initialize_review_system():
    """Initialize the review system in session state"""
    if "review_system" not in st.session_state:
        st.session_state["review_system"] = ReviewSystem()

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = st.session_state.get("username", "unknown_user")


def render_collaboration_dashboard():
    """Render the main collaboration dashboard"""
    initialize_review_system()

    review_system = st.session_state["review_system"]
    user_id = st.session_state["user_id"]

    # Get document name from session or context
    doc_name = st.session_state.get("current_doc_name", "all_documents")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Annotations", "🔄 Workflow", "👥 Activity", "📊 Dashboard"]
    )

    with tab1:
        render_annotation_ui(review_system, doc_name, user_id)

    with tab2:
        render_workflow_ui(review_system, doc_name, user_id)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            render_activity_ui(review_system, doc_name)
        with col2:
            render_decision_history(review_system, doc_name)

    with tab4:
        render_review_dashboard(review_system, user_id)


# ── End of Collaboration System ────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
