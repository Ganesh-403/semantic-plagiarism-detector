"""
Workflow Automation and Process Orchestration

Features:
- Process orchestration for complex workflows
- Task scheduling and automation
- Dependency management
- Event-driven actions
- Custom workflow creation
- Workflow templates
- Approval workflows
- External system integration
"""

import hashlib
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class WorkflowStatus(Enum):
    """Workflow status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    APPROVAL_PENDING = "approval_pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TaskStatus(Enum):
    """Task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TriggerType(Enum):
    """Trigger types."""

    SCHEDULED = "scheduled"
    EVENT = "event"
    MANUAL = "manual"
    API = "api"
    DEPENDENCY = "dependency"


class WorkflowCategory(Enum):
    """Workflow categories."""

    PLAGIARISM_DETECTION = "plagiarism_detection"
    DOCUMENT_PROCESSING = "document_processing"
    REPORT_GENERATION = "report_generation"
    APPROVAL = "approval"
    NOTIFICATION = "notification"
    INTEGRATION = "integration"
    MAINTENANCE = "maintenance"


@dataclass
class Workflow:
    """Workflow definition."""

    id: str
    name: str
    description: str
    category: WorkflowCategory
    status: WorkflowStatus
    tasks: list[dict[str, Any]]
    triggers: list[dict[str, Any]]
    created_at: float
    created_by: str
    updated_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_count: int = 0
    last_execution: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution record."""

    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: float
    completed_at: Optional[float] = None
    task_results: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Task definition."""

    id: str
    name: str
    description: str
    function: str
    parameters: dict[str, Any]
    dependencies: list[str]
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRequest:
    """Approval request."""

    id: str
    workflow_id: str
    task_id: str
    requester: str
    approvers: list[str]
    status: WorkflowStatus
    created_at: float
    responded_at: Optional[float] = None
    response: Optional[str] = None
    comments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTemplate:
    """Workflow template."""

    id: str
    name: str
    description: str
    category: WorkflowCategory
    tasks: list[dict[str, Any]]
    triggers: list[dict[str, Any]]
    created_at: float
    created_by: str
    is_active: bool = True
    usage_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# WORKFLOW ENGINE
# ==============================================================================


class WorkflowEngine:
    """
    Core workflow orchestration engine.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.workflows: dict[str, Workflow] = {}
        self.executions: list[WorkflowExecution] = []
        self.templates: list[WorkflowTemplate] = []
        self.approvals: list[ApprovalRequest] = []
        self.task_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.task_handlers: dict[str, Callable] = {}
        self._load_data()
        self._register_default_handlers()
        self._start_worker()

    def _load_data(self):
        """Load data from storage."""
        try:
            data_path = self.storage_path / "workflow_data.json"
            if data_path.exists():
                with open(data_path, "r") as f:
                    data = json.load(f)

                    self.workflows = {
                        k: Workflow(**v) for k, v in data.get("workflows", {}).items()
                    }

                    self.executions = [
                        WorkflowExecution(**e) for e in data.get("executions", [])
                    ]

                    self.templates = [
                        WorkflowTemplate(**t) for t in data.get("templates", [])
                    ]

                    self.approvals = [
                        ApprovalRequest(**a) for a in data.get("approvals", [])
                    ]
        except Exception as e:
            print(f"Error loading workflow data: {e}")

    def _save_data(self):
        """Save data to storage."""
        try:
            data_path = self.storage_path / "workflow_data.json"
            data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "workflows": {k: asdict(v) for k, v in self.workflows.items()},
                "executions": [asdict(e) for e in self.executions[-1000:]],
                "templates": [asdict(t) for t in self.templates],
                "approvals": [asdict(a) for a in self.approvals[-500:]],
            }

            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving workflow data: {e}")

    def _register_default_handlers(self):
        """Register default task handlers."""
        self.task_handlers = {
            "plagiarism_check": self._handle_plagiarism_check,
            "document_processing": self._handle_document_processing,
            "report_generation": self._handle_report_generation,
            "notification": self._handle_notification,
            "approval": self._handle_approval,
            "export": self._handle_export,
            "cleanup": self._handle_cleanup,
        }

    def _start_worker(self):
        """Start background worker."""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Worker thread for processing tasks."""
        while self.is_running:
            try:
                # Get task from queue
                task_data = self.task_queue.get(timeout=1)

                if task_data is None:
                    continue

                workflow_id = task_data.get("workflow_id")
                task_id = task_data.get("task_id")

                if workflow_id and task_id:
                    self._process_task(workflow_id, task_id)

                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(5)

    def _process_task(self, workflow_id: str, task_id: str):
        """Process a specific task."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return

        # Find task
        task_data = None
        for task in workflow.tasks:
            if task.get("id") == task_id:
                task_data = task
                break

        if not task_data:
            return

        # Check dependencies
        if not self._check_dependencies(workflow, task_data):
            task_data["status"] = TaskStatus.BLOCKED.value
            self._save_data()
            return

        # Update status
        task_data["status"] = TaskStatus.RUNNING.value
        task_data["started_at"] = time.time()
        workflow.status = WorkflowStatus.RUNNING
        self._save_data()

        try:
            # Execute task
            handler = self.task_handlers.get(task_data.get("function"))
            if handler:
                result = handler(task_data, workflow)
                task_data["result"] = result
                task_data["status"] = TaskStatus.COMPLETED.value
            else:
                raise ValueError(
                    f"No handler for function: {task_data.get('function')}"
                )

        except Exception as e:
            task_data["error"] = str(e)
            task_data["status"] = TaskStatus.FAILED.value

            # Check if should retry
            retry_count = task_data.get("retry_count", 0)
            max_retries = task_data.get("max_retries", 3)

            if retry_count < max_retries:
                task_data["retry_count"] = retry_count + 1
                task_data["status"] = TaskStatus.RETRYING.value

                # Re-queue with delay
                time.sleep(2**retry_count)  # Exponential backoff
                self.task_queue.put({"workflow_id": workflow_id, "task_id": task_id})

        finally:
            task_data["completed_at"] = time.time()
            self._save_data()

            # Check if workflow is complete
            self._check_workflow_completion(workflow_id)

    def _check_dependencies(self, workflow: Workflow, task: Dict) -> bool:
        """Check if task dependencies are met."""
        dependencies = task.get("dependencies", [])

        if not dependencies:
            return True

        # Find dependent tasks
        dependent_tasks = []
        for t in workflow.tasks:
            if t.get("id") in dependencies:
                dependent_tasks.append(t)

        # Check if all dependencies are completed
        for dep in dependent_tasks:
            if dep.get("status") != TaskStatus.COMPLETED.value:
                return False

        return True

    def _check_workflow_completion(self, workflow_id: str):
        """Check if workflow is complete."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return

        # Check all tasks
        all_completed = True
        any_failed = False

        for task in workflow.tasks:
            status = task.get("status")
            if status == TaskStatus.FAILED.value:
                any_failed = True
            elif status not in [TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value]:
                all_completed = False

        if any_failed:
            workflow.status = WorkflowStatus.FAILED
        elif all_completed:
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = time.time()

        workflow.execution_count += 1
        workflow.last_execution = time.time()
        self._save_data()

    # ==========================================================================
    # TASK HANDLERS
    # ==========================================================================

    def _handle_plagiarism_check(self, task: Dict, workflow: Workflow) -> Dict:
        """Handle plagiarism check task."""
        params = task.get("parameters", {})
        document_ids = params.get("document_ids", [])
        threshold = params.get("threshold", 0.75)

        # Simulate plagiarism check
        time.sleep(2)

        return {
            "status": "completed",
            "documents_checked": len(document_ids),
            "flagged": len(document_ids) // 3,
            "threshold_used": threshold,
            "timestamp": time.time(),
        }

    def _handle_document_processing(self, task: Dict, workflow: Workflow) -> Dict:
        """Handle document processing task."""
        params = task.get("parameters", {})
        documents = params.get("documents", [])

        # Simulate processing
        time.sleep(1)

        return {
            "status": "completed",
            "documents_processed": len(documents),
            "processing_time": len(documents) * 0.5,
            "timestamp": time.time(),
        }

    def _handle_report_generation(self, task: Dict, workflow: Workflow) -> Dict:
        """Handle report generation task."""
        params = task.get("parameters", {})
        report_type = params.get("type", "standard")

        # Simulate report generation
        time.sleep(1.5)

        return {
            "status": "completed",
            "report_type": report_type,
            "report_id": f"report_{int(time.time())}",
            "timestamp": time.time(),
        }

    def _handle_notification(self, task: Dict, workflow: Workflow) -> Dict:
        """Handle notification task."""
        params = task.get("parameters", {})
        recipients = params.get("recipients", [])
        message = params.get("message", "")

        # Simulate notification
        time.sleep(0.5)

        return {
            "status": "completed",
            "recipients_notified": len(recipients),
            "message_sent": message[:50] + "..." if len(message) > 50 else message,
            "timestamp": time.time(),
        }

    def _handle_approval(self, task: Dict, workflow: Workflow) -> Dict:
        """Handle approval task."""
        params = task.get("parameters", {})
        approvers = params.get("approvers", [])

        # Create approval request
        approval = ApprovalRequest(
            id=f"app_{int(time.time())}_{hashlib.md5(str(workflow.id).encode()).hexdigest()[:8]}",
            workflow_id=workflow.id,
            task_id=task.get("id"),
            requester=workflow.created_by,
            approvers=approvers,
            status=WorkflowStatus.APPROVAL_PENDING,
            created_at=time.time(),
            metadata=params.get("metadata", {}),
        )

        self.approvals.append(approval)
        self._save_data()

        # Wait for approval (simulated)
        timeout = params.get("timeout", 300)
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check approval status
            approval = next((a for a in self.approvals if a.id == approval.id), None)
            if approval and approval.status in [
                WorkflowStatus.APPROVED,
                WorkflowStatus.REJECTED,
            ]:
                break
            time.sleep(5)

        return {
            "status": "completed",
            "approval_id": approval.id,
            "approval_status": approval.status.value,
            "timestamp": time.time(),
        }

    def _handle_export(self, task: Dict, workflow: Workflow) -> Dict:
        """Handle export task."""
        params = task.get("parameters", {})
        format = params.get("format", "csv")
        data = params.get("data", {})

        # Simulate export
        time.sleep(1)

        return {
            "status": "completed",
            "format": format,
            "export_id": f"export_{int(time.time())}",
            "timestamp": time.time(),
        }

    def _handle_cleanup(self, task: Dict, workflow: Workflow) -> Dict:
        """Handle cleanup task."""
        params = task.get("parameters", {})
        cleanup_type = params.get("type", "temp_files")

        # Simulate cleanup
        time.sleep(0.5)

        return {
            "status": "completed",
            "cleanup_type": cleanup_type,
            "files_removed": 5,
            "space_freed_mb": 10.5,
            "timestamp": time.time(),
        }

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================

    def create_workflow(
        self,
        name: str,
        description: str,
        category: WorkflowCategory,
        tasks: List[Dict[str, Any]],
        triggers: List[Dict[str, Any]],
        created_by: str,
    ) -> Workflow:
        """Create a new workflow."""
        workflow = Workflow(
            id=f"wf_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            description=description,
            category=category,
            status=WorkflowStatus.PENDING,
            tasks=tasks,
            triggers=triggers,
            created_at=time.time(),
            created_by=created_by,
        )

        # Set initial task statuses
        for task in workflow.tasks:
            task["status"] = TaskStatus.PENDING.value
            task["created_at"] = time.time()

        self.workflows[workflow.id] = workflow
        self._save_data()

        return workflow

    def create_from_template(
        self, template_id: str, name: str, parameters: Dict[str, Any], created_by: str
    ) -> Optional[Workflow]:
        """Create workflow from template."""
        template = next((t for t in self.templates if t.id == template_id), None)
        if not template:
            return None

        # Clone template tasks
        tasks = []
        for task in template.tasks:
            task_copy = task.copy()
            # Apply parameters
            for key, value in parameters.items():
                if key in task_copy.get("parameters", {}):
                    task_copy["parameters"][key] = value
            tasks.append(task_copy)

        return self.create_workflow(
            name=name or template.name,
            description=template.description,
            category=template.category,
            tasks=tasks,
            triggers=template.triggers.copy(),
            created_by=created_by,
        )

    def execute_workflow(self, workflow_id: str) -> Optional[str]:
        """Execute a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None

        if workflow.status in [WorkflowStatus.RUNNING, WorkflowStatus.COMPLETED]:
            return None

        # Create execution record
        execution = WorkflowExecution(
            id=f"exec_{int(time.time())}_{workflow_id[:8]}",
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            started_at=time.time(),
        )

        self.executions.append(execution)

        # Reset task statuses
        for task in workflow.tasks:
            task["status"] = TaskStatus.PENDING.value
            task["started_at"] = None
            task["completed_at"] = None
            task["result"] = None
            task["error"] = None

        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = time.time()
        self._save_data()

        # Queue tasks
        for task in workflow.tasks:
            # Check if task has dependencies
            dependencies = task.get("dependencies", [])
            if dependencies:
                # Check if dependencies are met
                if self._check_dependencies(workflow, task):
                    self.task_queue.put(
                        {"workflow_id": workflow_id, "task_id": task.get("id")}
                    )
            else:
                self.task_queue.put(
                    {"workflow_id": workflow_id, "task_id": task.get("id")}
                )

        return execution.id

    def pause_workflow(self, workflow_id: str):
        """Pause a running workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow and workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            self._save_data()

    def resume_workflow(self, workflow_id: str):
        """Resume a paused workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow and workflow.status == WorkflowStatus.PAUSED:
            workflow.status = WorkflowStatus.RUNNING
            self._save_data()

            # Re-queue pending tasks
            for task in workflow.tasks:
                if task.get("status") == TaskStatus.PENDING.value:
                    self.task_queue.put(
                        {"workflow_id": workflow_id, "task_id": task.get("id")}
                    )

    def cancel_workflow(self, workflow_id: str):
        """Cancel a workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow and workflow.status in [
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
            WorkflowStatus.PAUSED,
        ]:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = time.time()

            # Mark pending tasks as cancelled
            for task in workflow.tasks:
                if task.get("status") in [
                    TaskStatus.PENDING.value,
                    TaskStatus.RUNNING.value,
                ]:
                    task["status"] = TaskStatus.SKIPPED.value

            self._save_data()

    def create_template(
        self,
        name: str,
        description: str,
        category: WorkflowCategory,
        tasks: List[Dict[str, Any]],
        triggers: List[Dict[str, Any]],
        created_by: str,
    ) -> WorkflowTemplate:
        """Create a workflow template."""
        template = WorkflowTemplate(
            id=f"tpl_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            description=description,
            category=category,
            tasks=tasks,
            triggers=triggers,
            created_at=time.time(),
            created_by=created_by,
        )

        self.templates.append(template)
        self._save_data()

        return template

    def approve_approval(self, approval_id: str, approver: str, comment: str = ""):
        """Approve an approval request."""
        approval = next((a for a in self.approvals if a.id == approval_id), None)
        if approval:
            approval.status = WorkflowStatus.APPROVED
            approval.responded_at = time.time()
            approval.response = "approved"
            approval.comments.append(
                {
                    "timestamp": time.time(),
                    "user": approver,
                    "comment": comment or "Approved",
                }
            )

            # Resume workflow
            self.resume_workflow(approval.workflow_id)

            self._save_data()

    def reject_approval(self, approval_id: str, approver: str, comment: str = ""):
        """Reject an approval request."""
        approval = next((a for a in self.approvals if a.id == approval_id), None)
        if approval:
            approval.status = WorkflowStatus.REJECTED
            approval.responded_at = time.time()
            approval.response = "rejected"
            approval.comments.append(
                {
                    "timestamp": time.time(),
                    "user": approver,
                    "comment": comment or "Rejected",
                }
            )

            # Cancel workflow
            self.cancel_workflow(approval.workflow_id)

            self._save_data()

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow status."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None

        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": self._calculate_progress(workflow),
            "tasks": workflow.tasks,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "execution_count": workflow.execution_count,
        }

    def _calculate_progress(self, workflow: Workflow) -> float:
        """Calculate workflow progress."""
        total_tasks = len(workflow.tasks)
        if total_tasks == 0:
            return 0.0

        completed_tasks = len(
            [
                t
                for t in workflow.tasks
                if t.get("status")
                in [TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value]
            ]
        )

        return completed_tasks / total_tasks

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        total_workflows = len(self.workflows)
        running = len(
            [w for w in self.workflows.values() if w.status == WorkflowStatus.RUNNING]
        )
        completed = len(
            [w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED]
        )
        failed = len(
            [w for w in self.workflows.values() if w.status == WorkflowStatus.FAILED]
        )

        total_executions = len(self.executions)
        total_approvals = len(self.approvals)
        pending_approvals = len(
            [a for a in self.approvals if a.status == WorkflowStatus.APPROVAL_PENDING]
        )

        return {
            "total_workflows": total_workflows,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total_executions": total_executions,
            "total_approvals": total_approvals,
            "pending_approvals": pending_approvals,
            "success_rate": completed / total_workflows if total_workflows > 0 else 0,
        }


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_workflow_automation():
    """Render workflow automation UI."""
    st.subheader("🔄 Workflow Automation")

    # Initialize
    if "workflow_engine" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        st.session_state.workflow_engine = WorkflowEngine(data_dir / "workflows")

    engine = st.session_state.workflow_engine

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "📋 Workflows", "📝 Templates", "✅ Approvals", "⚙️ Create"]
    )

    with tab1:
        render_workflow_dashboard(engine)

    with tab2:
        render_workflow_list(engine)

    with tab3:
        render_template_management(engine)

    with tab4:
        render_approval_management(engine)

    with tab5:
        render_workflow_creation(engine)


def render_workflow_dashboard(engine: WorkflowEngine):
    """Render workflow dashboard."""
    st.markdown("#### 📊 Workflow Dashboard")

    stats = engine.get_workflow_stats()

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Workflows", stats["total_workflows"])
    col2.metric("Running", stats["running"])
    col3.metric("Completed", stats["completed"])
    col4.metric("Failed", stats["failed"])
    col5.metric("Success Rate", f"{stats['success_rate']:.1%}")

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Executions", stats["total_executions"])
    col2.metric("Total Approvals", stats["total_approvals"])
    col3.metric("Pending Approvals", stats["pending_approvals"])

    # Chart: Workflow status distribution
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Workflow Status", "Approval Status")
    )

    # Workflow status
    status_counts = {
        "running": stats["running"],
        "completed": stats["completed"],
        "failed": stats["failed"],
        "pending": stats["total_workflows"]
        - stats["running"]
        - stats["completed"]
        - stats["failed"],
    }

    fig.add_trace(
        go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.3,
        ),
        row=1,
        col=1,
    )

    # Approval status
    approval_counts = {
        "pending": stats["pending_approvals"],
        "approved": stats["total_approvals"] - stats["pending_approvals"],
    }

    fig.add_trace(
        go.Pie(
            labels=list(approval_counts.keys()),
            values=list(approval_counts.values()),
            hole=0.3,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_workflow_list(engine: WorkflowEngine):
    """Render workflow list."""
    st.markdown("#### 📋 Workflows")

    if not engine.workflows:
        st.info("No workflows created")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status", ["All"] + [s.value for s in WorkflowStatus]
        )
    with col2:
        category_filter = st.selectbox(
            "Filter by Category", ["All"] + [c.value for c in WorkflowCategory]
        )

    # Get filtered workflows
    workflows = list(engine.workflows.values())

    if status_filter != "All":
        workflows = [w for w in workflows if w.status.value == status_filter]

    if category_filter != "All":
        workflows = [w for w in workflows if w.category.value == category_filter]

    # Display workflows
    for workflow in workflows:
        status_colors = {
            "completed": "🟢",
            "running": "🟠",
            "pending": "⚪",
            "failed": "🔴",
            "paused": "🟡",
            "cancelled": "⚫",
            "approval_pending": "🔵",
            "approved": "🟢",
            "rejected": "🔴",
        }

        with st.expander(
            f"{status_colors.get(workflow.status.value, '')} {workflow.name} - {workflow.status.value.upper()}",
            expanded=workflow.status in [WorkflowStatus.RUNNING, WorkflowStatus.FAILED],
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Description:** {workflow.description}")
                st.markdown(f"**Category:** {workflow.category.value}")
                st.caption(
                    f"Created: {datetime.fromtimestamp(workflow.created_at).strftime('%Y-%m-%d %H:%M')}"
                )
                st.caption(f"Created by: {workflow.created_by}")
                st.caption(f"Executions: {workflow.execution_count}")

                # Progress
                progress = engine._calculate_progress(workflow)
                st.progress(progress, text=f"Progress: {progress:.1%}")

            with col2:
                if workflow.status == WorkflowStatus.PENDING:
                    if st.button("▶️ Execute", key=f"exec_{workflow.id}"):
                        engine.execute_workflow(workflow.id)
                        st.rerun()

                if workflow.status == WorkflowStatus.RUNNING:
                    if st.button("⏸️ Pause", key=f"pause_{workflow.id}"):
                        engine.pause_workflow(workflow.id)
                        st.rerun()

                if workflow.status == WorkflowStatus.PAUSED:
                    if st.button("▶️ Resume", key=f"resume_{workflow.id}"):
                        engine.resume_workflow(workflow.id)
                        st.rerun()

                if workflow.status in [
                    WorkflowStatus.PENDING,
                    WorkflowStatus.RUNNING,
                    WorkflowStatus.PAUSED,
                ]:
                    if st.button("⏹️ Cancel", key=f"cancel_{workflow.id}"):
                        engine.cancel_workflow(workflow.id)
                        st.rerun()

            # Task list
            if workflow.tasks:
                st.markdown("**Tasks:**")
                for task in workflow.tasks:
                    status_icons = {
                        "pending": "⏳",
                        "running": "🔄",
                        "completed": "✅",
                        "failed": "❌",
                        "blocked": "🚫",
                        "skipped": "⏭️",
                        "retrying": "🔄",
                    }
                    icon = status_icons.get(task.get("status"), "⚪")
                    st.caption(
                        f"{icon} {task.get('name')} - {task.get('status', 'unknown')}"
                    )


def render_template_management(engine: WorkflowEngine):
    """Render template management UI."""
    st.markdown("#### 📝 Workflow Templates")

    if engine.templates:
        for template in engine.templates:
            with st.expander(f"📄 {template.name}", expanded=False):
                st.markdown(f"**Description:** {template.description}")
                st.markdown(f"**Category:** {template.category.value}")
                st.markdown(f"**Tasks:** {len(template.tasks)}")
                st.caption(
                    f"Created: {datetime.fromtimestamp(template.created_at).strftime('%Y-%m-%d')}"
                )
                st.caption(f"Usage: {template.usage_count} times")

                if st.button("📋 Use Template", key=f"use_{template.id}"):
                    st.session_state.template_to_use = template.id
                    st.rerun()
    else:
        st.info("No templates created")

    # Create from template
    if hasattr(st.session_state, "template_to_use"):
        template_id = st.session_state.template_to_use
        template = next((t for t in engine.templates if t.id == template_id), None)

        if template:
            st.markdown(f"#### Create Workflow from '{template.name}'")

            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input(
                    "Workflow Name",
                    f"{template.name} - {datetime.now().strftime('%Y-%m-%d')}",
                )
            with col2:
                created_by = st.text_input(
                    "Created By", st.session_state.get("username", "system")
                )

            if st.button("✅ Create Workflow", use_container_width=True):
                workflow = engine.create_from_template(
                    template_id=template_id,
                    name=name,
                    parameters={},
                    created_by=created_by,
                )
                if workflow:
                    st.success(f"✅ Workflow created: {workflow.id}")
                    del st.session_state.template_to_use
                    st.rerun()
                else:
                    st.error("Failed to create workflow")


def render_approval_management(engine: WorkflowEngine):
    """Render approval management UI."""
    st.markdown("#### ✅ Approvals")

    approvals = engine.approvals

    if not approvals:
        st.info("No approval requests")
        return

    # Filters
    status_filter = st.selectbox(
        "Filter by Status", ["All"] + [s.value for s in WorkflowStatus]
    )

    filtered = approvals
    if status_filter != "All":
        filtered = [a for a in approvals if a.status.value == status_filter]

    for approval in filtered[-20:]:
        status_colors = {"approval_pending": "🟡", "approved": "🟢", "rejected": "🔴"}

        with st.expander(
            f"{status_colors.get(approval.status.value, '')} Approval Request - {approval.workflow_id[:8]}",
            expanded=approval.status == WorkflowStatus.APPROVAL_PENDING,
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Requester:** {approval.requester}")
                st.markdown(f"**Approvers:** {', '.join(approval.approvers)}")
                st.caption(
                    f"Created: {datetime.fromtimestamp(approval.created_at).strftime('%Y-%m-%d %H:%M')}"
                )

                if approval.comments:
                    st.markdown("**Comments:**")
                    for comment in approval.comments[-5:]:
                        st.caption(f"{comment['user']}: {comment['comment']}")

            with col2:
                if approval.status == WorkflowStatus.APPROVAL_PENDING:
                    user = st.session_state.get("username", "system")

                    if user in approval.approvers:
                        comment = st.text_area(
                            "Comment",
                            key=f"comment_{approval.id}",
                            placeholder="Add comment...",
                        )

                        if st.button("✅ Approve", key=f"approve_{approval.id}"):
                            engine.approve_approval(approval.id, user, comment)
                            st.rerun()

                        if st.button("❌ Reject", key=f"reject_{approval.id}"):
                            engine.reject_approval(approval.id, user, comment)
                            st.rerun()
                    else:
                        st.caption("Waiting for approver response")


def render_workflow_creation(engine: WorkflowEngine):
    """Render workflow creation UI."""
    st.markdown("#### ⚙️ Create Workflow")

    with st.form("create_workflow_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Workflow Name", "Plagiarism Detection Workflow")
            category = st.selectbox("Category", [c.value for c in WorkflowCategory])
            created_by = st.text_input(
                "Created By", st.session_state.get("username", "system")
            )

        with col2:
            description = st.text_area(
                "Description", "Automated plagiarism detection workflow"
            )

        # Tasks
        st.markdown("#### 📋 Tasks")

        num_tasks = st.number_input("Number of Tasks", 1, 10, 3)

        tasks = []
        for i in range(num_tasks):
            st.markdown(f"**Task {i + 1}**")
            col1, col2, col3 = st.columns(3)

            with col1:
                task_name = st.text_input(
                    "Task Name", f"Task {i + 1}", key=f"task_name_{i}"
                )
                task_function = st.selectbox(
                    "Function",
                    [
                        "plagiarism_check",
                        "document_processing",
                        "report_generation",
                        "notification",
                        "approval",
                        "export",
                        "cleanup",
                    ],
                    key=f"task_func_{i}",
                )

            with col2:
                task_description = st.text_input(
                    "Description", "", key=f"task_desc_{i}"
                )
                task_timeout = st.number_input(
                    "Timeout (seconds)", 60, 3600, 300, key=f"task_timeout_{i}"
                )

            with col3:
                task_max_retries = st.number_input(
                    "Max Retries", 0, 5, 3, key=f"task_retry_{i}"
                )
                dependencies = st.text_input(
                    "Dependencies (task IDs, comma separated)", "", key=f"task_dep_{i}"
                )

            tasks.append(
                {
                    "id": f"task_{i}_{int(time.time())}",
                    "name": task_name,
                    "description": task_description,
                    "function": task_function,
                    "parameters": {},
                    "dependencies": [
                        d.strip() for d in dependencies.split(",") if d.strip()
                    ],
                    "timeout_seconds": task_timeout,
                    "max_retries": task_max_retries,
                }
            )

        # Triggers
        st.markdown("#### ⚡ Triggers")
        trigger_type = st.selectbox("Trigger Type", [t.value for t in TriggerType])
        trigger_config = st.text_area(
            "Trigger Configuration (JSON)", '{"schedule": "daily"}'
        )

        try:
            trigger_config_json = json.loads(trigger_config) if trigger_config else {}
        except:
            trigger_config_json = {}

        triggers = [{"type": trigger_type, "config": trigger_config_json}]

        # Submit
        if st.form_submit_button("🚀 Create Workflow", use_container_width=True):
            try:
                workflow = engine.create_workflow(
                    name=name,
                    description=description,
                    category=WorkflowCategory(category),
                    tasks=tasks,
                    triggers=triggers,
                    created_by=created_by,
                )

                st.success(f"✅ Workflow created: {workflow.id}")

                # Option to execute immediately
                if st.button("▶️ Execute Now", use_container_width=True):
                    engine.execute_workflow(workflow.id)
                    st.rerun()

            except Exception as e:
                st.error(f"Failed to create workflow: {e}")


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_workflow_automation():
    """Initialize workflow automation system."""
    if "workflow_automation_initialized" not in st.session_state:
        st.session_state.workflow_automation_initialized = True

        data_dir = Path(st.session_state.get("data_dir", "."))
        engine = WorkflowEngine(data_dir / "workflows")
        st.session_state.workflow_engine = engine


# ==============================================================================
# EXPORTED ITEMS
# ==============================================================================

__all__ = [
    "render_workflow_automation",
    "initialize_workflow_automation",
    "WorkflowEngine",
    "Workflow",
    "WorkflowExecution",
    "Task",
    "ApprovalRequest",
    "WorkflowTemplate",
    "WorkflowStatus",
    "TaskStatus",
    "TriggerType",
    "WorkflowCategory",
]
