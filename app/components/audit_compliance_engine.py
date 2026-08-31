"""
Audit and Compliance Engine for Academic Integrity

Features:
- Regulatory tracking for academic standards
- Policy violation detection
- Complete audit trail
- Compliance reporting
- Policy enforcement
- Compliance dashboard
- Certificate generation
- Regulatory updates
"""

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class ComplianceStatus(Enum):
    """Compliance status levels."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    EXEMPT = "exempt"


class AuditSeverity(Enum):
    """Audit event severity."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PolicyType(Enum):
    """Policy types."""

    ACADEMIC_INTEGRITY = "academic_integrity"
    PLAGIARISM = "plagiarism"
    DATA_PRIVACY = "data_privacy"
    ACCESS_CONTROL = "access_control"
    RETENTION = "retention"
    REPORTING = "reporting"


class Regulation(Enum):
    """Academic regulations."""

    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    INSTITUTIONAL = "institutional"
    GOVERNMENTAL = "governmental"


@dataclass
class AuditEvent:
    """Audit event record."""

    id: str
    timestamp: float
    user: str
    action: str
    resource: str
    severity: AuditSeverity
    description: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    compliance_impact: Optional[str] = None


@dataclass
class PolicyViolation:
    """Policy violation record."""

    id: str
    policy_type: PolicyType
    description: str
    severity: AuditSeverity
    detected_at: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    action_taken: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance report."""

    id: str
    title: str
    period_start: str
    period_end: str
    status: ComplianceStatus
    metrics: dict[str, Any]
    findings: list[dict[str, Any]]
    recommendations: list[str]
    generated_at: float
    generated_by: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCertificate:
    """Compliance certificate."""

    id: str
    title: str
    recipient: str
    regulation: Regulation
    issued_at: float
    expires_at: Optional[float] = None
    status: ComplianceStatus
    certificate_hash: str
    issued_by: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegulatoryStandard:
    """Regulatory standard."""

    id: str
    name: str
    type: Regulation
    version: str
    description: str
    requirements: list[str]
    last_updated: float
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# AUDIT TRAIL MANAGER
# ==============================================================================


class AuditTrailManager:
    """
    Comprehensive audit trail management.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.events: list[AuditEvent] = []
        self.violations: list[PolicyViolation] = []
        self._load_data()

    def _load_data(self):
        """Load audit data from storage."""
        try:
            audit_path = self.storage_path / "audit_data.json"
            if audit_path.exists():
                with open(audit_path, "r") as f:
                    data = json.load(f)

                    self.events = [AuditEvent(**e) for e in data.get("events", [])]

                    self.violations = [
                        PolicyViolation(**v) for v in data.get("violations", [])
                    ]
        except Exception as e:
            print(f"Error loading audit data: {e}")

    def _save_data(self):
        """Save audit data to storage."""
        try:
            audit_path = self.storage_path / "audit_data.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "events": [asdict(e) for e in self.events[-10000:]],
                "violations": [asdict(v) for v in self.violations[-1000:]],
            }

            with open(audit_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving audit data: {e}")

    def log_event(
        self,
        user: str,
        action: str,
        resource: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            user: User performing action
            action: Action performed
            resource: Resource affected
            description: Event description
            severity: Event severity
            metadata: Additional metadata

        Returns:
            AuditEvent: Created event
        """
        payload = f"{user}{action}{resource}".encode()
        event = AuditEvent(
            id=f"audit_{int(time.time())}_{hashlib.md5(f'{user}{action}{resource}'.encode()).hexdigest()[:8]}",
            timestamp=time.time(),
            user=user,
            action=action,
            resource=resource,
            severity=severity,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
            compliance_impact=self._assess_compliance_impact(action, resource),
        )

        self.events.append(event)
        self._save_data()

        return event

    def _assess_compliance_impact(self, action: str, resource: str) -> str:
        """Assess compliance impact of an action."""
        high_impact_actions = ["delete", "modify", "share", "export"]
        medium_impact_actions = ["view", "download", "print"]

        if any(act in action.lower() for act in high_impact_actions):
            return "high"
        elif any(act in action.lower() for act in medium_impact_actions):
            return "medium"
        else:
            return "low"

    def add_violation(
        self,
        policy_type: PolicyType,
        description: str,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        metadata: Dict[str, Any] = None,
    ) -> PolicyViolation:
        """
        Add a policy violation.

        Args:
            policy_type: Type of policy violated
            description: Violation description
            severity: Violation severity

        Returns:
            PolicyViolation: Created violation
        """
        viol_payload = f"{policy_type.value}{description}".encode()
        violation = PolicyViolation(
            id=f"viol_{int(time.time())}_{hashlib.md5(f'{policy_type.value}{description}'.encode()).hexdigest()[:8]}",
            policy_type=policy_type,
            description=description,
            severity=severity,
            detected_at=time.time(),
            resolved=False,
            metadata=metadata or {},
        )

        self.violations.append(violation)
        self._save_data()

        # Log violation as audit event
        self.log_event(
            user="system",
            action="policy_violation_detected",
            resource=policy_type.value,
            description=f"Policy violation: {description}",
            severity=severity,
            metadata={"violation_id": violation.id},
        )

        return violation

    def resolve_violation(self, violation_id: str, resolved_by: str, action_taken: str):
        """Resolve a policy violation."""
        for violation in self.violations:
            if violation.id == violation_id:
                violation.resolved = True
                violation.resolved_at = time.time()
                violation.resolved_by = resolved_by
                violation.action_taken = action_taken

                self.log_event(
                    user=resolved_by,
                    action="resolve_policy_violation",
                    resource=violation.policy_type.value,
                    description=f"Resolved violation: {violation.description}",
                    severity=AuditSeverity.INFO,
                    metadata={"violation_id": violation_id},
                )

                self._save_data()
                break

    def get_events(
        self,
        user: str = None,
        severity: AuditSeverity = None,
        action: str = None,
        resource: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """
        Get audit events with filters.

        Args:
            user: Filter by user
            severity: Filter by severity
            action: Filter by action
            resource: Filter by resource
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum results

        Returns:
            List[AuditEvent]: Filtered events
        """
        filtered = self.events

        if user:
            filtered = [e for e in filtered if e.user == user]

        if severity:
            filtered = [e for e in filtered if e.severity == severity]

        if action:
            filtered = [e for e in filtered if action.lower() in e.action.lower()]

        if resource:
            filtered = [e for e in filtered if resource.lower() in e.resource.lower()]

        if start_date:
            start_ts = datetime.fromisoformat(start_date).timestamp()
            filtered = [e for e in filtered if e.timestamp >= start_ts]

        if end_date:
            end_ts = datetime.fromisoformat(end_date).timestamp()
            filtered = [e for e in filtered if e.timestamp <= end_ts]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered[:limit]

    def get_violations(
        self,
        policy_type: PolicyType = None,
        severity: AuditSeverity = None,
        resolved: bool = None,
        limit: int = 50,
    ) -> List[PolicyViolation]:
        """Get policy violations with filters."""
        filtered = self.violations

        if policy_type:
            filtered = [v for v in filtered if v.policy_type == policy_type]

        if severity:
            filtered = [v for v in filtered if v.severity == severity]

        if resolved is not None:
            filtered = [v for v in filtered if v.resolved == resolved]

        filtered.sort(key=lambda x: x.detected_at, reverse=True)

        return filtered[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        total_events = len(self.events)
        total_violations = len(self.violations)

        # Severity distribution
        severity_counts = Counter([e.severity.value for e in self.events])

        # Violations by type
        violation_counts = Counter([v.policy_type.value for v in self.violations])

        # Resolution rate
        resolved_violations = len([v for v in self.violations if v.resolved])
        resolution_rate = (
            resolved_violations / total_violations if total_violations > 0 else 1.0
        )

        return {
            "total_events": total_events,
            "total_violations": total_violations,
            "severity_distribution": dict(severity_counts),
            "violation_distribution": dict(violation_counts),
            "resolution_rate": resolution_rate,
            "open_violations": total_violations - resolved_violations,
        }


# ==============================================================================
# POLICY ENFORCER
# ==============================================================================


class PolicyEnforcer:
    """
    Enforce academic integrity policies.
    """

    def __init__(self, audit_manager: AuditTrailManager):
        self.audit_manager = audit_manager
        self.policies: dict[PolicyType, list[dict]] = defaultdict(list)
        self._load_policies()

    def _load_policies(self):
        """Load policies from storage."""
        # Academic integrity policies
        self.policies[PolicyType.ACADEMIC_INTEGRITY] = [
            {
                "id": "ai_001",
                "name": "Original Work Requirement",
                "description": "All submitted work must be original",
                "threshold": 0.85,
                "action": "flag",
            },
            {
                "id": "ai_002",
                "name": "Citation Requirement",
                "description": "All sources must be properly cited",
                "threshold": 0.70,
                "action": "warn",
            },
        ]

        # Plagiarism policies
        self.policies[PolicyType.PLAGIARISM] = [
            {
                "id": "plag_001",
                "name": "Similarity Threshold",
                "description": "Maximum allowed similarity",
                "threshold": 0.75,
                "action": "block",
            },
            {
                "id": "plag_002",
                "name": "Content Copying",
                "description": "Direct copying detection",
                "threshold": 0.50,
                "action": "review",
            },
        ]

        # Data privacy policies
        self.policies[PolicyType.DATA_PRIVACY] = [
            {
                "id": "dp_001",
                "name": "Data Access Control",
                "description": "Restrict access to sensitive data",
                "threshold": 0.0,
                "action": "enforce",
            }
        ]

    def enforce_policies(
        self, data: Dict[str, Any], policy_types: List[PolicyType] = None
    ) -> List[PolicyViolation]:
        """
        Enforce policies on data.

        Args:
            data: Data to check against policies
            policy_types: Types of policies to enforce

        Returns:
            List[PolicyViolation]: Detected violations
        """
        violations = []

        if policy_types is None:
            policy_types = list(PolicyType)

        for policy_type in policy_types:
            policies = self.policies.get(policy_type, [])

            for policy in policies:
                # Check policy
                violation = self._check_policy(policy, data)
                if violation:
                    violations.append(violation)

                    # Log violation
                    self.audit_manager.add_violation(
                        policy_type=policy_type,
                        description=violation,
                        severity=AuditSeverity.HIGH,
                        metadata={"policy": policy},
                    )

        return violations

    def _check_policy(self, policy: Dict, data: Dict) -> Optional[str]:
        """Check a specific policy."""
        policy_type = policy.get("id", "")
        threshold = policy.get("threshold", 0.0)

        # Check similarity threshold
        if "similarity" in data and data["similarity"] >= threshold:
            if policy_type.startswith("plag"):
                return f"Similarity {data['similarity']:.1%} exceeds threshold {threshold:.0%}"
            elif policy_type.startswith("ai"):
                return f"Originality score below threshold: {data['similarity']:.1%} > {threshold:.0%}"

        # Check data access
        if policy_type == "dp_001" and data.get("access_level", "public") == "public":
            return "Public access to sensitive data detected"

        # Check document count
        if policy_type == "ai_002" and data.get("document_count", 0) > 10:
            return "Excessive document submissions detected"

        return None


# ==============================================================================
# COMPLIANCE REPORT GENERATOR
# ==============================================================================


class ComplianceReportGenerator:
    """
    Generate compliance reports.
    """

    def __init__(self, audit_manager: AuditTrailManager):
        self.audit_manager = audit_manager

    def generate_report(
        self, title: str, period_start: str, period_end: str, generated_by: str
    ) -> ComplianceReport:
        """
        Generate a compliance report.

        Args:
            title: Report title
            period_start: Start date
            period_end: End date
            generated_by: Who generated the report

        Returns:
            ComplianceReport: Generated report
        """
        # Collect metrics
        metrics = self._collect_metrics(period_start, period_end)

        # Generate findings
        findings = self._generate_findings(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(findings)

        # Determine overall status
        status = self._determine_status(findings)

        report = ComplianceReport(
            id=f"report_{int(time.time())}",
            title=title,
            period_start=period_start,
            period_end=period_end,
            status=status,
            metrics=metrics,
            findings=findings,
            recommendations=recommendations,
            generated_at=time.time(),
            generated_by=generated_by,
        )

        return report

    def _collect_metrics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Collect compliance metrics."""
        start_ts = datetime.fromisoformat(start_date).timestamp()
        end_ts = datetime.fromisoformat(end_date).timestamp()

        # Get events in period
        events = [
            e for e in self.audit_manager.events if start_ts <= e.timestamp <= end_ts
        ]
        violations = [
            v
            for v in self.audit_manager.violations
            if start_ts <= v.detected_at <= end_ts
        ]

        # Calculate metrics
        total_events = len(events)
        total_violations = len(violations)

        # Event severity breakdown
        severity_breakdown = Counter([e.severity.value for e in events])

        # Violation type breakdown
        violation_breakdown = Counter([v.policy_type.value for v in violations])

        # Resolution rate
        resolved = len([v for v in violations if v.resolved])
        resolution_rate = resolved / total_violations if total_violations > 0 else 1.0

        return {
            "period_start": start_date,
            "period_end": end_date,
            "total_events": total_events,
            "total_violations": total_violations,
            "severity_breakdown": dict(severity_breakdown),
            "violation_breakdown": dict(violation_breakdown),
            "resolution_rate": resolution_rate,
            "event_per_day": total_events / max(1, (end_ts - start_ts) / 86400),
        }

    def _generate_findings(self, metrics: Dict) -> List[Dict[str, Any]]:
        """Generate findings from metrics."""
        findings = []

        # Find critical issues
        if metrics.get("severity_breakdown", {}).get("critical", 0) > 0:
            findings.append(
                {
                    "type": "critical_events",
                    "description": f"Found {metrics['severity_breakdown']['critical']} critical events",
                    "severity": "high",
                    "recommendation": "Investigate critical events immediately",
                }
            )

        # Check violation rate
        if metrics.get("total_violations", 0) > 10:
            findings.append(
                {
                    "type": "high_violation_rate",
                    "description": f"High violation rate: {metrics['total_violations']} violations detected",
                    "severity": "medium",
                    "recommendation": "Review policies and enforcement mechanisms",
                }
            )

        # Check resolution rate
        if metrics.get("resolution_rate", 1.0) < 0.5:
            findings.append(
                {
                    "type": "low_resolution_rate",
                    "description": f"Low resolution rate: {metrics['resolution_rate']:.1%}",
                    "severity": "medium",
                    "recommendation": "Implement faster violation resolution process",
                }
            )

        return findings

    def _generate_recommendations(self, findings: List[Dict]) -> List[str]:
        """Generate recommendations from findings."""
        recommendations = set()

        for finding in findings:
            if "recommendation" in finding:
                recommendations.add(finding["recommendation"])

        # Add general recommendations
        recommendations.add("Regular compliance reviews are recommended")
        recommendations.add("Maintain audit trail for all actions")

        return list(recommendations)

    def _determine_status(self, findings: List[Dict]) -> ComplianceStatus:
        """Determine overall compliance status."""
        critical_findings = [f for f in findings if f.get("severity") == "high"]

        if critical_findings:
            return ComplianceStatus.NON_COMPLIANT
        elif len(findings) > 3:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        elif findings:
            return ComplianceStatus.UNDER_REVIEW
        else:
            return ComplianceStatus.COMPLIANT


# ==============================================================================
# CERTIFICATE MANAGER
# ==============================================================================


class CertificateManager:
    """
    Manage compliance certificates.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.certificates: list[ComplianceCertificate] = []
        self._load_certificates()

    def _load_certificates(self):
        """Load certificates from storage."""
        try:
            cert_path = self.storage_path / "certificates.json"
            if cert_path.exists():
                with open(cert_path, "r") as f:
                    data = json.load(f)
                    self.certificates = [
                        ComplianceCertificate(**c) for c in data.get("certificates", [])
                    ]
        except Exception as e:
            print(f"Error loading certificates: {e}")

    def _save_certificates(self):
        """Save certificates to storage."""
        try:
            cert_path = self.storage_path / "certificates.json"
            cert_path.parent.mkdir(parents=True, exist_ok=True)

            data = {"certificates": [asdict(c) for c in self.certificates[-1000:]]}

            with open(cert_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving certificates: {e}")

    def issue_certificate(
        self,
        title: str,
        recipient: str,
        regulation: Regulation,
        issued_by: str,
        expires_at: Optional[float] = None,
    ) -> ComplianceCertificate:
        """
        Issue a compliance certificate.

        Args:
            title: Certificate title
            recipient: Certificate recipient
            regulation: Applicable regulation
            issued_by: Who issued the certificate
            expires_at: Expiration timestamp

        Returns:
            ComplianceCertificate: Issued certificate
        """
        cert_id = f"cert_{int(time.time())}_{hashlib.md5(f'{recipient}{title}'.encode()).hexdigest()[:8]}"

        # Generate certificate hash
        cert_hash = hashlib.sha256(
            f"{cert_id}{recipient}{title}{time.time()}".encode()
        ).hexdigest()[:16]

        certificate = ComplianceCertificate(
            id=cert_id,
            title=title,
            recipient=recipient,
            regulation=regulation,
            issued_at=time.time(),
            expires_at=expires_at,
            status=ComplianceStatus.COMPLIANT,
            certificate_hash=cert_hash,
            issued_by=issued_by,
        )

        self.certificates.append(certificate)
        self._save_certificates()

        return certificate

    def get_certificate(self, cert_id: str) -> Optional[ComplianceCertificate]:
        """Get certificate by ID."""
        for cert in self.certificates:
            if cert.id == cert_id:
                return cert
        return None

    def revoke_certificate(self, cert_id: str, reason: str):
        """Revoke a certificate."""
        cert = self.get_certificate(cert_id)
        if cert:
            cert.status = ComplianceStatus.NON_COMPLIANT
            cert.metadata["revoked_at"] = time.time()
            cert.metadata["revocation_reason"] = reason
            self._save_certificates()


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_audit_compliance_engine():
    """Render audit compliance engine UI."""
    st.subheader("📋 Audit & Compliance Engine")

    # Initialize
    if "audit_compliance_engine" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        audit_manager = AuditTrailManager(data_dir / "audit")
        st.session_state.audit_compliance_engine = {
            "audit_manager": audit_manager,
            "policy_enforcer": PolicyEnforcer(audit_manager),
            "report_generator": ComplianceReportGenerator(audit_manager),
            "certificate_manager": CertificateManager(data_dir / "certificates"),
            "initialized": True,
        }

    engine = st.session_state.audit_compliance_engine

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Dashboard",
            "📝 Audit Trail",
            "🚨 Violations",
            "📄 Reports",
            "📜 Certificates",
        ]
    )

    with tab1:
        render_compliance_dashboard(engine)

    with tab2:
        render_audit_trail(engine)

    with tab3:
        render_violation_management(engine)

    with tab4:
        render_compliance_reports(engine)

    with tab5:
        render_certificate_management(engine)


def render_compliance_dashboard(engine: dict):
    """Render compliance dashboard."""
    st.markdown("#### 📊 Compliance Dashboard")

    audit_manager = engine["audit_manager"]
    stats = audit_manager.get_stats()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", stats["total_events"])
    col2.metric("Total Violations", stats["total_violations"])
    col3.metric("Resolution Rate", f"{stats['resolution_rate']:.1%}")
    col4.metric("Open Violations", stats["open_violations"])

    # Charts
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Severity Distribution", "Violation Distribution"),
    )

    # Severity distribution
    if stats["severity_distribution"]:
        severities = list(stats["severity_distribution"].keys())
        counts = list(stats["severity_distribution"].values())
        fig.add_trace(go.Pie(labels=severities, values=counts), row=1, col=1)

    # Violation distribution
    if stats["violation_distribution"]:
        violations = list(stats["violation_distribution"].keys())
        counts = list(stats["violation_distribution"].values())
        fig.add_trace(go.Pie(labels=violations, values=counts), row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_audit_trail(engine: dict):
    """Render audit trail UI."""
    st.markdown("#### 📝 Audit Trail")

    audit_manager = engine["audit_manager"]

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        user_filter = st.text_input("User", placeholder="Filter by user")
    with col2:
        severity_filter = st.selectbox(
            "Severity", ["All"] + [s.value for s in AuditSeverity]
        )
    with col3:
        action_filter = st.text_input("Action", placeholder="Filter by action")
    with col4:
        limit = st.number_input("Limit", 10, 500, 100)

    # Get events
    severity = None if severity_filter == "All" else AuditSeverity(severity_filter)
    events = audit_manager.get_events(
        user=user_filter if user_filter else None,
        severity=severity,
        action=action_filter if action_filter else None,
        limit=limit,
    )

    if events:
        # Display events
        df = pd.DataFrame(
            [
                {
                    "Timestamp": datetime.fromtimestamp(e.timestamp).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "User": e.user,
                    "Action": e.action,
                    "Resource": e.resource,
                    "Severity": e.severity.value.upper(),
                    "Description": e.description[:50] + "..."
                    if len(e.description) > 50
                    else e.description,
                }
                for e in events
            ]
        )

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Export button
        if st.button("📥 Export Audit Log", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    else:
        st.info("No audit events found")


def render_violation_management(engine: dict):
    """Render violation management UI."""
    st.markdown("#### 🚨 Policy Violations")

    audit_manager = engine["audit_manager"]
    policy_enforcer = engine["policy_enforcer"]

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        policy_filter = st.selectbox(
            "Policy Type", ["All"] + [p.value for p in PolicyType]
        )
    with col2:
        severity_filter = st.selectbox(
            "Severity", ["All"] + [s.value for s in AuditSeverity]
        )
    with col3:
        resolution_filter = st.selectbox("Resolution", ["All", "Resolved", "Open"])

    # Get violations
    policy_type = None if policy_filter == "All" else PolicyType(policy_filter)
    severity = None if severity_filter == "All" else AuditSeverity(severity_filter)
    resolved = None if resolution_filter == "All" else resolution_filter == "Resolved"

    violations = audit_manager.get_violations(
        policy_type=policy_type, severity=severity, resolved=resolved
    )

    if violations:
        for violation in violations:
            with st.expander(
                f"Violation: {violation.policy_type.value} - {violation.severity.value.upper()}",
                expanded=violation.severity
                in [AuditSeverity.CRITICAL, AuditSeverity.HIGH],
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Description:** {violation.description}")
                    st.caption(
                        f"Detected: {datetime.fromtimestamp(violation.detected_at).strftime('%Y-%m-%d %H:%M')}"
                    )

                    if violation.resolved:
                        st.success(f"✅ Resolved by {violation.resolved_by}")
                        st.caption(f"Action: {violation.action_taken}")
                    else:
                        st.warning("⚠️ Open - Needs attention")

                with col2:
                    if not violation.resolved:
                        resolution_action = st.text_input(
                            "Action Taken",
                            key=f"action_{violation.id}",
                            placeholder="Describe action taken",
                        )
                        if st.button("Resolve", key=f"resolve_{violation.id}"):
                            if resolution_action:
                                audit_manager.resolve_violation(
                                    violation.id,
                                    st.session_state.get("username", "admin"),
                                    resolution_action,
                                )
                                st.rerun()
                            else:
                                st.error("Please describe the action taken")
    else:
        st.info("No violations found")


def render_compliance_reports(engine: dict):
    """Render compliance reports UI."""
    st.markdown("#### 📄 Compliance Reports")

    report_generator = engine["report_generator"]

    # Generate report
    with st.expander("📝 Generate New Report", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            report_title = st.text_input("Report Title", "Compliance Report")
            start_date = st.date_input(
                "Period Start", datetime.now() - timedelta(days=30)
            )
        with col2:
            report_desc = st.text_area("Description", placeholder="Report description")
            end_date = st.date_input("Period End", datetime.now())

        if st.button("Generate Report", type="primary", use_container_width=True):
            report = report_generator.generate_report(
                title=report_title,
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
                generated_by=st.session_state.get("username", "system"),
            )

            st.session_state.last_report = report
            st.success("✅ Report generated")
            st.rerun()

    # Display last report
    if hasattr(st.session_state, "last_report"):
        report = st.session_state.last_report

        st.markdown(f"#### 📊 {report.title}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", report.status.value.upper())
        col2.metric("Findings", len(report.findings))
        col3.metric("Recommendations", len(report.recommendations))

        # Metrics
        st.markdown("**Metrics:**")
        metrics_df = pd.DataFrame([report.metrics])
        st.dataframe(metrics_df, use_container_width=True)

        # Findings
        if report.findings:
            st.markdown("**Findings:**")
            for finding in report.findings:
                st.caption(
                    f"• {finding['description']} (Severity: {finding.get('severity', 'info')})"
                )

        # Recommendations
        if report.recommendations:
            st.markdown("**Recommendations:**")
            for rec in report.recommendations:
                st.caption(f"• {rec}")

        # Download
        if st.button("📥 Download Report", use_container_width=True):
            report_data = json.dumps(asdict(report), indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=report_data,
                file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


def render_certificate_management(engine: dict):
    """Render certificate management UI."""
    st.markdown("#### 📜 Compliance Certificates")

    cert_manager = engine["certificate_manager"]

    # Issue certificate
    with st.expander("📜 Issue New Certificate", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cert_title = st.text_input(
                "Certificate Title", "Academic Integrity Compliance"
            )
            recipient = st.text_input("Recipient")
        with col2:
            regulation = st.selectbox("Regulation", [r.value for r in Regulation])
            issued_by = st.text_input(
                "Issued By", st.session_state.get("username", "system")
            )

        expires = st.checkbox("Add Expiration")
        expires_at = None
        if expires:
            expires_at = st.date_input(
                "Expiration Date", datetime.now() + timedelta(days=365)
            )
            expires_at = datetime.combine(expires_at, datetime.min.time()).timestamp()

        if st.button("Issue Certificate", type="primary", use_container_width=True):
            if cert_title and recipient:
                certificate = cert_manager.issue_certificate(
                    title=cert_title,
                    recipient=recipient,
                    regulation=Regulation(regulation),
                    issued_by=issued_by,
                    expires_at=expires_at,
                )
                st.success(f"✅ Certificate issued: {certificate.id}")
                st.rerun()
            else:
                st.error("Please fill in all required fields")

    # Display certificates
    if cert_manager.certificates:
        cert_df = pd.DataFrame(
            [
                {
                    "ID": c.id[:8],
                    "Title": c.title,
                    "Recipient": c.recipient,
                    "Regulation": c.regulation.value,
                    "Issued": datetime.fromtimestamp(c.issued_at).strftime("%Y-%m-%d"),
                    "Status": c.status.value.upper(),
                }
                for c in cert_manager.certificates[-20:]
            ]
        )

        st.dataframe(cert_df, use_container_width=True, hide_index=True)

        # Certificate details
        selected_cert = st.selectbox(
            "View Certificate",
            [f"{c.title} - {c.recipient}" for c in cert_manager.certificates],
        )

        if selected_cert:
            idx = [
                f"{c.title} - {c.recipient}" for c in cert_manager.certificates
            ].index(selected_cert)
            cert = cert_manager.certificates[idx]

            with st.expander("📄 Certificate Details", expanded=True):
                st.markdown(f"**Title:** {cert.title}")
                st.markdown(f"**Recipient:** {cert.recipient}")
                st.markdown(f"**Regulation:** {cert.regulation.value.upper()}")
                st.markdown(f"**Status:** {cert.status.value.upper()}")
                st.markdown(
                    f"**Issued:** {datetime.fromtimestamp(cert.issued_at).strftime('%Y-%m-%d %H:%M')}"
                )
                if cert.expires_at:
                    st.markdown(
                        f"**Expires:** {datetime.fromtimestamp(cert.expires_at).strftime('%Y-%m-%d')}"
                    )
                st.markdown(f"**Certificate Hash:** `{cert.certificate_hash}`")

                if st.button(
                    "Revoke Certificate", type="secondary", use_container_width=True
                ):
                    reason = st.text_input("Revocation Reason")
                    if reason:
                        cert_manager.revoke_certificate(cert.id, reason)
                        st.success("Certificate revoked")
                        st.rerun()
    else:
        st.info("No certificates issued")


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_audit_compliance_engine():
    """Initialize audit compliance engine."""
    if "audit_compliance_engine_initialized" not in st.session_state:
        st.session_state.audit_compliance_engine_initialized = True

        data_dir = Path(st.session_state.get("data_dir", "."))
        audit_manager = AuditTrailManager(data_dir / "audit")

        st.session_state.audit_compliance_engine = {
            "audit_manager": audit_manager,
            "policy_enforcer": PolicyEnforcer(audit_manager),
            "report_generator": ComplianceReportGenerator(audit_manager),
            "certificate_manager": CertificateManager(data_dir / "certificates"),
            "initialized": True,
        }


# ==============================================================================
# EXPORTED ITEMS
# ==============================================================================

__all__ = [
    "render_audit_compliance_engine",
    "initialize_audit_compliance_engine",
    "AuditTrailManager",
    "PolicyEnforcer",
    "ComplianceReportGenerator",
    "CertificateManager",
    "AuditEvent",
    "PolicyViolation",
    "ComplianceReport",
    "ComplianceCertificate",
    "ComplianceStatus",
    "AuditSeverity",
    "PolicyType",
    "Regulation",
]
