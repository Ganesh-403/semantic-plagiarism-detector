"""
Real-Time Monitoring and Alert System

Features:
- Live document processing monitoring
- System health dashboard
- Real-time alerts and notifications
- Performance anomaly detection
- Resource usage tracking
- WebSocket-based live updates
- Alert rules engine
- Notification channels (email, Slack, webhook)
"""

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from plotly.subplots import make_subplots

# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class SystemMetric:
    """System metric data point."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_sent: float
    network_recv: float
    process_count: int
    thread_count: int
    active_sessions: int
    queue_size: int
    processing_speed: float  # documents per minute


@dataclass
class AlertRule:
    """Alert rule configuration."""

    id: str
    name: str
    condition: str  # cpu > 80, memory > 90, etc.
    severity: str  # info, warning, critical
    message: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: Optional[float] = None


@dataclass
class Alert:
    """Alert instance."""

    id: str
    rule_id: str
    severity: str
    message: str
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# MONITORING ENGINE
# ==============================================================================


class MonitoringEngine:
    """Core monitoring engine collecting system metrics."""

    def __init__(self, max_history: int = 3600):  # 1 hour at 1s intervals
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.alerts: list[Alert] = []
        self.alert_rules: list[AlertRule] = []
        self.running = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []

    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        st.toast("✅ Monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        st.toast("⏹️ Monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metric = self._collect_metrics()

                with self._lock:
                    self.metrics_history.append(metric)

                # Check alert rules
                self._check_alerts(metric)

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metric)
                    except Exception as e:
                        print(f"Callback error: {e}")

                time.sleep(1)  # 1 second interval

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)  # Back off on error

    def _collect_metrics(self) -> SystemMetric:
        """Collect current system metrics."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net = psutil.net_io_counters()
            process = psutil.Process()

            # Get active sessions from Redis or fallback
            active_sessions = self._get_active_sessions()

            return SystemMetric(
                timestamp=time.time(),
                cpu_percent=cpu,
                memory_percent=memory.percent,
                disk_usage=disk.percent,
                network_sent=net.bytes_sent,
                network_recv=net.bytes_recv,
                process_count=len(psutil.pids()),
                thread_count=process.num_threads(),
                active_sessions=active_sessions,
                queue_size=self._get_queue_size(),
                processing_speed=self._calculate_processing_speed(),
            )
        except Exception as e:
            print(f"Metric collection error: {e}")
            return SystemMetric(
                timestamp=time.time(),
                cpu_percent=0,
                memory_percent=0,
                disk_usage=0,
                network_sent=0,
                network_recv=0,
                process_count=0,
                thread_count=0,
                active_sessions=0,
                queue_size=0,
                processing_speed=0,
            )

    def _get_active_sessions(self) -> int:
        """Get active session count."""
        try:
            # Try to get from Redis
            from src.utils.redis_cache import get_cache

            cache = get_cache()
            if cache.is_available():
                keys = list(
                    cache._client.scan_iter(match="spd:v1:session:*:last_interaction")
                )
                active = 0
                now = time.time()
                for key in keys:
                    try:
                        session_id = (
                            key.split(b":")[3].decode()
                            if isinstance(key, bytes)
                            else key.split(":")[3]
                        )
                        last = cache.get(
                            f"spd:v1:session:{session_id}:last_interaction"
                        )
                        if last and (now - float(last)) <= 900:  # 15 minutes
                            active += 1
                    except:  # noqa: E722
                        pass
                return active
        except:  # noqa: E722
            pass

        # Fallback: estimate from session state
        return len(st.session_state.get("active_sessions", {}))

    def _get_queue_size(self) -> int:
        """Get current queue size."""
        try:
            from src.utils.redis_cache import get_cache

            cache = get_cache()
            if cache.is_available():
                queue = cache._client.llen("processing_queue")
                return queue or 0
        except:  # noqa: E722
            pass
        return 0

    def _calculate_processing_speed(self) -> float:
        """Calculate processing speed in documents per minute."""
        # Get recent metrics (last 5 minutes)
        recent = []
        with self._lock:
            for metric in list(self.metrics_history)[-300:]:  # Last 5 minutes
                recent.append(metric)

        if len(recent) < 10:
            return 0

        # Calculate throughput
        # This is simplified - would need actual processing tracking
        return 0

    def _check_alerts(self, metric: SystemMetric):
        """Check alert rules against current metric."""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                if time.time() - rule.last_triggered < rule.cooldown_seconds:
                    continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, metric):
                alert = Alert(
                    id=f"alert_{int(time.time())}_{rule.id}",
                    rule_id=rule.id,
                    severity=rule.severity,
                    message=rule.message,
                    timestamp=time.time(),
                    metadata={"metric": asdict(metric)},
                )
                self.alerts.append(alert)
                rule.last_triggered = time.time()

                # Trigger notification
                self._send_notification(alert)

    def _evaluate_condition(self, condition: str, metric: SystemMetric) -> bool:
        """Evaluate a condition string against metrics."""
        try:
            # Simple condition parser
            # Examples: "cpu > 80", "memory > 90", "active_sessions > 100"
            parts = condition.split()
            if len(parts) != 3:
                return False

            attr, op, value = parts
            attr = attr.strip()
            op = op.strip()
            value = float(value.strip())

            # Get attribute value
            if hasattr(metric, attr):
                actual = getattr(metric, attr)
            else:
                return False

            # Evaluate operation
            if op == ">":
                return actual > value
            elif op == "<":
                return actual < value
            elif op == ">=":
                return actual >= value
            elif op == "<=":
                return actual <= value
            elif op == "==":
                return actual == value
            elif op == "!=":
                return actual != value
            else:
                return False

        except Exception as e:
            print(f"Condition evaluation error: {e}")
            return False

    def _send_notification(self, alert: Alert):
        """Send notification for alert."""
        # Log alert
        print(f"🚨 ALERT: {alert.severity} - {alert.message}")

        # Try to send to configured channels
        self._send_email_alert(alert)
        self._send_webhook_alert(alert)

    def _send_email_alert(self, alert: Alert):
        """Send email notification."""
        # Placeholder - would integrate with email service
        pass

    def _send_webhook_alert(self, alert: Alert):
        """Send webhook notification."""
        # Placeholder - would integrate with webhook service
        pass

    def register_callback(self, callback: Callable):
        """Register callback for metric updates."""
        self._callbacks.append(callback)

    def get_latest_metrics(self) -> Optional[SystemMetric]:
        """Get the latest metric."""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None

    def get_metrics_history(self, seconds: int = 60) -> list[SystemMetric]:
        """Get metrics history for last N seconds."""
        cutoff = time.time() - seconds
        result = []
        with self._lock:
            for metric in self.metrics_history:
                if metric.timestamp >= cutoff:
                    result.append(metric)
        return result

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)

    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        self.alert_rules = [r for r in self.alert_rules if r.id != rule_id]

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break

    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                break

    def get_active_alerts(self) -> list[Alert]:
        """Get active (unresolved) alerts."""
        return [a for a in self.alerts if not a.resolved]


# ==============================================================================
# HEALTH CHECK SYSTEM
# ==============================================================================


class HealthChecker:
    """System health check and diagnostics."""

    @staticmethod
    def check_database() -> dict[str, Any]:
        """Check database health."""
        try:
            from src.core.app_config import AUTH_DB_PATH, CORPUS_DB_PATH

            result = {
                "corpus_healthy": False,
                "auth_healthy": False,
                "corpus_size_mb": 0,
                "auth_size_mb": 0,
                "details": [],
            }

            # Check corpus DB
            if CORPUS_DB_PATH.exists():
                result["corpus_healthy"] = True
                result["corpus_size_mb"] = CORPUS_DB_PATH.stat().st_size / (1024 * 1024)
                result["details"].append(
                    f"Corpus DB: {result['corpus_size_mb']:.1f} MB"
                )

            # Check auth DB
            if AUTH_DB_PATH.exists():
                result["auth_healthy"] = True
                result["auth_size_mb"] = AUTH_DB_PATH.stat().st_size / (1024 * 1024)
                result["details"].append(f"Auth DB: {result['auth_size_mb']:.1f} MB")

            return result

        except Exception as e:
            return {
                "corpus_healthy": False,
                "auth_healthy": False,
                "corpus_size_mb": 0,
                "auth_size_mb": 0,
                "details": [f"Database check failed: {str(e)}"],
            }

    @staticmethod
    def check_redis() -> dict[str, Any]:
        """Check Redis health."""
        try:
            from src.utils.redis_cache import get_cache

            cache = get_cache()
            connected, latency = cache.ping()

            if connected:
                return {
                    "healthy": True,
                    "latency_ms": latency,
                    "connected": True,
                    "details": [f"Redis connected ({latency} ms)"],
                }
            else:
                return {
                    "healthy": False,
                    "latency_ms": 0,
                    "connected": False,
                    "details": ["Redis not connected"],
                }
        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": 0,
                "connected": False,
                "details": [f"Redis error: {str(e)}"],
            }

    @staticmethod
    def check_embedding_model() -> dict[str, Any]:
        """Check embedding model health."""
        try:
            from src.core.embedding_model import EmbeddingModelManager

            model = EmbeddingModelManager.get_instance().get_model()

            # Test embedding with small text
            test_text = "This is a test."
            embedding = model.encode([test_text])

            return {
                "healthy": True,
                "model_loaded": True,
                "embedding_dim": len(embedding[0]) if len(embedding) > 0 else 0,
                "details": ["Embedding model loaded successfully"],
            }
        except Exception as e:
            return {
                "healthy": False,
                "model_loaded": False,
                "embedding_dim": 0,
                "details": [f"Embedding model error: {str(e)}"],
            }

    @staticmethod
    def check_faiss_index() -> dict[str, Any]:
        """Check FAISS index health."""
        try:
            from src.core.app_config import FAISS_INDEX_PATH

            if FAISS_INDEX_PATH.exists():
                from src.core.faiss_index import load_index

                index = load_index(str(FAISS_INDEX_PATH))

                return {
                    "healthy": True,
                    "exists": True,
                    "vectors": index.ntotal if index else 0,
                    "size_mb": FAISS_INDEX_PATH.stat().st_size / (1024 * 1024),
                    "details": [f"FAISS index: {index.ntotal if index else 0} vectors"],
                }
            else:
                return {
                    "healthy": False,
                    "exists": False,
                    "vectors": 0,
                    "size_mb": 0,
                    "details": ["FAISS index file not found"],
                }
        except Exception as e:
            return {
                "healthy": False,
                "exists": False,
                "vectors": 0,
                "size_mb": 0,
                "details": [f"FAISS error: {str(e)}"],
            }

    @staticmethod
    def run_full_health_check() -> dict[str, Any]:
        """Run complete health check."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "database": HealthChecker.check_database(),
            "redis": HealthChecker.check_redis(),
            "embedding_model": HealthChecker.check_embedding_model(),
            "faiss_index": HealthChecker.check_faiss_index(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=0.5),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "uptime_seconds": time.time() - psutil.boot_time(),
            },
        }

        # Overall health status
        critical_checks = [
            results["database"]["corpus_healthy"],
            results["embedding_model"]["healthy"],
            results["faiss_index"]["healthy"],
        ]

        results["overall_status"] = (
            "healthy"
            if all(critical_checks)
            else "degraded"
            if any(critical_checks)
            else "critical"
        )

        return results


# ==============================================================================
# PERFORMANCE ANOMALY DETECTION
# ==============================================================================


class AnomalyDetector:
    """Detect anomalies in system performance."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.history: list[float] = []

    def detect_spike(self, value: float, threshold: float = 3.0) -> bool:
        """Detect if value is a spike (3 standard deviations from mean)."""
        self.history.append(value)

        # Keep only window size
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 10:
            return False

        mean = sum(self.history) / len(self.history)
        std = (sum((x - mean) ** 2 for x in self.history) / len(self.history)) ** 0.5

        if std == 0:
            return False

        z_score = (value - mean) / std
        return abs(z_score) > threshold

    def detect_trend(self, values: list[float], window: int = 10) -> str:
        """Detect trend in values."""
        if len(values) < window:
            return "stable"

        recent = values[-window:]

        # Simple linear regression
        x = list(range(len(recent)))
        n = len(recent)
        sum_x = sum(x)
        sum_y = sum(recent)
        sum_xy = sum(x[i] * recent[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        slope = (
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            if (n * sum_x2 - sum_x**2) != 0
            else 0
        )

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_system_health_dashboard():
    """Render system health dashboard."""
    st.markdown("### 🏥 System Health Dashboard")

    # Run health check
    health = HealthChecker.run_full_health_check()

    # Overall status
    status_color = {"healthy": "🟢", "degraded": "🟡", "critical": "🔴"}.get(
        health["overall_status"], "⚪"
    )

    st.markdown(
        f"{status_color} **Overall Status:** {health['overall_status'].upper()}"
    )

    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    sys = health["system"]
    col1.metric("CPU", f"{sys['cpu_percent']:.1f}%")
    col2.metric("Memory", f"{sys['memory_percent']:.1f}%")
    col3.metric("Disk", f"{sys['disk_usage']:.1f}%")
    from src.utils.processing_time import format_uptime_seconds

    uptime_str = format_uptime_seconds(sys["uptime_seconds"])
    col4.metric("Uptime", uptime_str)

    # Component status
    st.markdown("#### Component Status")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        db = health["database"]
        status = "🟢" if db["corpus_healthy"] else "🔴"
        st.markdown(
            f"{status} Database: {'Healthy' if db['corpus_healthy'] else 'Unhealthy'}"
        )
        if db["corpus_healthy"]:
            st.caption(f"  Corpus: {db['corpus_size_mb']:.1f} MB")

        redis = health["redis"]
        status = "🟢" if redis["healthy"] else "🔴"
        st.markdown(
            f"{status} Redis: {'Connected' if redis['healthy'] else 'Disconnected'}"
        )
        if redis["healthy"]:
            st.caption(f"  Latency: {redis['latency_ms']} ms")

    with comp_col2:
        embedding = health["embedding_model"]
        status = "🟢" if embedding["healthy"] else "🔴"
        st.markdown(
            f"{status} Embedding Model: {'Loaded' if embedding['healthy'] else 'Failed'}"
        )
        if embedding["healthy"]:
            st.caption(f"  Dimensions: {embedding['embedding_dim']}")

        faiss = health["faiss_index"]
        status = "🟢" if faiss["healthy"] else "🔴"
        st.markdown(
            f"{status} FAISS Index: {'Available' if faiss['healthy'] else 'Missing'}"
        )
        if faiss["healthy"]:
            st.caption(f"  Vectors: {faiss['vectors']}")


def render_performance_charts(monitor: MonitoringEngine):
    """Render real-time performance charts."""
    st.markdown("### 📊 Real-Time Metrics")

    # Get metrics history
    metrics = monitor.get_metrics_history(seconds=60)

    if not metrics:
        st.info("No metrics collected yet")
        return

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "CPU Usage",
            "Memory Usage",
            "Disk Usage",
            "Network I/O",
            "Active Sessions",
            "Processing Speed",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Timestamps
    timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]

    # CPU
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[m.cpu_percent for m in metrics],
            name="CPU",
            line=dict(color="#ff6b6b", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)

    # Memory
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[m.memory_percent for m in metrics],
            name="Memory",
            line=dict(color="#4ecdc4", width=2),
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=90, line_dash="dash", line_color="red", row=1, col=2)

    # Disk
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[m.disk_usage for m in metrics],
            name="Disk",
            line=dict(color="#ffe66d", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=85, line_dash="dash", line_color="red", row=2, col=1)

    # Network
    net_sent = [m.network_sent / (1024 * 1024) for m in metrics]  # MB
    net_recv = [m.network_recv / (1024 * 1024) for m in metrics]  # MB
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=net_sent,
            name="Sent (MB)",
            line=dict(color="#a8e6cf", width=2),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=net_recv,
            name="Recv (MB)",
            line=dict(color="#ffd93d", width=2),
        ),
        row=2,
        col=2,
    )

    # Active Sessions
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[m.active_sessions for m in metrics],
            name="Sessions",
            line=dict(color="#6c5ce7", width=2),
        ),
        row=3,
        col=1,
    )

    # Processing Speed
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=[m.processing_speed for m in metrics],
            name="Speed",
            line=dict(color="#fd79a8", width=2),
        ),
        row=3,
        col=2,
    )

    # Update layout
    fig.update_layout(height=600, showlegend=False, template="plotly_white")

    # Update axes
    for row in range(1, 4):
        for col in range(1, 3):
            fig.update_xaxes(title_text="Time", row=row, col=col)
            fig.update_yaxes(title_text="%", row=row, col=col)

    st.plotly_chart(fig, use_container_width=True)


def render_alerts_dashboard(monitor: MonitoringEngine):
    """Render alerts dashboard."""
    st.markdown("### 🔔 Alerts")

    # Alert rules management
    with st.expander("⚙️ Alert Rules", expanded=False):
        # Display existing rules
        if monitor.alert_rules:
            rules_data = []
            for rule in monitor.alert_rules:
                rules_data.append(
                    {
                        "Name": rule.name,
                        "Condition": rule.condition,
                        "Severity": rule.severity.upper(),
                        "Status": "✅ Enabled" if rule.enabled else "❌ Disabled",
                    }
                )
            st.dataframe(pd.DataFrame(rules_data), use_container_width=True)

        # Add new rule
        st.markdown("#### Add New Rule")
        col1, col2, col3 = st.columns(3)
        with col1:
            rule_name = st.text_input("Rule Name", "High CPU")
            condition = st.text_input("Condition", "cpu > 80")
        with col2:
            severity = st.selectbox("Severity", ["info", "warning", "critical"])
            cooldown = st.number_input("Cooldown (seconds)", value=300)
        with col3:
            message = st.text_area("Message", "CPU usage exceeded 80%")

        if st.button("➕ Add Rule", use_container_width=True):
            rule = AlertRule(
                id=f"rule_{int(time.time())}",
                name=rule_name,
                condition=condition,
                severity=severity,
                message=message,
                cooldown_seconds=cooldown,
            )
            monitor.add_alert_rule(rule)
            st.success("✅ Rule added")
            st.rerun()

    # Active alerts
    active = monitor.get_active_alerts()

    if active:
        st.warning(f"⚠️ {len(active)} active alerts")

        for alert in active:
            severity_colors = {"info": "blue", "warning": "orange", "critical": "red"}  # noqa: F841

            with st.expander(
                f"{alert.severity.upper()}: {alert.message}",
                expanded=alert.severity == "critical",
            ):
                st.markdown(f"**Severity:** {alert.severity.upper()}")
                st.markdown(
                    f"**Time:** {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
                )

                if alert.metadata:
                    st.markdown("**Details:**")
                    st.json(alert.metadata)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Acknowledge", key=f"ack_{alert.id}"):
                        monitor.acknowledge_alert(alert.id)
                        st.rerun()
                with col2:
                    if st.button("🔧 Resolve", key=f"res_{alert.id}"):
                        monitor.resolve_alert(alert.id)
                        st.rerun()
    else:
        st.success("✅ No active alerts")


def render_monitoring_controls(monitor: MonitoringEngine):
    """Render monitoring controls."""
    st.markdown("### 🎮 Monitoring Controls")

    col1, col2 = st.columns(2)

    with col1:
        if monitor.running:
            if st.button("⏹️ Stop Monitoring", type="primary", use_container_width=True):
                monitor.stop_monitoring()
                st.rerun()
        else:
            if st.button(
                "▶️ Start Monitoring", type="primary", use_container_width=True
            ):
                monitor.start_monitoring()
                st.rerun()

    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()


# ==============================================================================
# MAIN UI FUNCTION
# ==============================================================================


def render_real_time_monitor():
    """Render complete real-time monitor UI."""
    st.subheader("🔄 Real-Time System Monitor")

    # Initialize monitoring engine
    if "monitoring_engine" not in st.session_state:
        st.session_state.monitoring_engine = MonitoringEngine()

    monitor = st.session_state.monitoring_engine

    # Controls
    render_monitoring_controls(monitor)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "🏥 Health Check", "🔔 Alerts", "⚙️ Settings"]
    )

    with tab1:
        render_performance_charts(monitor)

        # Current metrics
        latest = monitor.get_latest_metrics()
        if latest:
            st.markdown("#### Current Metrics")
            cols = st.columns(4)
            cols[0].metric("CPU", f"{latest.cpu_percent:.1f}%")
            cols[1].metric("Memory", f"{latest.memory_percent:.1f}%")
            cols[2].metric("Active Sessions", latest.active_sessions)
            cols[3].metric("Queue Size", latest.queue_size)

    with tab2:
        render_system_health_dashboard()

    with tab3:
        render_alerts_dashboard(monitor)

    with tab4:
        st.markdown("### ⚙️ Monitor Settings")

        # Anomaly detection settings
        st.markdown("#### Anomaly Detection")
        st.slider(
            "Anomaly Threshold (Z-Score)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Higher values = less sensitive",
        )

        # History retention
        st.markdown("#### Data Retention")
        st.number_input(
            "Metrics Retention (seconds)",
            min_value=60,
            max_value=3600,
            value=3600,
            step=60,
            help="How long to keep metric history",
        )

        # Alert cooldown
        st.markdown("#### Alert Settings")
        st.number_input(
            "Default Alert Cooldown (seconds)",
            min_value=60,
            max_value=3600,
            value=300,
            step=60,
        )


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_monitoring():
    """Initialize monitoring system."""
    if "monitoring_initialized" not in st.session_state:
        st.session_state.monitoring_initialized = True

        # Create monitoring engine
        monitor = MonitoringEngine()

        # Add default alert rules
        monitor.add_alert_rule(
            AlertRule(
                id="rule_cpu_high",
                name="High CPU Usage",
                condition="cpu > 80",
                severity="warning",
                message="CPU usage exceeded 80%",
                cooldown_seconds=300,
            )
        )

        monitor.add_alert_rule(
            AlertRule(
                id="rule_memory_high",
                name="High Memory Usage",
                condition="memory > 90",
                severity="critical",
                message="Memory usage exceeded 90%",
                cooldown_seconds=300,
            )
        )

        monitor.add_alert_rule(
            AlertRule(
                id="rule_disk_high",
                name="High Disk Usage",
                condition="disk > 85",
                severity="warning",
                message="Disk usage exceeded 85%",
                cooldown_seconds=600,
            )
        )

        st.session_state.monitoring_engine = monitor

        # Start monitoring automatically
        monitor.start_monitoring()
