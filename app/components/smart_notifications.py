"""
Smart Notification and Alert System

Features:
- Real-time alerts for plagiarism detection
- Customizable alert rules
- Multi-channel delivery (Email, Slack, Webhook, In-app)
- Smart filtering to reduce noise
- Notification history
- User preferences
- Batch digests
- Priority levels
"""

import json
import queue
import smtplib  # noqa: F401
import threading
import time
from collections import defaultdict, deque  # noqa: F401
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta  # noqa: F401
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set  # noqa: F401

import pandas as pd  # noqa: F401
import plotly.graph_objects as go
import requests  # noqa: F401
import streamlit as st

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class NotificationPriority(Enum):
    """Notification priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class NotificationChannel(Enum):
    """Notification channels."""

    IN_APP = "in_app"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


class NotificationStatus(Enum):
    """Notification delivery status."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"
    DISMISSED = "dismissed"


@dataclass
class Notification:
    """Notification data."""

    id: str
    title: str
    message: str
    priority: NotificationPriority
    channel: NotificationChannel
    recipient: str
    timestamp: float
    status: NotificationStatus
    read_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AlertRule:
    """Alert rule configuration."""

    id: str
    name: str
    enabled: bool
    condition: str  # e.g., "similarity > 0.85"
    priority: NotificationPriority
    channels: List[NotificationChannel]
    cooldown_seconds: int = 300
    last_triggered: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserNotificationPreferences:
    """User notification preferences."""

    user_id: str
    enabled: bool = True
    channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.IN_APP]
    )
    min_priority: NotificationPriority = NotificationPriority.NORMAL
    daily_digest: bool = True
    weekly_summary: bool = False
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None
    muted_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# NOTIFICATION MANAGER
# ==============================================================================


class NotificationManager:
    """
    Smart notification and alert system.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.notifications: List[Notification] = []
        self.alert_rules: List[AlertRule] = []
        self.user_preferences: Dict[str, UserNotificationPreferences] = {}
        self.notification_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}

        # Tracking
        self.delivery_stats = {
            "total": 0,
            "sent": 0,
            "failed": 0,
            "by_channel": defaultdict(int),
            "by_priority": defaultdict(int),
        }

        # Initialize
        self._load_data()
        self._register_default_handlers()
        self._start_worker()

    def _load_data(self):
        """Load data from storage."""
        try:
            data_path = self.storage_path / "notifications.json"
            if data_path.exists():
                with open(data_path, "r") as f:
                    data = json.load(f)

                    self.notifications = [
                        Notification(**n) for n in data.get("notifications", [])
                    ]

                    self.alert_rules = [AlertRule(**r) for r in data.get("rules", [])]

                    self.user_preferences = {
                        uid: UserNotificationPreferences(**pref)
                        for uid, pref in data.get("preferences", {}).items()
                    }
        except Exception as e:
            print(f"Error loading notification data: {e}")

    def _save_data(self):
        """Save data to storage."""
        try:
            data_path = self.storage_path / "notifications.json"
            data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "notifications": [asdict(n) for n in self.notifications],
                "rules": [asdict(r) for r in self.alert_rules],
                "preferences": {
                    uid: asdict(pref) for uid, pref in self.user_preferences.items()
                },
            }

            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving notification data: {e}")

    def _register_default_handlers(self):
        """Register default channel handlers."""
        self.channel_handlers[NotificationChannel.IN_APP] = self._send_in_app
        self.channel_handlers[NotificationChannel.EMAIL] = self._send_email
        self.channel_handlers[NotificationChannel.SLACK] = self._send_slack
        self.channel_handlers[NotificationChannel.WEBHOOK] = self._send_webhook

    def _start_worker(self):
        """Start notification worker thread."""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Worker thread for processing notifications."""
        while self.is_running:
            try:
                # Get notification from queue
                notification = self.notification_queue.get(timeout=1)

                # Process notification
                self._process_notification(notification)

                # Mark task done
                self.notification_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Notification worker error: {e}")

    def _process_notification(self, notification: Notification):
        """Process and send notification."""
        # Check user preferences
        if not self._should_send_notification(notification):
            notification.status = NotificationStatus.DISMISSED
            return

        # Send via channel
        handler = self.channel_handlers.get(notification.channel)
        if handler:
            try:
                handler(notification)
                notification.status = NotificationStatus.SENT
                self.delivery_stats["sent"] += 1
                self.delivery_stats["by_channel"][notification.channel.value] += 1
                self.delivery_stats["by_priority"][notification.priority.name] += 1
            except Exception as e:
                notification.status = NotificationStatus.FAILED
                self.delivery_stats["failed"] += 1
                print(f"Failed to send notification: {e}")

        self.delivery_stats["total"] += 1

        # Save to history
        self.notifications.append(notification)

        # Keep last 1000 notifications
        if len(self.notifications) > 1000:
            self.notifications = self.notifications[-1000:]

        self._save_data()

    def _should_send_notification(self, notification: Notification) -> bool:
        """Check if notification should be sent based on preferences."""
        prefs = self.user_preferences.get(notification.recipient)
        if not prefs:
            return True

        if not prefs.enabled:
            return False

        # Check priority
        if notification.priority.value < prefs.min_priority.value:
            return False

        # Check quiet hours
        if prefs.quiet_hours_start is not None and prefs.quiet_hours_end is not None:
            current_hour = datetime.now().hour
            if prefs.quiet_hours_start <= current_hour <= prefs.quiet_hours_end:
                return False

        # Check muted keywords
        if prefs.muted_keywords:
            for keyword in prefs.muted_keywords:
                if keyword.lower() in notification.message.lower():
                    return False

        return True

    def _send_in_app(self, notification: Notification):
        """Send in-app notification."""
        # Store in session state
        if "in_app_notifications" not in st.session_state:
            st.session_state.in_app_notifications = []

        # Add to in-app notifications
        notif_data = {
            "id": notification.id,
            "title": notification.title,
            "message": notification.message,
            "priority": notification.priority.name,
            "timestamp": notification.timestamp,
            "read": False,
        }
        st.session_state.in_app_notifications.append(notif_data)

        # Keep last 100
        if len(st.session_state.in_app_notifications) > 100:
            st.session_state.in_app_notifications = (
                st.session_state.in_app_notifications[-100:]
            )

    def _send_email(self, notification: Notification):
        """Send email notification."""
        # Placeholder - would integrate with email service
        print(f"📧 EMAIL to {notification.recipient}: {notification.title}")

        # In production, this would use SMTP or email service API
        try:
            # Example SMTP (commented out)
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login('sender@gmail.com', 'password')
            # message = f"Subject: {notification.title}\n\n{notification.message}"
            # server.sendmail('sender@gmail.com', notification.recipient, message)
            # server.quit()
            pass
        except Exception as e:
            print(f"Email error: {e}")
            raise

    def _send_slack(self, notification: Notification):
        """Send Slack notification."""
        # Placeholder - would integrate with Slack API
        print(f"💬 SLACK to {notification.recipient}: {notification.title}")

        # In production, this would use Slack webhook
        # webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        # if webhook_url:
        #     payload = {
        #         "text": f"*{notification.title}*\n{notification.message}",
        #         "attachments": [{"color": self._get_slack_color(notification.priority)}]
        #     }
        #     requests.post(webhook_url, json=payload)

    def _send_webhook(self, notification: Notification):
        """Send webhook notification."""
        # Placeholder - would integrate with webhook service
        print(f"🌐 WEBHOOK: {notification.title}")

        # In production, this would send to configured webhook URL
        # webhook_url = os.getenv("WEBHOOK_URL")
        # if webhook_url:
        #     payload = {
        #         "id": notification.id,
        #         "title": notification.title,
        #         "message": notification.message,
        #         "priority": notification.priority.name,
        #         "timestamp": notification.timestamp
        #     }
        #     requests.post(webhook_url, json=payload)

    def _get_slack_color(self, priority: NotificationPriority) -> str:
        """Get Slack color for priority."""
        colors = {
            NotificationPriority.CRITICAL: "danger",
            NotificationPriority.HIGH: "warning",
            NotificationPriority.NORMAL: "good",
            NotificationPriority.LOW: "#808080",
        }
        return colors.get(priority, "good")

    def send_notification(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        recipient: str = "all",
        metadata: Dict[str, Any] = None,
        actions: List[Dict[str, str]] = None,
    ) -> str:
        """
        Send a notification.

        Args:
            title: Notification title
            message: Notification message
            priority: Notification priority
            channel: Delivery channel
            recipient: Recipient ID
            metadata: Additional metadata
            actions: Action buttons

        Returns:
            str: Notification ID
        """
        notification_id = f"notif_{int(time.time())}"

        notification = Notification(
            id=notification_id,
            title=title,
            message=message,
            priority=priority,
            channel=channel,
            recipient=recipient,
            timestamp=time.time(),
            status=NotificationStatus.PENDING,
            metadata=metadata or {},
            actions=actions or [],
        )

        # Add to queue
        self.notification_queue.put(notification)

        return notification_id

    def send_plagiarism_alert(
        self,
        doc_a: str,
        doc_b: str,
        similarity: float,
        threshold: float = 0.75,
        recipients: List[str] = None,
    ):
        """
        Send plagiarism detection alert.

        Args:
            doc_a: Document A name
            doc_b: Document B name
            similarity: Similarity score
            threshold: Threshold used
            recipients: List of recipients
        """
        # Determine priority
        if similarity >= 0.95:
            priority = NotificationPriority.CRITICAL
        elif similarity >= 0.85:
            priority = NotificationPriority.HIGH
        elif similarity >= threshold:
            priority = NotificationPriority.NORMAL
        else:
            return  # Below threshold, don't send

        title = f"🚨 Plagiarism Detected: {similarity:.1%} similarity"
        message = f"Document '{doc_a}' and '{doc_b}' have {similarity:.1%} similarity (threshold: {threshold:.0%})"

        # Send to each recipient
        recipients = recipients or ["all"]
        for recipient in recipients:
            self.send_notification(
                title=title,
                message=message,
                priority=priority,
                channel=NotificationChannel.IN_APP,
                recipient=recipient,
                metadata={
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "similarity": similarity,
                    "threshold": threshold,
                    "type": "plagiarism_alert",
                },
            )

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        self._save_data()

    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        self.alert_rules = [r for r in self.alert_rules if r.id != rule_id]
        self._save_data()

    def set_user_preferences(
        self, user_id: str, preferences: UserNotificationPreferences
    ):
        """Set user notification preferences."""
        self.user_preferences[user_id] = preferences
        self._save_data()

    def get_user_preferences(
        self, user_id: str
    ) -> Optional[UserNotificationPreferences]:
        """Get user notification preferences."""
        return self.user_preferences.get(user_id)

    def get_notifications(
        self,
        user_id: str = None,
        status: NotificationStatus = None,
        limit: int = 50,
        include_read: bool = False,
    ) -> List[Notification]:
        """
        Get notifications for a user.

        Args:
            user_id: User ID
            status: Filter by status
            limit: Maximum number of notifications
            include_read: Include read notifications

        Returns:
            List[Notification]: Notifications
        """
        # Filter notifications
        filtered = []

        for notification in reversed(self.notifications):
            # Filter by user
            if user_id and notification.recipient not in ["all", user_id]:
                continue

            # Filter by status
            if status and notification.status != status:
                continue

            # Filter read status
            if not include_read and notification.status == NotificationStatus.READ:
                continue

            filtered.append(notification)

            if len(filtered) >= limit:
                break

        return filtered

    def mark_as_read(self, notification_id: str):
        """Mark notification as read."""
        for notification in self.notifications:
            if notification.id == notification_id:
                notification.status = NotificationStatus.READ
                notification.read_at = time.time()
                self._save_data()
                break

    def mark_all_read(self, user_id: str):
        """Mark all notifications as read for a user."""
        for notification in self.notifications:
            if notification.recipient in ["all", user_id]:
                if notification.status != NotificationStatus.READ:
                    notification.status = NotificationStatus.READ
                    notification.read_at = time.time()
        self._save_data()

    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications."""
        count = 0
        for notification in self.notifications:
            if notification.recipient in ["all", user_id]:
                if notification.status not in [
                    NotificationStatus.READ,
                    NotificationStatus.DISMISSED,
                ]:
                    count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        total = len(self.notifications)
        unread = len(
            [
                n
                for n in self.notifications
                if n.status
                not in [NotificationStatus.READ, NotificationStatus.DISMISSED]
            ]
        )

        # Count by priority
        by_priority = defaultdict(int)
        for notification in self.notifications:
            by_priority[notification.priority.name] += 1

        # Count by channel
        by_channel = defaultdict(int)
        for notification in self.notifications:
            by_channel[notification.channel.value] += 1

        return {
            "total": total,
            "unread": unread,
            "by_priority": dict(by_priority),
            "by_channel": dict(by_channel),
            "delivery_stats": self.delivery_stats,
        }


# ==============================================================================
# SMART FILTER
# ==============================================================================


class SmartFilter:
    """
    Smart filtering to reduce notification noise.
    """

    def __init__(self, notification_manager: NotificationManager):
        self.manager = notification_manager
        self.similarity_history: Dict[str, List[float]] = defaultdict(list)
        self.notification_cooldown: Dict[str, float] = {}

    def should_filter(self, event: Dict[str, Any], cooldown_seconds: int = 300) -> bool:
        """
        Check if an event should be filtered out.

        Args:
            event: Event data
            cooldown_seconds: Cooldown period in seconds

        Returns:
            bool: True if should filter (not send notification)
        """
        # Check for similar recent events
        event_key = self._get_event_key(event)

        # Check cooldown
        if event_key in self.notification_cooldown:
            elapsed = time.time() - self.notification_cooldown[event_key]
            if elapsed < cooldown_seconds:
                return True

        # Update cooldown
        self.notification_cooldown[event_key] = time.time()

        return False

    def _get_event_key(self, event: Dict[str, Any]) -> str:
        """Generate event key for deduplication."""
        # For plagiarism events, use document pair
        if "doc_a" in event and "doc_b" in event:
            docs = sorted([event["doc_a"], event["doc_b"]])
            return f"plagiarism_{docs[0]}_{docs[1]}"

        # Default: use event type
        return event.get("type", "default")

    def should_send_based_on_history(
        self,
        doc_a: str,
        doc_b: str,
        similarity: float,
        min_interval: int = 3600,  # 1 hour
    ) -> bool:
        """
        Check if notification should be sent based on history.

        Args:
            doc_a: Document A
            doc_b: Document B
            similarity: Current similarity
            min_interval: Minimum interval between notifications

        Returns:
            bool: True if should send notification
        """
        key = f"{doc_a}_{doc_b}"

        # Check history
        if key in self.similarity_history:
            history = self.similarity_history[key]

            # Check if similarity is significantly different from previous
            if history:
                avg_history = sum(history) / len(history)

                # Only notify if similarity changed significantly
                if abs(similarity - avg_history) < 0.05:
                    return False

        # Update history
        self.similarity_history[key].append(similarity)

        # Keep last 10 entries
        if len(self.similarity_history[key]) > 10:
            self.similarity_history[key] = self.similarity_history[key][-10:]

        return True


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_notification_center():
    """Render notification center UI."""
    st.subheader("🔔 Notification Center")

    # Initialize notification manager
    if "notification_manager" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        st.session_state.notification_manager = NotificationManager(
            data_dir / "notifications"
        )

    manager = st.session_state.notification_manager
    user = st.session_state.get("username", "anonymous")

    # Notification header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        unread = manager.get_unread_count(user)
        st.markdown(f"### 📬 Notifications ({unread} unread)")

    with col2:
        if unread > 0:
            if st.button("✅ Mark All Read", use_container_width=True):
                manager.mark_all_read(user)
                st.rerun()

    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔔 All", "📋 Unread", "⚙️ Settings"])

    with tab1:
        render_notification_list(manager, user, include_read=True)

    with tab2:
        render_notification_list(manager, user, include_read=False)

    with tab3:
        render_notification_settings(manager, user)


def render_notification_list(
    manager: NotificationManager, user: str, include_read: bool = True
):
    """Render notification list."""
    notifications = manager.get_notifications(
        user_id=user, limit=50, include_read=include_read
    )

    if not notifications:
        st.info("No notifications found")
        return

    # Priority colors
    priority_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "NORMAL": "🔵", "LOW": "⚪"}

    for notification in notifications:
        # Determine if read
        is_read = notification.status == NotificationStatus.READ

        # Priority icon
        icon = priority_colors.get(notification.priority.name, "ℹ️")

        # Create expander
        with st.expander(
            f"{icon} {notification.title} - {datetime.fromtimestamp(notification.timestamp).strftime('%H:%M')}",
            expanded=not is_read,
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(notification.message)

                if notification.metadata:
                    st.caption("Metadata:")
                    for key, value in notification.metadata.items():
                        st.caption(f"• {key}: {value}")

                if notification.actions:
                    cols = st.columns(len(notification.actions))
                    for col, action in zip(cols, notification.actions):
                        if col.button(
                            action.get("label", "Action"),
                            key=f"notif_{notification.id}_{action.get('id', '')}",
                        ):
                            # Handle action
                            pass

                st.caption(
                    f"Priority: {notification.priority.name} | Channel: {notification.channel.value}"
                )

            with col2:
                if not is_read:
                    if st.button("✅ Read", key=f"read_{notification.id}"):
                        manager.mark_as_read(notification.id)
                        st.rerun()

                st.caption(f"Status: {notification.status.value}")

                if notification.read_at:
                    st.caption(
                        f"Read: {datetime.fromtimestamp(notification.read_at).strftime('%H:%M:%S')}"
                    )


def render_notification_settings(manager: NotificationManager, user: str):
    """Render notification settings."""
    st.markdown("#### ⚙️ Notification Settings")

    # Get user preferences
    prefs = manager.get_user_preferences(user)
    if not prefs:
        prefs = UserNotificationPreferences(user_id=user)

    # Settings form
    with st.form("notification_settings_form"):
        st.markdown("##### General Settings")
        enabled = st.checkbox("Enable Notifications", value=prefs.enabled)

        st.markdown("##### Channels")
        channels = st.multiselect(
            "Notification Channels",
            options=[c.value for c in NotificationChannel],
            default=[c.value for c in prefs.channels],
        )

        st.markdown("##### Priority")
        min_priority = st.selectbox(
            "Minimum Priority Level",
            options=[p.name for p in NotificationPriority],
            index=list(NotificationPriority).index(prefs.min_priority),
        )

        st.markdown("##### Quiet Hours")
        col1, col2 = st.columns(2)
        with col1:
            quiet_start = st.number_input(
                "Quiet Hours Start", 0, 23, prefs.quiet_hours_start or 22
            )
        with col2:
            quiet_end = st.number_input(
                "Quiet Hours End", 0, 23, prefs.quiet_hours_end or 6
            )

        quiet_enabled = st.checkbox(
            "Enable Quiet Hours", value=prefs.quiet_hours_start is not None
        )

        st.markdown("##### Muted Keywords")
        muted_keywords = st.text_area(
            "Muted Keywords (one per line)", value="\n".join(prefs.muted_keywords)
        )

        st.markdown("##### Digest Settings")
        col1, col2 = st.columns(2)
        with col1:
            daily_digest = st.checkbox("Daily Digest", value=prefs.daily_digest)
        with col2:
            weekly_summary = st.checkbox("Weekly Summary", value=prefs.weekly_summary)

        # Submit button
        if st.form_submit_button("💾 Save Settings", use_container_width=True):
            # Update preferences
            updated_prefs = UserNotificationPreferences(
                user_id=user,
                enabled=enabled,
                channels=[NotificationChannel(c) for c in channels],
                min_priority=NotificationPriority[min_priority],
                quiet_hours_start=quiet_start if quiet_enabled else None,
                quiet_hours_end=quiet_end if quiet_enabled else None,
                muted_keywords=[
                    k.strip() for k in muted_keywords.split("\n") if k.strip()
                ],
                daily_digest=daily_digest,
                weekly_summary=weekly_summary,
            )

            manager.set_user_preferences(user, updated_prefs)
            st.success("✅ Settings saved successfully!")
            st.rerun()


def render_notification_stats():
    """Render notification statistics."""
    st.subheader("📊 Notification Analytics")

    if "notification_manager" not in st.session_state:
        st.info("No notification data available")
        return

    manager = st.session_state.notification_manager
    stats = manager.get_stats()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Notifications", stats["total"])
    col2.metric("Unread", stats["unread"])
    col3.metric("Sent", stats["delivery_stats"]["sent"])
    col4.metric("Failed", stats["delivery_stats"]["failed"])

    # Charts
    fig = make_subplots(rows=1, cols=2, subplot_titles=("By Priority", "By Channel"))  # noqa: F821

    # Priority chart
    if stats["by_priority"]:
        priorities = list(stats["by_priority"].keys())
        counts = list(stats["by_priority"].values())
        fig.add_trace(go.Bar(x=priorities, y=counts, marker_color="blue"), row=1, col=1)

    # Channel chart
    if stats["by_channel"]:
        channels = list(stats["by_channel"].keys())
        counts = list(stats["by_channel"].values())
        fig.add_trace(go.Bar(x=channels, y=counts, marker_color="green"), row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_notification_badge():
    """Render notification badge in sidebar."""
    if "notification_manager" in st.session_state:
        manager = st.session_state.notification_manager
        user = st.session_state.get("username", "anonymous")

        unread = manager.get_unread_count(user)

        if unread > 0:
            st.sidebar.markdown(
                f"""
                <div style="
                    background-color: #ff4b4b;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 20px;
                    text-align: center;
                    font-weight: bold;
                    margin: 10px 0;
                ">
                    🔔 {unread} New Notification{"s" if unread > 1 else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_notifications():
    """Initialize notification system."""
    if "notification_initialized" not in st.session_state:
        st.session_state.notification_initialized = True

        # Create notification manager
        data_dir = Path(st.session_state.get("data_dir", "."))
        manager = NotificationManager(data_dir / "notifications")
        st.session_state.notification_manager = manager
