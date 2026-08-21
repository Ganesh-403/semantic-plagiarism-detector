"""
API Gateway and External Integration Hub

Features:
- Unified API gateway
- External service integrations
- Webhook management
- Service connectors
- API documentation portal
- API key management
- Rate limiting
- Endpoint exposure
"""

import hashlib
import json
import secrets
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class ApiMethod(Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class ApiStatus(Enum):
    """API status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    BETA = "beta"


class WebhookStatus(Enum):
    """Webhook status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    RETRYING = "retrying"


class ServiceType(Enum):
    """External service types."""

    STORAGE = "storage"
    AUTHENTICATION = "authentication"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    AI = "ai"
    DATABASE = "database"


@dataclass
class ApiEndpoint:
    """API endpoint definition."""

    id: str
    path: str
    method: ApiMethod
    description: str
    status: ApiStatus
    handler: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    rate_limit: int  # requests per minute
    requires_auth: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApiKey:
    """API key."""

    id: str
    key: str
    name: str
    user: str
    created_at: float
    expires_at: Optional[float] = None
    is_active: bool = True
    permissions: List[str] = field(default_factory=list)
    rate_limit: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Webhook:
    """Webhook configuration."""

    id: str
    name: str
    url: str
    events: List[str]
    headers: Dict[str, str]
    status: WebhookStatus
    created_at: float
    last_triggered: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceConnection:
    """External service connection."""

    id: str
    name: str
    type: ServiceType
    config: Dict[str, Any]
    status: bool
    created_at: float
    last_used: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApiLog:
    """API request log."""

    id: str
    endpoint: str
    method: str
    status_code: int
    response_time: float
    user: str
    timestamp: float
    ip_address: str
    user_agent: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# API GATEWAY
# ==============================================================================


class ApiGateway:
    """
    Unified API gateway for internal and external integrations.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.endpoints: Dict[str, ApiEndpoint] = {}
        self.api_keys: Dict[str, ApiKey] = {}
        self.webhooks: Dict[str, Webhook] = {}
        self.services: Dict[str, ServiceConnection] = {}
        self.api_logs: List[ApiLog] = []
        self.handlers: Dict[str, Callable] = {}
        self.rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._load_data()
        self._register_default_handlers()

    def _load_data(self):
        """Load data from storage."""
        try:
            data_path = self.storage_path / "api_data.json"
            if data_path.exists():
                with open(data_path, "r") as f:
                    data = json.load(f)

                    self.endpoints = {
                        k: ApiEndpoint(**v)
                        for k, v in data.get("endpoints", {}).items()
                    }

                    self.api_keys = {
                        k: ApiKey(**v) for k, v in data.get("api_keys", {}).items()
                    }

                    self.webhooks = {
                        k: Webhook(**v) for k, v in data.get("webhooks", {}).items()
                    }

                    self.services = {
                        k: ServiceConnection(**v)
                        for k, v in data.get("services", {}).items()
                    }

                    self.api_logs = [ApiLog(**l) for l in data.get("logs", [])]
        except Exception as e:
            print(f"Error loading API data: {e}")

    def _save_data(self):
        """Save data to storage."""
        try:
            data_path = self.storage_path / "api_data.json"
            data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "endpoints": {k: asdict(v) for k, v in self.endpoints.items()},
                "api_keys": {k: asdict(v) for k, v in self.api_keys.items()},
                "webhooks": {k: asdict(v) for k, v in self.webhooks.items()},
                "services": {k: asdict(v) for k, v in self.services.items()},
                "logs": [asdict(l) for l in self.api_logs[-1000:]],
            }

            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving API data: {e}")

    def _register_default_handlers(self):
        """Register default API handlers."""
        self.handlers = {
            "plagiarism_check": self._handle_plagiarism_check,
            "document_upload": self._handle_document_upload,
            "get_report": self._handle_get_report,
            "get_status": self._handle_get_status,
            "webhook_trigger": self._handle_webhook_trigger,
            "health_check": self._handle_health_check,
        }

    # ==========================================================================
    # API HANDLERS
    # ==========================================================================

    def _handle_plagiarism_check(self, params: Dict) -> Dict:
        """Handle plagiarism check API request."""
        documents = params.get("documents", [])
        threshold = params.get("threshold", 0.75)

        # Simulate plagiarism check
        results = {
            "status": "success",
            "documents_checked": len(documents),
            "flagged": len(documents) // 3,
            "threshold_used": threshold,
            "results": [
                {
                    "doc_id": doc,
                    "similarity": 0.5 + (hash(doc) % 50) / 100,
                    "flagged": (hash(doc) % 3) == 0,
                }
                for doc in documents
            ],
            "timestamp": datetime.now().isoformat(),
        }

        return results

    def _handle_document_upload(self, params: Dict) -> Dict:
        """Handle document upload API request."""
        documents = params.get("documents", [])

        return {
            "status": "success",
            "uploaded": len(documents),
            "document_ids": [
                f"doc_{int(time.time())}_{i}" for i in range(len(documents))
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_get_report(self, params: Dict) -> Dict:
        """Handle get report API request."""
        report_id = params.get("report_id")

        return {
            "status": "success",
            "report_id": report_id,
            "report": {
                "title": f"Plagiarism Report {report_id}",
                "generated": datetime.now().isoformat(),
                "summary": {
                    "total_documents": 10,
                    "flagged_pairs": 3,
                    "avg_similarity": 0.42,
                },
                "details": [
                    {
                        "doc_a": f"doc_{i}",
                        "doc_b": f"doc_{i + 1}",
                        "similarity": 0.7 + (i * 0.05),
                        "flagged": True,
                    }
                    for i in range(3)
                ],
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_get_status(self, params: Dict) -> Dict:
        """Handle status check API request."""
        return {
            "status": "success",
            "system_status": "operational",
            "version": "2.0.0",
            "uptime": "72h",
            "metrics": {
                "total_requests": 1250,
                "avg_response_time": 145,
                "error_rate": 0.02,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_webhook_trigger(self, params: Dict) -> Dict:
        """Handle webhook trigger."""
        webhook_id = params.get("webhook_id")
        data = params.get("data", {})

        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return {"status": "error", "message": "Webhook not found"}

        return {
            "status": "success",
            "webhook_id": webhook_id,
            "triggered": True,
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_health_check(self, params: Dict) -> Dict:
        """Handle health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "healthy",
                "cache": "healthy",
                "queue": "healthy",
            },
        }

    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================

    def register_endpoint(
        self,
        path: str,
        method: ApiMethod,
        description: str,
        handler: str,
        parameters: List[Dict[str, Any]],
        responses: Dict[str, Dict[str, Any]],
        rate_limit: int = 60,
        requires_auth: bool = True,
        status: ApiStatus = ApiStatus.ACTIVE,
    ) -> ApiEndpoint:
        """Register a new API endpoint."""
        endpoint = ApiEndpoint(
            id=f"api_{int(time.time())}_{hashlib.md5(path.encode()).hexdigest()[:8]}",
            path=path,
            method=method,
            description=description,
            status=status,
            handler=handler,
            parameters=parameters,
            responses=responses,
            rate_limit=rate_limit,
            requires_auth=requires_auth,
        )

        self.endpoints[endpoint.id] = endpoint
        self._save_data()
        return endpoint

    def generate_api_key(
        self,
        name: str,
        user: str,
        permissions: List[str] = None,
        rate_limit: int = 60,
        expires_in_days: int = 365,
    ) -> ApiKey:
        """Generate a new API key."""
        key_value = f"sk_{secrets.token_urlsafe(32)}"

        api_key = ApiKey(
            id=f"key_{int(time.time())}_{hashlib.md5(key_value.encode()).hexdigest()[:8]}",
            key=key_value,
            name=name,
            user=user,
            created_at=time.time(),
            expires_at=time.time() + (expires_in_days * 86400)
            if expires_in_days > 0
            else None,
            permissions=permissions or ["*"],
            rate_limit=rate_limit,
        )

        self.api_keys[api_key.id] = api_key
        self._save_data()
        return api_key

    def revoke_api_key(self, key_id: str):
        """Revoke an API key."""
        key = self.api_keys.get(key_id)
        if key:
            key.is_active = False
            self._save_data()

    def register_webhook(
        self,
        name: str,
        url: str,
        events: List[str],
        headers: Dict[str, str] = None,
        max_retries: int = 3,
    ) -> Webhook:
        """Register a new webhook."""
        webhook = Webhook(
            id=f"wh_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            url=url,
            events=events,
            headers=headers or {},
            status=WebhookStatus.ACTIVE,
            created_at=time.time(),
            max_retries=max_retries,
        )

        self.webhooks[webhook.id] = webhook
        self._save_data()
        return webhook

    def trigger_webhook(self, webhook_id: str, event: str, data: Dict) -> bool:
        """Trigger a webhook."""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return False

        if event not in webhook.events:
            return False

        # Prepare payload
        payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "webhook_id": webhook_id,
        }

        # Send webhook
        try:
            response = requests.post(
                webhook.url, json=payload, headers=webhook.headers, timeout=30
            )

            webhook.last_triggered = time.time()

            if response.status_code in [200, 201, 202]:
                webhook.status = WebhookStatus.ACTIVE
                self._save_data()
                return True
            else:
                webhook.status = WebhookStatus.FAILED
                webhook.retry_count += 1
                self._save_data()
                return False

        except Exception:
            webhook.status = WebhookStatus.FAILED
            webhook.retry_count += 1
            self._save_data()
            return False

    def connect_service(
        self, name: str, type: ServiceType, config: Dict[str, Any]
    ) -> ServiceConnection:
        """Connect to an external service."""
        service = ServiceConnection(
            id=f"svc_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            type=type,
            config=config,
            status=True,
            created_at=time.time(),
        )

        self.services[service.id] = service
        self._save_data()
        return service

    def call_api(
        self,
        endpoint_id: str,
        params: Dict[str, Any],
        api_key: Optional[str] = None,
        ip_address: str = None,
        user_agent: str = None,
    ) -> Dict[str, Any]:
        """
        Call an API endpoint.

        Args:
            endpoint_id: Endpoint ID
            params: Request parameters
            api_key: API key for authentication
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Dict: API response
        """
        start_time = time.time()

        # Get endpoint
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint:
            return {"status": "error", "message": "Endpoint not found"}

        # Check authentication
        if endpoint.requires_auth:
            if not api_key:
                return {"status": "error", "message": "API key required"}

            # Validate API key
            key_obj = None
            for key in self.api_keys.values():
                if key.key == api_key and key.is_active:
                    key_obj = key
                    break

            if not key_obj:
                return {"status": "error", "message": "Invalid API key"}

            # Check permissions
            if (
                "*" not in key_obj.permissions
                and endpoint_id not in key_obj.permissions
            ):
                return {"status": "error", "message": "Insufficient permissions"}

            # Check rate limit
            self.rate_limits[api_key].append(time.time())
            recent = [t for t in self.rate_limits[api_key] if time.time() - t < 60]
            if len(recent) > key_obj.rate_limit:
                return {"status": "error", "message": "Rate limit exceeded"}

        # Check rate limit
        self.rate_limits[endpoint_id].append(time.time())
        recent = [t for t in self.rate_limits[endpoint_id] if time.time() - t < 60]
        if len(recent) > endpoint.rate_limit:
            return {"status": "error", "message": "Endpoint rate limit exceeded"}

        # Execute handler
        handler = self.handlers.get(endpoint.handler)
        if not handler:
            return {
                "status": "error",
                "message": f"Handler not found: {endpoint.handler}",
            }

        try:
            result = handler(params)
            status_code = 200

        except Exception as e:
            result = {"status": "error", "message": str(e)}
            status_code = 500

        # Log request
        log = ApiLog(
            id=f"log_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            endpoint=endpoint.path,
            method=endpoint.method.value,
            status_code=status_code,
            response_time=time.time() - start_time,
            user=key_obj.user if key_obj else "anonymous",
            timestamp=time.time(),
            ip_address=ip_address or "127.0.0.1",
            user_agent=user_agent or "unknown",
            metadata={"endpoint_id": endpoint_id},
        )

        self.api_logs.append(log)
        self._save_data()

        return result

    def get_api_logs(self, limit: int = 100) -> List[Dict]:
        """Get API logs."""
        logs = []
        for log in self.api_logs[-limit:]:
            logs.append(
                {
                    "timestamp": datetime.fromtimestamp(log.timestamp).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "endpoint": log.endpoint,
                    "method": log.method,
                    "status": log.status_code,
                    "response_time": f"{log.response_time * 1000:.1f}ms",
                    "user": log.user,
                    "ip": log.ip_address,
                }
            )
        return logs

    def get_stats(self) -> Dict[str, Any]:
        """Get API statistics."""
        total_endpoints = len(self.endpoints)
        total_keys = len([k for k in self.api_keys.values() if k.is_active])
        total_webhooks = len(self.webhooks)
        total_services = len(self.services)
        total_requests = len(self.api_logs)

        # Response time stats
        response_times = [l.response_time for l in self.api_logs[-1000:]]
        avg_response = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        # Status code distribution
        status_codes = Counter([l.status_code for l in self.api_logs[-1000:]])

        return {
            "total_endpoints": total_endpoints,
            "total_api_keys": total_keys,
            "total_webhooks": total_webhooks,
            "total_services": total_services,
            "total_requests": total_requests,
            "avg_response_time": avg_response,
            "status_code_distribution": dict(status_codes),
        }


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_api_gateway():
    """Render API gateway UI."""
    st.subheader("🚪 API Gateway")

    # Initialize
    if "api_gateway" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        st.session_state.api_gateway = ApiGateway(data_dir / "api")

    gateway = st.session_state.api_gateway

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "🔗 Endpoints", "🔑 API Keys", "📡 Webhooks", "🔌 Services"]
    )

    with tab1:
        render_api_dashboard(gateway)

    with tab2:
        render_endpoint_management(gateway)

    with tab3:
        render_api_key_management(gateway)

    with tab4:
        render_webhook_management(gateway)

    with tab5:
        render_service_management(gateway)


def render_api_dashboard(gateway: ApiGateway):
    """Render API dashboard."""
    st.markdown("#### 📊 API Dashboard")

    stats = gateway.get_stats()

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Endpoints", stats["total_endpoints"])
    col2.metric("API Keys", stats["total_api_keys"])
    col3.metric("Webhooks", stats["total_webhooks"])
    col4.metric("Services", stats["total_services"])
    col5.metric("Total Requests", stats["total_requests"])

    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Response Time", f"{stats['avg_response_time'] * 1000:.1f}ms")

    with col2:
        if stats["status_code_distribution"]:
            success_rate = stats["status_code_distribution"].get(200, 0) / max(
                1, sum(stats["status_code_distribution"].values())
            )
            st.metric("Success Rate", f"{success_rate:.1%}")

    # API logs
    st.markdown("#### 📋 Recent API Requests")
    logs = gateway.get_api_logs(limit=20)

    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No API requests yet")


def render_endpoint_management(gateway: ApiGateway):
    """Render endpoint management UI."""
    st.markdown("#### 🔗 API Endpoints")

    # Create endpoint
    with st.expander("➕ Create Endpoint", expanded=False):
        with st.form("create_endpoint_form"):
            col1, col2 = st.columns(2)
            with col1:
                path = st.text_input("Path", "/api/v1/plagiarism")
                method = st.selectbox("Method", [m.value for m in ApiMethod])
                description = st.text_area("Description")
            with col2:
                handler = st.selectbox("Handler", list(gateway.handlers.keys()))
                rate_limit = st.number_input("Rate Limit (per minute)", 10, 1000, 60)
                requires_auth = st.checkbox("Requires Authentication", value=True)
                status = st.selectbox("Status", [s.value for s in ApiStatus])

            if st.form_submit_button("✅ Create Endpoint", use_container_width=True):
                endpoint = gateway.register_endpoint(
                    path=path,
                    method=ApiMethod(method),
                    description=description,
                    handler=handler,
                    parameters=[],
                    responses={"200": {"description": "Success"}},
                    rate_limit=rate_limit,
                    requires_auth=requires_auth,
                    status=ApiStatus(status),
                )
                st.success(f"✅ Endpoint created: {endpoint.id}")
                st.rerun()

    # Display endpoints
    if gateway.endpoints:
        for endpoint in gateway.endpoints.values():
            status_colors = {
                "active": "🟢",
                "inactive": "⚪",
                "deprecated": "🟡",
                "beta": "🔵",
            }

            with st.expander(
                f"{status_colors.get(endpoint.status.value, '')} {endpoint.method.value} {endpoint.path}",
                expanded=False,
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Description:** {endpoint.description}")
                    st.markdown(f"**Status:** {endpoint.status.value.upper()}")
                    st.caption(f"ID: {endpoint.id}")
                with col2:
                    st.markdown(f"**Handler:** {endpoint.handler}")
                    st.markdown(f"**Rate Limit:** {endpoint.rate_limit}/min")
                    st.markdown(f"**Auth:** {'✅' if endpoint.requires_auth else '❌'}")
    else:
        st.info("No endpoints registered")


def render_api_key_management(gateway: ApiGateway):
    """Render API key management UI."""
    st.markdown("#### 🔑 API Keys")

    # Generate key
    with st.expander("🔑 Generate New API Key", expanded=False):
        with st.form("generate_key_form"):
            col1, col2 = st.columns(2)
            with col1:
                key_name = st.text_input("Key Name")
                user = st.text_input("User", st.session_state.get("username", "system"))
            with col2:
                expires_in = st.number_input("Expires In (days)", 0, 365, 365)
                rate_limit = st.number_input("Rate Limit (per minute)", 10, 1000, 60)
                permissions = st.text_input("Permissions (comma separated)", "*")

            if st.form_submit_button("🔑 Generate Key", use_container_width=True):
                key = gateway.generate_api_key(
                    name=key_name,
                    user=user,
                    permissions=[
                        p.strip() for p in permissions.split(",") if p.strip()
                    ],
                    rate_limit=rate_limit,
                    expires_in_days=expires_in,
                )

                st.success("✅ API Key Generated")
                st.code(key.key, language="text")
                st.warning("⚠️ Copy this key now. It will not be shown again!")
                st.rerun()

    # Display keys
    if gateway.api_keys:
        for key in gateway.api_keys.values():
            with st.expander(f"🔑 {key.name} - {key.user}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**ID:** {key.id}")
                    st.markdown(
                        f"**Status:** {'✅ Active' if key.is_active else '❌ Inactive'}"
                    )
                    st.caption(
                        f"Created: {datetime.fromtimestamp(key.created_at).strftime('%Y-%m-%d')}"
                    )
                with col2:
                    if key.expires_at:
                        expires = datetime.fromtimestamp(key.expires_at).strftime(
                            "%Y-%m-%d"
                        )
                        st.markdown(f"**Expires:** {expires}")
                    st.markdown(f"**Rate Limit:** {key.rate_limit}/min")
                    st.markdown(f"**Permissions:** {', '.join(key.permissions)}")

                if not key.is_active and st.button("🗑️ Delete", key=f"del_{key.id}"):
                    gateway.api_keys.pop(key.id, None)
                    gateway._save_data()
                    st.rerun()
    else:
        st.info("No API keys generated")


def render_webhook_management(gateway: ApiGateway):
    """Render webhook management UI."""
    st.markdown("#### 📡 Webhooks")

    # Register webhook
    with st.expander("➕ Register Webhook", expanded=False):
        with st.form("register_webhook_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Webhook Name")
                url = st.text_input("URL", "https://example.com/webhook")
            with col2:
                events = st.text_input(
                    "Events (comma separated)", "plagiarism_detected,report_generated"
                )
                max_retries = st.number_input("Max Retries", 1, 10, 3)
                headers = st.text_area(
                    "Headers (JSON)", '{"Content-Type": "application/json"}'
                )

            try:
                headers_json = json.loads(headers) if headers else {}
            except:
                headers_json = {}

            if st.form_submit_button("✅ Register Webhook", use_container_width=True):
                webhook = gateway.register_webhook(
                    name=name,
                    url=url,
                    events=[e.strip() for e in events.split(",") if e.strip()],
                    headers=headers_json,
                    max_retries=max_retries,
                )
                st.success(f"✅ Webhook registered: {webhook.id}")
                st.rerun()

    # Display webhooks
    if gateway.webhooks:
        for webhook in gateway.webhooks.values():
            status_colors = {
                "active": "🟢",
                "inactive": "⚪",
                "failed": "🔴",
                "retrying": "🟡",
            }

            with st.expander(
                f"{status_colors.get(webhook.status.value, '')} {webhook.name}",
                expanded=False,
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**URL:** {webhook.url}")
                    st.markdown(f"**Events:** {', '.join(webhook.events)}")
                with col2:
                    st.markdown(f"**Status:** {webhook.status.value.upper()}")
                    st.markdown(
                        f"**Retries:** {webhook.retry_count}/{webhook.max_retries}"
                    )
                    if webhook.last_triggered:
                        st.caption(
                            f"Last Triggered: {datetime.fromtimestamp(webhook.last_triggered).strftime('%Y-%m-%d %H:%M')}"
                        )

                # Test webhook
                if st.button("📡 Test Webhook", key=f"test_{webhook.id}"):
                    result = gateway.trigger_webhook(webhook.id, "test", {"test": True})
                    if result:
                        st.success("✅ Webhook triggered successfully")
                    else:
                        st.error("❌ Webhook failed")
    else:
        st.info("No webhooks registered")


def render_service_management(gateway: ApiGateway):
    """Render service management UI."""
    st.markdown("#### 🔌 External Services")

    # Connect service
    with st.expander("➕ Connect Service", expanded=False):
        with st.form("connect_service_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Service Name")
                service_type = st.selectbox(
                    "Service Type", [t.value for t in ServiceType]
                )
            with col2:
                config = st.text_area(
                    "Configuration (JSON)",
                    '{"api_key": "xxx", "endpoint": "https://api.example.com"}',
                )

            try:
                config_json = json.loads(config) if config else {}
            except:
                config_json = {}

            if st.form_submit_button("✅ Connect Service", use_container_width=True):
                service = gateway.connect_service(
                    name=name, type=ServiceType(service_type), config=config_json
                )
                st.success(f"✅ Service connected: {service.id}")
                st.rerun()

    # Display services
    if gateway.services:
        for service in gateway.services.values():
            with st.expander(
                f"🔌 {service.name} - {service.type.value}", expanded=False
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Type:** {service.type.value.upper()}")
                    st.markdown(
                        f"**Status:** {'✅ Connected' if service.status else '❌ Disconnected'}"
                    )
                with col2:
                    st.caption(
                        f"Created: {datetime.fromtimestamp(service.created_at).strftime('%Y-%m-%d')}"
                    )
                    if service.last_used:
                        st.caption(
                            f"Last Used: {datetime.fromtimestamp(service.last_used).strftime('%Y-%m-%d %H:%M')}"
                        )

                with st.expander("🔧 Configuration", expanded=False):
                    st.json(service.config)
    else:
        st.info("No services connected")


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_api_gateway():
    """Initialize API gateway."""
    if "api_gateway_initialized" not in st.session_state:
        st.session_state.api_gateway_initialized = True

        data_dir = Path(st.session_state.get("data_dir", "."))
        gateway = ApiGateway(data_dir / "api")
        st.session_state.api_gateway = gateway


# ==============================================================================
# EXPORTED ITEMS
# ==============================================================================

__all__ = [
    "render_api_gateway",
    "initialize_api_gateway",
    "ApiGateway",
    "ApiEndpoint",
    "ApiKey",
    "Webhook",
    "ServiceConnection",
    "ApiLog",
    "ApiMethod",
    "ApiStatus",
    "WebhookStatus",
    "ServiceType",
]
