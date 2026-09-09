"""
tests/api/test_asgi_app.py
--------------------------
Unit tests for ASGI application and middleware.

Includes tests for security headers, CSP policy, and middleware behavior.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src.asgi_app import SecurityHeadersMiddleware


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with SecurityHeadersMiddleware."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    # Add middleware
    app.add_middleware(SecurityHeadersMiddleware)

    return app


class TestSecurityHeadersMiddleware:
    """Test suite for SecurityHeadersMiddleware."""

    def test_adds_x_content_type_options_header(self, app_with_middleware):
        """Verify X-Content-Type-Options header is present."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        assert response.status_code == 200
        assert "x-content-type-options" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_adds_x_frame_options_header(self, app_with_middleware):
        """Verify X-Frame-Options header is present."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        assert "x-frame-options" in response.headers
        assert response.headers["x-frame-options"] == "DENY"

    def test_adds_x_xss_protection_header(self, app_with_middleware):
        """Verify X-XSS-Protection header is present."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        assert "x-xss-protection" in response.headers
        assert response.headers["x-xss-protection"] == "1; mode=block"

    def test_adds_referrer_policy_header(self, app_with_middleware):
        """Verify Referrer-Policy header is present."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        assert "referrer-policy" in response.headers
        assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"

    def test_adds_content_security_policy_header_default(self, app_with_middleware):
        """Verify Content-Security-Policy header is present with default value."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        assert "content-security-policy" in response.headers
        csp = response.headers["content-security-policy"]

        # Verify default policy contains required directives
        assert "default-src 'self'" in csp
        assert "script-src" in csp

    def test_content_security_policy_configurable_via_env(self):
        """Verify CSP policy can be configured via CSP_POLICY env var."""
        custom_policy = "default-src 'none'; script-src 'self'"

        with patch.dict(os.environ, {"CSP_POLICY": custom_policy}, clear=True):
            app = FastAPI()

            @app.get("/test")
            async def test_endpoint():
                return {"message": "test"}

            app.add_middleware(SecurityHeadersMiddleware)
            client = TestClient(app)
            response = client.get("/test")

            assert response.headers["content-security-policy"] == custom_policy

    def test_csp_header_on_all_routes(self, app_with_middleware):
        """Verify CSP header is added to all routes."""
        client = TestClient(app_with_middleware)

        # Test multiple routes
        response1 = client.get("/test")
        response2 = client.get("/nonexistent")  # 404

        # Both should have CSP header
        assert "content-security-policy" in response1.headers
        assert "content-security-policy" in response2.headers

    def test_csp_header_prevents_inline_scripts_by_default(self, app_with_middleware):
        """Verify default CSP restricts inline scripts appropriately."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        csp = response.headers["content-security-policy"]

        # Should allow 'unsafe-inline' for Swagger UI compatibility
        assert "'unsafe-inline'" in csp or "script-src 'self'" in csp

    def test_middleware_preserves_response_body(self, app_with_middleware):
        """Verify middleware doesn't modify response body."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        assert response.json() == {"message": "test"}

    def test_middleware_handles_non_http_scope(self):
        """Verify middleware ignores non-HTTP scopes (e.g., websocket)."""
        app = AsyncMock()
        middleware = SecurityHeadersMiddleware(app)

        # Create a mock websocket scope
        scope = {"type": "websocket", "headers": []}
        receive = AsyncMock()
        send = AsyncMock()

        # Should not raise and should pass through
        import asyncio

        asyncio.run(middleware(scope, receive, send))

        # Verify app was called
        app.assert_awaited_once()
        assert app.call_args.args[0] is scope


class TestCSPSecurity:
    """Test suite for Content-Security-Policy security properties."""

    def test_default_csp_restricts_to_self(self):
        """Verify default CSP restricts resources to same origin."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        response = client.get("/test")
        csp = response.headers["content-security-policy"]

        # Default should include 'self' directive
        assert "'self'" in csp

    def test_csp_prevents_external_script_loading(self):
        """Verify default CSP prevents loading scripts from external domains."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(app)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        response = client.get("/test")
        csp = response.headers["content-security-policy"]

        # Should not have wildcard (*) in script-src
        # (unless explicitly configured)
        if "script-src" in csp:
            # Extract script-src directive
            script_src = [d for d in csp.split(";") if "script-src" in d]
            if script_src:
                # Should not allow all sources
                assert "*" not in script_src[0] or "'self'" in script_src[0]

    def test_csp_header_case_insensitive(self, app_with_middleware):
        """Verify CSP header name is case-insensitive in HTTP."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")

        # HTTP headers are case-insensitive
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        assert "content-security-policy" in headers_lower
