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

"""ASGI entry point and request-security middleware.

The Streamlit dashboard is wrapped at the ASGI layer so security
policies can be enforced before a request reaches application code.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager

import streamlit as st
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.api.middleware import validate_bearer_tokens_config
from src.core.scheduler import start_scheduler, stop_scheduler
from src.utils.tracing import _tracer_provider, init_tracer_provider

DEFAULT_MAX_REQUEST_BYTES = 52_428_800
JSON_API_PREFIX = "/api/"
NON_JSON_API_PATHS = frozenset(
    {
        # File-upload endpoint validated separately as multipart.
        "/api/v1/scan",
    }
)


class ClientIPLoggingMiddleware(BaseHTTPMiddleware):
    """Attach the originating client IP to request.state."""

    async def dispatch(self, request: Request, call_next) -> Response:
        forwarded_for = request.headers.get("x-forwarded-for")

        if forwarded_for:
            # Take the first IP if multiple proxies are present.
            client_ip = forwarded_for.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = None

        request.state.client_ip = client_ip

        response = await call_next(request)
        return response


class SecurityHeadersMiddleware:
    """Middleware that adds security HTTP headers to all responses.

    Adds headers to defend against common web vulnerabilities:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking via iframes
    - X-XSS-Protection: Enables browser XSS filters
    - Referrer-Policy: Controls referrer information leakage
    - Content-Security-Policy: Restricts resource loading (Issue #2014)

    The Content-Security-Policy header can be configured via the CSP_POLICY
    environment variable. If not set, a restrictive default policy is used
    that only allows resources from the same origin.
    """

    def __init__(self, app):
        self.app = app
        # Read CSP policy from environment, with secure default
        # Default policy: only allow resources from same origin
        # 'unsafe-inline' is needed for Swagger UI inline scripts
        self.csp_policy = os.getenv(
            "CSP_POLICY",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'",
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))

                # Add security headers
                headers[b"x-content-type-options"] = b"nosniff"
                headers[b"x-frame-options"] = b"DENY"
                headers[b"x-xss-protection"] = b"1; mode=block"
                headers[b"referrer-policy"] = b"strict-origin-when-cross-origin"

                # Add Content-Security-Policy header (Issue #2014)
                # This restricts where resources can be loaded from, preventing XSS
                headers[b"content-security-policy"] = self.csp_policy.encode("utf-8")

                # HSTS support
                enable_hsts = os.getenv("ENABLE_HSTS", "").strip().lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )
                if enable_hsts:
                    headers[
                        b"strict-transport-security"
                    ] = b"max-age=31536000; includeSubDomains"

                # Convert back to list of tuples
                message["headers"] = [(k, v) for k, v in headers.items()]

            await send(message)

        await self.app(scope, receive, send_with_security_headers)


class ContentLengthLimitMiddleware(BaseHTTPMiddleware):
    """Reject declared HTTP request bodies exceeding MAX_REQUEST_BYTES (default 50MB)."""

    async def dispatch(self, request, call_next):
        max_bytes_str = os.environ.get(
            "MAX_REQUEST_BYTES",
            str(DEFAULT_MAX_REQUEST_BYTES),
        )
        try:
            max_bytes = int(max_bytes_str)
        except ValueError:
            max_bytes = DEFAULT_MAX_REQUEST_BYTES

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_bytes:
                    return Response(
                        "Payload Too Large",
                        status_code=413,
                    )
            except ValueError:
                pass

        return await call_next(request)


def _normalized_media_type(content_type: str | None) -> str:
    """Return the lowercase media type without parameters."""
    if content_type is None:
        return ""
    return content_type.split(";", 1)[0].strip().casefold()


def _is_json_media_type(content_type: str | None) -> bool:
    """Return whether a Content-Type represents JSON.

    Besides ``application/json``, RFC-compatible structured syntax
    suffixes such as ``application/problem+json`` are accepted.
    """
    media_type = _normalized_media_type(content_type)
    if media_type == "application/json":
        return True
    return (
        media_type.startswith("application/")
        and media_type.endswith("+json")
        and len(media_type) > len("application/+json")
    )


class JSONContentTypeMiddleware(BaseHTTPMiddleware):
    """Require JSON Content-Type for API POST and PUT payloads.

    Only API paths are inspected. Known non-JSON endpoints such as the
    multipart scan route are excluded. Requests without a declared or
    streamed body are allowed because there is no JSON payload to
    inspect.
    """

    def __init__(
        self,
        app,
        *,
        api_prefix: str = JSON_API_PREFIX,
        excluded_paths: Iterable[str] = NON_JSON_API_PATHS,
    ) -> None:
        super().__init__(app)
        self.api_prefix = api_prefix
        self.excluded_paths = frozenset(excluded_paths)

    @staticmethod
    def _has_request_payload(request: Request) -> bool:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                return int(content_length) > 0
            except ValueError:
                # A malformed length should not be treated as proof of
                # a body. Other middleware may reject it separately.
                return False

        # Chunked/streamed requests may legitimately omit a length.
        transfer_encoding = request.headers.get(
            "transfer-encoding",
            "",
        )
        return "chunked" in transfer_encoding.casefold()

    def _requires_json(self, request: Request) -> bool:
        if request.method.upper() not in {"POST", "PUT"}:
            return False

        path = request.url.path
        if not path.startswith(self.api_prefix):
            return False
        if path in self.excluded_paths:
            return False

        return self._has_request_payload(request)

    async def dispatch(self, request, call_next):
        if self._requires_json(request) and not _is_json_media_type(
            request.headers.get("content-type")
        ):
            return JSONResponse(
                status_code=415,
                content={
                    "detail": (
                        "Unsupported Media Type: Request must be " "application/json"
                    )
                },
            )

        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique ``X-Request-ID`` to every request/response cycle.

    If the incoming request already carries an ``X-Request-ID`` header
    its value is reused (after a length sanity check) so upstream
    services can propagate a correlation id they generated. Otherwise a
    fresh RFC 4122 v4 UUID is produced.

    The resolved identifier is:

    * exposed to downstream handlers via ``request.state.request_id``
      so application code and loggers can include it in structured logs;
    * attached to the outgoing response under the ``X-Request-ID``
      header so clients can quote it when reporting issues.
    """

    HEADER_NAME = "X-Request-ID"
    # Guards against malicious oversized incoming headers; a UUID4 hex
    # string is 32 chars, but callers may pass longer trace IDs.
    MAX_INCOMING_LENGTH = 128

    @staticmethod
    def _is_valid_incoming(value: str) -> bool:
        return bool(value) and len(value) <= RequestIDMiddleware.MAX_INCOMING_LENGTH

    async def dispatch(self, request, call_next):
        incoming = request.headers.get(self.HEADER_NAME, "").strip()
        if self._is_valid_incoming(incoming):
            request_id = incoming
        else:
            request_id = uuid.uuid4().hex

        # Make the id available to downstream handlers / loggers.
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[self.HEADER_NAME] = request_id
        return response


class TokenBucketRateLimiter(BaseHTTPMiddleware):
    """In-memory Token Bucket rate limiter per IP address for REST API routes.

    Refills tokens at a rate of ``rate_limit_per_minute`` tokens per minute up to
    ``burst_capacity`` tokens. Requests exceeding available capacity return HTTP 429
    Too Many Requests with a ``Retry-After`` header.
    """

    def __init__(
        self,
        app,
        *,
        rate_limit_per_minute: int = 60,
        burst_capacity: int = 10,
        api_prefix: str = JSON_API_PREFIX,
    ) -> None:
        super().__init__(app)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.burst_capacity = burst_capacity
        self.api_prefix = api_prefix
        self._buckets: dict[str, tuple[float, float]] = {}

    def _get_client_ip(self, request: Request) -> str:
        client_ip = getattr(request.state, "client_ip", None)
        if client_ip:
            return client_ip
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    async def dispatch(self, request, call_next):
        if self.api_prefix and not request.url.path.startswith(self.api_prefix):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.monotonic()
        refill_rate = self.rate_limit_per_minute / 60.0

        if client_ip in self._buckets:
            tokens, last_update = self._buckets[client_ip]
            elapsed = now - last_update
            tokens = min(float(self.burst_capacity), tokens + elapsed * refill_rate)
        else:
            tokens = float(self.burst_capacity)

        if tokens >= 1.0:
            tokens -= 1.0
            self._buckets[client_ip] = (tokens, now)
            return await call_next(request)

        needed = 1.0 - tokens
        retry_after = math.ceil(needed / refill_rate) if refill_rate > 0 else 60
        retry_after = max(1, int(retry_after))

        return Response(
            "Too Many Requests",
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )


@asynccontextmanager
async def _lifespan(app):
    """Start/stop the scheduled plagiarism-rescan background job and OpenTelemetry provider.

    Wraps ``src.core.scheduler.RescanScheduler``, which re-checks recently
    uploaded documents against the full corpus on a configurable interval
    so cross-submission plagiarism that only becomes apparent once a later
    document is uploaded still gets caught (and reviewers get notified via
    the existing webhook layer).
    """
    init_tracer_provider()
    validate_bearer_tokens_config()
    await start_scheduler()
    try:
        yield
    finally:
        await stop_scheduler()
        if _tracer_provider and hasattr(_tracer_provider, "shutdown"):
            _tracer_provider.shutdown()


app = st.App(
    "app/streamlit_app.py",
    middleware=[
        Middleware(ClientIPLoggingMiddleware),
        Middleware(RequestIDMiddleware),
        Middleware(SecurityHeadersMiddleware),
        Middleware(ContentLengthLimitMiddleware),
        Middleware(JSONContentTypeMiddleware),
        Middleware(TokenBucketRateLimiter),
    ],
    lifespan=_lifespan,
)
