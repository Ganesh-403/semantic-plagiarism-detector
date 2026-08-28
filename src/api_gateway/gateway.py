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

"""Core API Gateway Implementation."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from src.api_gateway.api_keys import APIKeyService
from src.api_gateway.integrations import IntegrationService
from src.api_gateway.models import APIKeyRecord, ExposedEndpointRecord
from src.api_gateway.rate_limiter import RateLimiter
from src.api_gateway.webhooks import WebhookService


class RateLimitExceededException(Exception):
    """Exception raised when API key or IP exceeds rate limit."""

    pass


class InvalidAPIKeyException(Exception):
    """Exception raised when API key is missing, invalid, or revoked."""

    pass


class RouteNotFoundException(Exception):
    """Exception raised when requested gateway route is not registered."""

    pass


class APIGateway:
    """Unified API Gateway coordinating Authentication, Rate Limiting, Routing, Webhooks, and Integrations."""

    def __init__(
        self,
        api_key_service: APIKeyService | None = None,
        rate_limiter: RateLimiter | None = None,
        webhook_service: WebhookService | None = None,
        integration_service: IntegrationService | None = None,
    ) -> None:
        self.api_keys = api_key_service or APIKeyService()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.webhooks = webhook_service or WebhookService()
        self.integrations = integration_service or IntegrationService()

        # Registered routes: (method.upper(), path) -> ExposedEndpointRecord
        self._routes: dict[tuple[str, str], ExposedEndpointRecord] = {}

    def register(
        self,
        method: str,
        path: str,
        handler: Callable[..., Any],
        name: str | None = None,
        require_api_key: bool = True,
    ) -> ExposedEndpointRecord:
        """Register an internal function or handler as a Gateway route."""
        formatted_method = method.upper()
        formatted_path = "/" + path.lstrip("/")
        route_key = (formatted_method, formatted_path)

        record = ExposedEndpointRecord(
            method=formatted_method,
            path=formatted_path,
            handler=handler,
            name=name or handler.__name__,
            require_api_key=require_api_key,
        )
        self._routes[route_key] = record
        return record

    def get_route(self, method: str, path: str) -> ExposedEndpointRecord | None:
        """Find registered endpoint record for method and path."""
        formatted_method = method.upper()
        formatted_path = "/" + path.lstrip("/")
        return self._routes.get((formatted_method, formatted_path))

    def list_routes(self) -> list[ExposedEndpointRecord]:
        """List all registered exposed routes."""
        return list(self._routes.values())

    async def dispatch(
        self,
        method: str,
        path: str,
        api_key: str | None = None,
        client_ip: str = "127.0.0.1",
        **kwargs: Any,
    ) -> Any:
        """Authenticate, rate limit, and dispatch a request to its registered handler."""
        endpoint = self.get_route(method, path)
        if endpoint is None:
            raise RouteNotFoundException(
                f"No route registered for {method.upper()} {path}"
            )

        key_record: APIKeyRecord | None = None

        # 1. API Key Validation
        if endpoint.require_api_key:
            if not api_key:
                raise InvalidAPIKeyException("API Key is required for this endpoint.")
            key_record = self.api_keys.validate_key(api_key)
            if key_record is None:
                raise InvalidAPIKeyException("Invalid, expired, or revoked API Key.")

        # 2. Rate Limiting
        limiter_key = key_record.id if key_record else client_ip
        limit = key_record.rate_limit if key_record else self.rate_limiter.default_limit

        if not self.rate_limiter.allow(limiter_key, limit=limit):
            raise RateLimitExceededException(
                f"Rate limit exceeded ({limit} req/min). Try again later."
            )

        # 3. Route Execution
        handler = endpoint.handler
        if inspect.iscoroutinefunction(handler):
            return await handler(**kwargs)
        return handler(**kwargs)
