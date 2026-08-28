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

"""External Integration Management Service."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Type

from src.api_gateway.connectors.base import ServiceConnector
from src.api_gateway.connectors.github import GitHubConnector
from src.api_gateway.connectors.slack import SlackConnector
from src.api_gateway.models import IntegrationRecord


class IntegrationService:
    """Service managing third-party integrations and service connectors."""

    def __init__(self) -> None:
        self._integrations: dict[str, IntegrationRecord] = {}
        self._connectors: dict[str, ServiceConnector] = {}
        self._connector_registry: dict[str, type[ServiceConnector]] = {
            "github": GitHubConnector,
            "slack": SlackConnector,
        }

    def register_connector_class(
        self, provider: str, connector_cls: type[ServiceConnector]
    ) -> None:
        """Register custom ServiceConnector class for provider."""
        self._connector_registry[provider.lower()] = connector_cls

    def create_integration(
        self,
        name: str,
        provider: str,
        credentials: dict[str, Any],
        enabled: bool = True,
    ) -> IntegrationRecord:
        """Register a new third-party service integration."""
        integration_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        record = IntegrationRecord(
            id=integration_id,
            name=name,
            provider=provider.lower(),
            credentials=credentials,
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )

        self._integrations[integration_id] = record

        # Instantiate connector if provider is registered
        provider_key = provider.lower()
        if provider_key in self._connector_registry:
            connector_cls = self._connector_registry[provider_key]
            connector = connector_cls(name, credentials)
            self._connectors[integration_id] = connector

        return record

    def get_integration(self, integration_id: str) -> IntegrationRecord | None:
        """Retrieve integration record by ID."""
        return self._integrations.get(integration_id)

    def get_connector(self, integration_id: str) -> ServiceConnector | None:
        """Retrieve active ServiceConnector instance for integration."""
        record = self._integrations.get(integration_id)
        if record is None or not record.enabled:
            return None
        return self._connectors.get(integration_id)

    def list_integrations(self) -> list[IntegrationRecord]:
        """List all integration records."""
        return list(self._integrations.values())

    def update_integration(
        self,
        integration_id: str,
        name: str | None = None,
        enabled: bool | None = None,
        credentials: dict[str, Any] | None = None,
    ) -> IntegrationRecord | None:
        """Update an existing integration."""
        record = self._integrations.get(integration_id)
        if record is None:
            return None

        if name is not None:
            record.name = name
        if enabled is not None:
            record.enabled = enabled
        if credentials is not None:
            record.credentials = credentials
            # Re-instantiate connector with updated credentials
            if record.provider in self._connector_registry:
                connector_cls = self._connector_registry[record.provider]
                self._connectors[integration_id] = connector_cls(
                    record.name, credentials
                )

        record.updated_at = datetime.now(timezone.utc)
        return record

    def delete_integration(self, integration_id: str) -> bool:
        """Delete an integration by ID."""
        if integration_id in self._integrations:
            del self._integrations[integration_id]
            self._connectors.pop(integration_id, None)
            return True
        return False

    async def execute_request(
        self, integration_id: str, method: str, path: str, **kwargs: Any
    ) -> Any:
        """Execute request using integration's connector."""
        record = self._integrations.get(integration_id)
        if record is None:
            raise ValueError(f"Integration '{integration_id}' not found.")
        if not record.enabled:
            raise RuntimeError(f"Integration '{record.name}' is disabled.")

        connector = self._connectors.get(integration_id)
        if connector is None:
            raise RuntimeError(
                f"No service connector available for provider '{record.provider}'."
            )

        if not connector.is_connected:
            connected = await connector.connect()
            if not connected:
                raise RuntimeError(f"Failed to connect to integration '{record.name}'.")

        return await connector.request(method, path, **kwargs)
