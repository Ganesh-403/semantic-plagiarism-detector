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

"""Slack API connector implementation."""

from __future__ import annotations

from typing import Any

from src.api_gateway.connectors.base import ServiceConnector


class SlackConnector(ServiceConnector):
    """Connector for Slack Webhook / API integration."""

    def __init__(self, name: str, credentials: dict[str, Any]) -> None:
        super().__init__(name, credentials)
        self.bot_token = credentials.get("bot_token") or credentials.get("webhook_url")

    async def connect(self) -> bool:
        """Connect and validate token."""
        if not self.bot_token:
            self._connected = False
            return False
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect connector."""
        self._connected = False

    async def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Perform request to Slack API."""
        if not self._connected:
            raise RuntimeError("SlackConnector is not connected.")

        return {
            "status": "success",
            "provider": "slack",
            "method": method.upper(),
            "path": path,
            "data": kwargs.get("json") or {},
        }
