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

"""Unit tests for Kubernetes /health/live and /health/ready probe endpoints (Issue #2923)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_live_endpoint_returns_200_ok():
    """Verify /health/live returns 200 OK immediately with status 'alive'."""
    for path in ["/health/live", "/api/v1/health/live"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "service" in data
        assert "version" in data


def test_health_ready_endpoint_success_when_db_and_redis_ready():
    """Verify /health/ready returns 200 OK when both DB and Redis checks pass."""
    with patch("src.api.routers.admin._connect") as mock_connect, patch(
        "src.utils.redis_cache.get_cache"
    ) as mock_get_cache:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        mock_cache = MagicMock()
        mock_cache.ping.return_value = (True, 0.5)
        mock_cache.is_available.return_value = True
        mock_get_cache.return_value = mock_cache

        for path in ["/health/ready", "/api/v1/health/ready"]:
            response = client.get(path)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["db"] == "connected"
            assert data["redis"] == "connected"
            assert "timestamp" in data


def test_health_ready_endpoint_503_when_db_fails():
    """Verify /health/ready returns 503 Service Unavailable when DB check fails."""
    with patch(
        "src.api.routers.admin._connect", side_effect=Exception("DB Connection Error")
    ), patch("src.utils.redis_cache.get_cache") as mock_get_cache:
        mock_cache = MagicMock()
        mock_cache.ping.return_value = (True, 0.5)
        mock_cache.is_available.return_value = True
        mock_get_cache.return_value = mock_cache

        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["db"] == "disconnected"
        assert data["redis"] == "connected"


def test_health_ready_endpoint_503_when_redis_fails():
    """Verify /health/ready returns 503 Service Unavailable when Redis check fails."""
    with patch("src.api.routers.admin._connect") as mock_connect, patch(
        "src.utils.redis_cache.get_cache"
    ) as mock_get_cache:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        mock_cache = MagicMock()
        mock_cache.ping.return_value = (False, None)
        mock_cache.is_available.return_value = False
        mock_get_cache.return_value = mock_cache

        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["db"] == "connected"
        assert data["redis"] == "disconnected"
