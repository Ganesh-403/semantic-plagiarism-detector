"""
test_ssrf_protector_port_allowlist_issue_3190.py
------------------------------------------------
Unit test suite for Issue #3190:
Validates that SSRFProtector restricts webhook destination ports to 443 by default
(and configurable allowed ports via ALLOWED_WEBHOOK_PORTS), raising SSRFSecurityException for unauthorized ports.
"""

import os
from unittest.mock import patch
import pytest

from src.security.ssrf_protector import SSRFProtector, SSRFSecurityException


def test_default_port_allowlist_permits_443_and_blocks_custom_ports(monkeypatch):
    """Verify default port allowlist permits 443 and blocks unauthorized ports."""
    monkeypatch.delenv("ALLOWED_WEBHOOK_PORTS", raising=False)

    # Standard https URL without explicit port (defaults to 443) or explicit :443
    ports = SSRFProtector.get_allowed_webhook_ports()
    assert ports == {443}

    with pytest.raises(SSRFSecurityException) as exc_info:
        SSRFProtector.validate_webhook_url("https://example.com:6379/webhook")
    assert "Unauthorized port 6379" in str(exc_info.value)


def test_configurable_allowed_webhook_ports_env_var(monkeypatch):
    """Verify ALLOWED_WEBHOOK_PORTS env var allows additional configured ports."""
    monkeypatch.setenv("ALLOWED_WEBHOOK_PORTS", "443,8443")

    ports = SSRFProtector.get_allowed_webhook_ports()
    assert ports == {443, 8443}

    # Port 8443 should pass port check
    with patch.object(SSRFProtector, "_resolve_hostname", return_value="93.184.216.34"):
        with patch.object(SSRFProtector, "_make_validation_request") as mock_req:
            mock_req.return_value = True
            assert SSRFProtector.validate_webhook_url("https://example.com:8443/webhook") is True

    # Unauthorized port 8080 should still fail
    with pytest.raises(SSRFSecurityException) as exc_info:
        SSRFProtector.validate_webhook_url("https://example.com:8080/webhook")
    assert "Unauthorized port 8080" in str(exc_info.value)
