import socket
from unittest.mock import patch

import pytest

from src.security.ssrf_protector import SSRFProtector, SSRFSecurityException


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure the DNS cache is cleared before every test."""
    SSRFProtector._dns_cache.clear()

def test_validate_webhook_url_empty():
    with pytest.raises(SSRFSecurityException, match="cannot be empty"):
        SSRFProtector.validate_webhook_url("")

def test_validate_webhook_url_insecure_scheme():
    with pytest.raises(SSRFSecurityException, match="must use 'https'"):
        SSRFProtector.validate_webhook_url("http://example.com/webhook")

def test_validate_webhook_url_missing_hostname():
    with pytest.raises(SSRFSecurityException, match="missing hostname"):
        SSRFProtector.validate_webhook_url("https://")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_dns_failure(mock_getaddrinfo):
    mock_getaddrinfo.side_effect = socket.gaierror("Name or service not known")
    with pytest.raises(SSRFSecurityException, match="DNS resolution failed"):
        SSRFProtector.validate_webhook_url("https://nonexistent.domain.local/api")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_loopback(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, '', ('127.0.0.1', 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP: 127.0.0.1"):
        SSRFProtector.validate_webhook_url("https://localhost:8080/hook")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_private_ip(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, '', ('10.0.0.5', 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked private network IP: 10.0.0.5"):
        SSRFProtector.validate_webhook_url("https://internal.corp.network/webhook")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_link_local(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, '', ('169.254.169.254', 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked link-local IP: 169.254.169.254"):
        SSRFProtector.validate_webhook_url("https://aws-metadata-service.local/data")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_multicast(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, '', ('224.0.0.1', 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked multicast IP: 224.0.0.1"):
        SSRFProtector.validate_webhook_url("https://multicast.local/data")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_loopback(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, '', ('::1', 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP: ::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-localhost.local/hook")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_private(mock_getaddrinfo):
    # Unique Local Address (ULA) fd00::/8
    mock_getaddrinfo.return_value = [(10, 1, 6, '', ('fd00::1', 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked private network IP: fd00::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-internal.local/hook")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_safe(mock_getaddrinfo):
    # Mocking a valid public IP (e.g., Discord or Slack webhook IP)
    mock_getaddrinfo.return_value = [(2, 1, 6, '', ('142.250.190.46', 443))]
    
    # Should not raise an exception
    result = SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    assert result is True

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_dns_caching_behavior(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, '', ('142.250.190.46', 443))]
    
    # First call triggers DNS resolution
    SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    assert mock_getaddrinfo.call_count == 1
    
    # Second call should use cache
    SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    assert mock_getaddrinfo.call_count == 1 # Unchanged!
    
@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_empty_dns_resolution(mock_getaddrinfo):
    # Simulate a DNS resolution that returns an empty list
    mock_getaddrinfo.return_value = []
    with pytest.raises(SSRFSecurityException, match="No addresses found for hostname"):
        SSRFProtector.validate_webhook_url("https://empty.domain.local/api")

