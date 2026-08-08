import socket
from unittest.mock import patch

import pytest

from src.security.ssrf_protector import (
    RESTRICTED_IPV4_CIDR_BLOCKS,
    SSRFProtector,
    SSRFSecurityException,
    is_ip_in_cidr_block,
)



@pytest.fixture(autouse=True)
def mock_requests_head():
    with patch("src.security.ssrf_protector.requests.head") as mock_head:
        mock_head.return_value.status_code = 200
        yield mock_head

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
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP: 127.0.0.1"):
        SSRFProtector.validate_webhook_url("https://localhost:8080/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
@pytest.mark.parametrize(
    "blocked_ip",
    [
        "10.0.0.1",
        "10.0.0.5",
        "172.16.5.10",
        "192.168.1.1",
        "192.168.1.25",
    ],
)
def test_validate_webhook_url_private_ipv4_subnet_blocklist(
    mock_getaddrinfo, blocked_ip
):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", (blocked_ip, 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked private network IP"):
        SSRFProtector.validate_webhook_url("https://internal.corp.network/webhook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_link_local(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("169.254.169.254", 443))]
    with pytest.raises(
        SSRFSecurityException, match="Blocked link-local IP: 169.254.169.254"
    ):
        SSRFProtector.validate_webhook_url("https://aws-metadata-service.local/data")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_multicast(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("224.0.0.1", 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked multicast IP: 224.0.0.1"):
        SSRFProtector.validate_webhook_url("https://multicast.local/data")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_loopback(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP: ::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-localhost.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_private(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("fd00::1", 443, 0, 0))]
    with pytest.raises(
        SSRFSecurityException, match="Blocked private network IP: fd00::1"
    ):
        SSRFProtector.validate_webhook_url("https://ipv6-internal.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_link_local(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("fe80::1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked link-local IP: fe80::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-link-local.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_multicast(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("ff00::1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked multicast IP: ff00::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-multicast.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_unspecified(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked unspecified IP"):
        SSRFProtector.validate_webhook_url("https://unspecified.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_mapped_ipv4_loopback(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::ffff:127.0.0.1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP"):
        SSRFProtector.validate_webhook_url("https://mapped-ipv6-loopback.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_mapped_ipv4_private(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::ffff:10.0.0.1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked private network IP"):
        SSRFProtector.validate_webhook_url("https://mapped-ipv6-private.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_invalid_ip_format(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("not_an_ip", 443))]
    with pytest.raises(SSRFSecurityException, match="invalid IP address format"):
        SSRFProtector.validate_webhook_url("https://invalid-ip.local/hook")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_safe(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]
    result = SSRFProtector.validate_webhook_url(
        "https://discord.com/api/webhooks/123/abc"
    )
    assert result is True


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_dns_caching_behavior(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]
    SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    assert mock_getaddrinfo.call_count == 1
    SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    assert mock_getaddrinfo.call_count == 1


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_dns_cache_expired(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]
    SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    assert mock_getaddrinfo.call_count == 1
    hostname = "discord.com"
    cached_ip, _ = SSRFProtector._dns_cache[hostname]
    SSRFProtector._dns_cache[hostname] = (cached_ip, 0)
    SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
    assert mock_getaddrinfo.call_count == 2


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_empty_dns_resolution(mock_getaddrinfo):
    mock_getaddrinfo.return_value = []
    with pytest.raises(SSRFSecurityException, match="No addresses found for hostname"):
        SSRFProtector.validate_webhook_url("https://empty.domain.local/api")


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_resolve_hostname_success(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("8.8.8.8", 53))]
    ip = SSRFProtector._resolve_hostname("dns.google")
    assert ip == "8.8.8.8"


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_allowed_webhook_domains(mock_getaddrinfo, monkeypatch):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]
    monkeypatch.setenv("ALLOWED_WEBHOOK_DOMAINS", "hooks.slack.com, discord.com")

    # Allowed domain exact match passes
    assert SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc") is True

    # Allowed domain subdomain match passes
    assert SSRFProtector.validate_webhook_url("https://hooks.slack.com/services/123") is True
    assert SSRFProtector.validate_webhook_url("https://sub.hooks.slack.com/services/123") is True

    # Disallowed domain raises SSRFSecurityException without calling DNS resolution
    mock_getaddrinfo.reset_mock()
    with pytest.raises(SSRFSecurityException, match="is not in ALLOWED_WEBHOOK_DOMAINS"):
        SSRFProtector.validate_webhook_url("https://unallowed-domain.org/webhook")

    mock_getaddrinfo.assert_not_called()




@pytest.mark.parametrize(
    ("ip_str", "cidr_block"),
    [
        ("127.0.0.0", "127.0.0.0/8"),
        ("127.255.255.255", "127.0.0.0/8"),
        ("10.0.0.0", "10.0.0.0/8"),
        ("10.255.255.255", "10.0.0.0/8"),
        ("172.16.0.0", "172.16.0.0/12"),
        ("172.31.255.255", "172.16.0.0/12"),
        ("192.168.0.0", "192.168.0.0/16"),
        ("192.168.255.255", "192.168.0.0/16"),
    ],
)
def test_is_ip_in_cidr_block_accepts_required_range_boundaries(
    ip_str,
    cidr_block,
):
    assert is_ip_in_cidr_block(ip_str, cidr_block) is True


@pytest.mark.parametrize(
    ("ip_str", "cidr_block"),
    [
        ("126.255.255.255", "127.0.0.0/8"),
        ("128.0.0.0", "127.0.0.0/8"),
        ("9.255.255.255", "10.0.0.0/8"),
        ("11.0.0.0", "10.0.0.0/8"),
        ("172.15.255.255", "172.16.0.0/12"),
        ("172.32.0.0", "172.16.0.0/12"),
        ("192.167.255.255", "192.168.0.0/16"),
        ("192.169.0.0", "192.168.0.0/16"),
        ("8.8.8.8", "10.0.0.0/8"),
    ],
)
def test_is_ip_in_cidr_block_rejects_addresses_outside_range(
    ip_str,
    cidr_block,
):
    assert is_ip_in_cidr_block(ip_str, cidr_block) is False


@pytest.mark.parametrize(
    ("ip_str", "cidr_block"),
    [
        ("not-an-ip", "10.0.0.0/8"),
        ("10.0.0.1", "not-a-cidr"),
        ("10.0.0.1", "2001:db8::/32"),
        ("2001:db8::1", "10.0.0.0/8"),
        ("", "10.0.0.0/8"),
    ],
)
def test_is_ip_in_cidr_block_handles_invalid_or_mismatched_input(
    ip_str,
    cidr_block,
):
    assert is_ip_in_cidr_block(ip_str, cidr_block) is False


def test_required_restricted_cidr_policy_is_complete():
    assert RESTRICTED_IPV4_CIDR_BLOCKS == (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
    )


@patch("src.security.ssrf_protector.socket.getaddrinfo")
@pytest.mark.parametrize(
    "blocked_ip",
    [
        "127.44.55.66",
        "10.200.1.2",
        "172.31.255.254",
        "192.168.240.10",
    ],
)
def test_validate_url_safety_integrates_required_cidr_filter(
    mock_getaddrinfo,
    blocked_ip,
):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (blocked_ip, 443))
    ]

    with pytest.raises(SSRFSecurityException):
        SSRFProtector.validate_url_safety(
            "https://blocked.example/webhook"
        )


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_url_safety_allows_public_address(
    mock_getaddrinfo,
):
    mock_getaddrinfo.return_value = [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            ("93.184.216.34", 443),
        )
    ]

    validated_url, pinned_ip = SSRFProtector.validate_url_safety(
        "https://example.com/webhook"
    )

    assert validated_url == "https://example.com/webhook"
    assert pinned_ip == "93.184.216.34"
