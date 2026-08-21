"""
tests/security/test_ssrf_protector.py
-------------------------------------
Unit tests for the SSRF protection module.

Validates URL validation, DNS resolution, IP blocking, redirect handling,
and User-Agent header attachment (Issue #2212).
"""

import socket
from unittest.mock import MagicMock, patch

import pytest

from src.security.ssrf_protector import (
    DEFAULT_USER_AGENT,
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
def test_dns_cache_is_bounded_and_evicts_oldest_entries(mock_getaddrinfo):
    """Regression test for the unbounded-cache memory-leak issue: a
    malicious actor hitting webhook validation with thousands of unique
    randomly generated subdomains must not grow _dns_cache without limit.
    Once DNS_CACHE_MAX_SIZE is exceeded, the least-recently-used entries
    must be evicted."""
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]

    flood_size = SSRFProtector.DNS_CACHE_MAX_SIZE + 500
    for i in range(flood_size):
        SSRFProtector._resolve_hostname(f"malicious-{i}.example.com")

    assert len(SSRFProtector._dns_cache) == SSRFProtector.DNS_CACHE_MAX_SIZE

    # The earliest-inserted (now least-recently-used) hostnames must have
    # been evicted, not merely left to grow the dict indefinitely.
    assert "malicious-0.example.com" not in SSRFProtector._dns_cache
    assert "malicious-1.example.com" not in SSRFProtector._dns_cache

    # The most recently inserted hostname must still be cached.
    assert f"malicious-{flood_size - 1}.example.com" in SSRFProtector._dns_cache


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_dns_cache_eviction_is_lru_not_fifo(mock_getaddrinfo):
    """Re-accessing an older cache entry must protect it from eviction --
    confirming genuine least-recently-used semantics rather than a simple
    insertion-order cap that would evict a hostname still being actively
    (re-)validated."""
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("1.2.3.4", 443))]

    original_max_size = SSRFProtector.DNS_CACHE_MAX_SIZE
    SSRFProtector.DNS_CACHE_MAX_SIZE = 3
    try:
        SSRFProtector._resolve_hostname("a.example.com")
        SSRFProtector._resolve_hostname("b.example.com")
        SSRFProtector._resolve_hostname("c.example.com")

        # Re-access "a" so it becomes the most-recently-used entry.
        SSRFProtector._resolve_hostname("a.example.com")

        # A 4th unique hostname should now evict "b" (the true LRU entry),
        # not "a" (which was just re-accessed).
        SSRFProtector._resolve_hostname("d.example.com")

        assert "a.example.com" in SSRFProtector._dns_cache
        assert "b.example.com" not in SSRFProtector._dns_cache
        assert "c.example.com" in SSRFProtector._dns_cache
        assert "d.example.com" in SSRFProtector._dns_cache
    finally:
        SSRFProtector.DNS_CACHE_MAX_SIZE = original_max_size


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
    assert (
        SSRFProtector.validate_webhook_url("https://discord.com/api/webhooks/123/abc")
        is True
    )

    # Allowed domain subdomain match passes
    assert (
        SSRFProtector.validate_webhook_url("https://hooks.slack.com/services/123")
        is True
    )
    assert (
        SSRFProtector.validate_webhook_url("https://sub.hooks.slack.com/services/123")
        is True
    )

    # Disallowed domain raises SSRFSecurityException without calling DNS resolution
    mock_getaddrinfo.reset_mock()
    with pytest.raises(
        SSRFSecurityException, match="is not in ALLOWED_WEBHOOK_DOMAINS"
    ):
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
        SSRFProtector.validate_url_safety("https://blocked.example/webhook")


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


def test_default_user_agent_constant_defined():
    assert DEFAULT_USER_AGENT == "SemanticPlagiarismDetector/1.0"
    assert SSRFProtector.DEFAULT_USER_AGENT == "SemanticPlagiarismDetector/1.0"


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_url_safety_attaches_default_user_agent_header(
    mock_getaddrinfo,
    mock_requests_head,
):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    SSRFProtector.validate_url_safety("https://example.com/webhook")

    mock_requests_head.assert_called_once_with(
        "https://example.com/webhook",
        headers={"User-Agent": "SemanticPlagiarismDetector/1.0"},
        timeout=5.0,
        allow_redirects=False,
    )


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_url_safety_attaches_custom_user_agent_header(
    mock_getaddrinfo,
    mock_requests_head,
):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    custom_agent = "CustomBot/2.0"
    SSRFProtector.validate_url_safety(
        "https://example.com/webhook",
        user_agent=custom_agent,
    )

    mock_requests_head.assert_called_once_with(
        "https://example.com/webhook",
        headers={"User-Agent": custom_agent},
        timeout=5.0,
        allow_redirects=False,
    )


class TestUserAgentHeaderInspection:
    """Test suite for User-Agent header attachment (Issue #2212)."""

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_default_user_agent_attached_to_head_request(
        self, mock_validate, mock_head
    ):
        """Verify DEFAULT_USER_AGENT is attached to outgoing HEAD requests."""
        mock_head.return_value = MagicMock(status_code=200)

        SSRFProtector.validate_webhook_url("https://example.com/webhook")

        mock_head.assert_called_once()
        call_kwargs = mock_head.call_args[1]

        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["User-Agent"] == DEFAULT_USER_AGENT
        assert call_kwargs["headers"]["User-Agent"] == "SemanticPlagiarismDetector/1.0"

    @patch("src.security.ssrf_protector.requests.head")
    @patch("src.security.ssrf_protector.requests.get")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_fallback_to_get_when_head_rejected(
        self, mock_validate, mock_get, mock_head
    ):
        """Verify fallback to GET when server rejects HEAD with 405."""
        # HEAD returns 405 Method Not Allowed
        mock_head.return_value = MagicMock(status_code=405)
        # GET succeeds
        mock_get.return_value = MagicMock(status_code=200)

        SSRFProtector.validate_webhook_url("https://example.com/webhook")

        # Both HEAD and GET should be called
        mock_head.assert_called_once()
        mock_get.assert_called_once()

        # GET should also have User-Agent header
        get_kwargs = mock_get.call_args[1]
        assert get_kwargs["headers"]["User-Agent"] == DEFAULT_USER_AGENT

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_custom_user_agent_override(self, mock_validate, mock_head):
        """Verify custom User-Agent can override the default."""
        mock_head.return_value = MagicMock(status_code=200)
        custom_ua = "CustomBot/2.0"

        SSRFProtector.validate_webhook_url(
            "https://example.com/webhook", user_agent=custom_ua
        )

        call_kwargs = mock_head.call_args[1]
        assert call_kwargs["headers"]["User-Agent"] == custom_ua

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_check_redirect_depth", return_value=None)
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_user_agent_attached_to_redirect_validation(
        self, mock_validate, mock_redirect, mock_head
    ):
        """Verify User-Agent is attached to requests during redirect chain validation."""
        mock_head.return_value = MagicMock(status_code=200)

        SSRFProtector.validate_url_safety("https://example.com/webhook")

        # User-Agent should be in the HEAD request
        call_kwargs = mock_head.call_args[1]
        assert call_kwargs["headers"]["User-Agent"] == DEFAULT_USER_AGENT

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_user_agent_constant_value(self, mock_validate, mock_head):
        """Verify DEFAULT_USER_AGENT constant has the expected value."""
        assert DEFAULT_USER_AGENT == "SemanticPlagiarismDetector/1.0"
        assert SSRFProtector.DEFAULT_USER_AGENT == "SemanticPlagiarismDetector/1.0"


class TestIPCIDRBlocking:
    """Test suite for IP address CIDR block validation."""

    def test_ip_in_cidr_block(self):
        """Verify IP correctly identified as within CIDR block."""
        assert is_ip_in_cidr_block("192.168.1.100", "192.168.0.0/16") is True
        assert is_ip_in_cidr_block("10.0.0.5", "10.0.0.0/8") is True
        assert is_ip_in_cidr_block("127.0.0.1", "127.0.0.0/8") is True

    def test_ip_not_in_cidr_block(self):
        """Verify IP correctly identified as outside CIDR block."""
        assert is_ip_in_cidr_block("8.8.8.8", "192.168.0.0/16") is False
        assert is_ip_in_cidr_block("1.1.1.1", "10.0.0.0/8") is False

    def test_invalid_ip_returns_false(self):
        """Verify invalid IP strings return False instead of raising."""
        assert is_ip_in_cidr_block("not-an-ip", "192.168.0.0/16") is False
        assert is_ip_in_cidr_block("", "10.0.0.0/8") is False

    def test_invalid_cidr_returns_false(self):
        """Verify invalid CIDR strings return False instead of raising."""
        assert is_ip_in_cidr_block("192.168.1.1", "not-a-cidr") is False
        assert is_ip_in_cidr_block("10.0.0.1", "") is False

    def test_ipv6_support(self):
        """Verify IPv6 addresses and CIDR blocks are supported."""
        assert is_ip_in_cidr_block("::1", "::1/128") is True
        assert is_ip_in_cidr_block("2001:db8::1", "2001:db8::/32") is True
        assert is_ip_in_cidr_block("2001:db8::1", "192.168.0.0/16") is False


class TestURLValidation:
    """Test suite for URL target validation."""

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="93.184.216.34")
    def test_valid_https_url_passes(self, mock_resolve):
        """Verify valid HTTPS URL with public IP passes validation."""
        ip = SSRFProtector._validate_url_target("https://example.com/webhook")
        assert ip == "93.184.216.34"

    def test_http_scheme_rejected(self):
        """Verify non-HTTPS schemes are rejected."""
        with pytest.raises(SSRFSecurityException, match="Insecure scheme"):
            SSRFProtector._validate_url_target("http://example.com/webhook")

    def test_empty_url_rejected(self):
        """Verify empty URLs are rejected."""
        with pytest.raises(SSRFSecurityException, match="empty"):
            SSRFProtector._validate_url_target("")

        with pytest.raises(SSRFSecurityException, match="empty"):
            SSRFProtector._validate_url_target(None)

    def test_missing_hostname_rejected(self):
        """Verify URLs without hostname are rejected."""
        with pytest.raises(SSRFSecurityException, match="Missing hostname"):
            SSRFProtector._validate_url_target("https://")

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="127.0.0.1")
    def test_loopback_ip_rejected(self, mock_resolve):
        """Verify loopback IPs (127.x.x.x) are rejected."""
        with pytest.raises(SSRFSecurityException, match="loopback"):
            SSRFProtector._validate_url_target("https://localhost/webhook")

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="192.168.1.100")
    def test_private_ip_rejected(self, mock_resolve):
        """Verify private IPs (192.168.x.x, 10.x.x.x) are rejected."""
        with pytest.raises(SSRFSecurityException, match="private"):
            SSRFProtector._validate_url_target("https://internal.local/webhook")

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="169.254.1.1")
    def test_link_local_ip_rejected(self, mock_resolve):
        """Verify link-local IPs (169.254.x.x) are rejected."""
        with pytest.raises(SSRFSecurityException, match="link-local"):
            SSRFProtector._validate_url_target("https://linklocal/webhook")


class TestRedirectChainValidation:
    """Test suite for redirect chain validation."""

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_no_redirect_returns_original_url(self, mock_validate, mock_head):
        """Verify URL without redirects returns original URL."""
        mock_head.return_value = MagicMock(status_code=200, headers={})

        final_url, ip = SSRFProtector.validate_url_safety("https://example.com/webhook")

        assert final_url == "https://example.com/webhook"
        assert ip == "93.184.216.34"

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_single_redirect_followed(self, mock_validate, mock_head):
        """Verify single redirect is followed and validated."""
        # First request returns redirect
        mock_head.side_effect = [
            MagicMock(
                status_code=301, headers={"Location": "https://example.com/final"}
            ),
            MagicMock(status_code=200, headers={}),
        ]

        final_url, ip = SSRFProtector.validate_url_safety("https://example.com/start")

        assert final_url == "https://example.com/final"

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_max_redirects_exceeded_raises(self, mock_validate, mock_head):
        """Verify exceeding max redirects raises exception."""
        # Always return redirect
        mock_head.return_value = MagicMock(
            status_code=301, headers={"Location": "https://example.com/loop"}
        )

        with pytest.raises(SSRFSecurityException, match="max redirects"):
            SSRFProtector.validate_url_safety(
                "https://example.com/start", max_redirects=2
            )

    @patch("src.security.ssrf_protector.requests.head")
    @patch.object(SSRFProtector, "_validate_url_target", return_value="93.184.216.34")
    def test_circular_redirect_detected(self, mock_validate, mock_head):
        """Verify circular redirect chains are detected."""
        # Create a loop: A -> B -> A
        mock_head.side_effect = [
            MagicMock(status_code=301, headers={"Location": "https://example.com/b"}),
            MagicMock(status_code=301, headers={"Location": "https://example.com/a"}),
        ]

        with pytest.raises(SSRFSecurityException, match="circular"):
            SSRFProtector.validate_url_safety("https://example.com/a")


class TestEmptyAllowedDomainsBehavior:
    """Test suite for empty allowed_domains behavior (Issue #2434).

    Verifies that when allowed_domains is empty or None, the SSRF protector
    permits all external domains while still enforcing private IP restrictions.
    This is critical security behavior that must be explicitly tested.
    """

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="93.184.216.34")
    def test_empty_list_permits_any_external_domain(self, mock_resolve):
        """Verify empty allowed_domains list permits random external domains."""
        # Pass an empty list explicitly
        ip = SSRFProtector._validate_url_target(
            "https://random-external-domain.com/webhook", allowed_domains=[]
        )

        # Should succeed and return the resolved IP
        assert ip == "93.184.216.34"
        mock_resolve.assert_called_once_with("random-external-domain.com")

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="104.16.132.229")
    def test_none_permits_any_external_domain(self, mock_resolve):
        """Verify None allowed_domains permits any external domain."""
        # Pass None explicitly (will trigger get_allowed_webhook_domains fallback)
        with patch(
            "src.security.ssrf_protector.get_allowed_webhook_domains", return_value=[]
        ):
            ip = SSRFProtector._validate_url_target(
                "https://another-random-domain.com/webhook", allowed_domains=None
            )

            assert ip == "104.16.132.229"

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="127.0.0.1")
    def test_empty_list_still_blocks_loopback(self, mock_resolve):
        """Verify empty allowlist still blocks loopback addresses (127.0.0.1)."""
        # Even with empty allowlist, loopback should be blocked
        with pytest.raises(SSRFSecurityException, match="loopback"):
            SSRFProtector._validate_url_target(
                "https://localhost/webhook", allowed_domains=[]
            )

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="192.168.1.100")
    def test_empty_list_still_blocks_private_ips(self, mock_resolve):
        """Verify empty allowlist still blocks private IP ranges (192.168.x.x)."""
        # Even with empty allowlist, private IPs should be blocked
        with pytest.raises(SSRFSecurityException, match="private"):
            SSRFProtector._validate_url_target(
                "https://internal.local/webhook", allowed_domains=[]
            )

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="10.0.0.5")
    def test_empty_list_still_blocks_10_network(self, mock_resolve):
        """Verify empty allowlist still blocks 10.0.0.0/8 private range."""
        with pytest.raises(SSRFSecurityException, match="private"):
            SSRFProtector._validate_url_target(
                "https://internal.corp/webhook", allowed_domains=[]
            )

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="172.16.0.100")
    def test_empty_list_still_blocks_172_16_network(self, mock_resolve):
        """Verify empty allowlist still blocks 172.16.0.0/12 private range."""
        with pytest.raises(SSRFSecurityException, match="private"):
            SSRFProtector._validate_url_target(
                "https://internal.vpc/webhook", allowed_domains=[]
            )

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="169.254.1.1")
    def test_empty_list_still_blocks_link_local(self, mock_resolve):
        """Verify empty allowlist still blocks link-local addresses (169.254.x.x)."""
        with pytest.raises(SSRFSecurityException, match="link-local"):
            SSRFProtector._validate_url_target(
                "https://linklocal/webhook", allowed_domains=[]
            )

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="93.184.216.34")
    def test_nonempty_list_restricts_to_approved_domains(self, mock_resolve):
        """Verify non-empty allowlist restricts to approved domains only."""
        # Should succeed for approved domain
        ip = SSRFProtector._validate_url_target(
            "https://slack.com/webhook", allowed_domains=["slack.com", "discord.com"]
        )
        assert ip == "93.184.216.34"

        # Should fail for non-approved domain
        with pytest.raises(SSRFSecurityException, match="not in the allowed domains"):
            SSRFProtector._validate_url_target(
                "https://malicious.com/webhook",
                allowed_domains=["slack.com", "discord.com"],
            )

    @patch.object(SSRFProtector, "_resolve_hostname", return_value="93.184.216.34")
    def test_subdomain_matching_with_empty_list(self, mock_resolve):
        """Verify subdomains work correctly with empty allowlist."""
        # Any subdomain should be permitted when allowlist is empty
        ip = SSRFProtector._validate_url_target(
            "https://api.random-service.com/webhook", allowed_domains=[]
        )
        assert ip == "93.184.216.34"

    def test_empty_list_logs_debug_message(self, caplog):
        """Verify empty allowlist logs a debug message for audit trail."""
        with patch.object(
            SSRFProtector, "_resolve_hostname", return_value="93.184.216.34"
        ):
            with caplog.at_level("DEBUG"):
                SSRFProtector._validate_url_target(
                    "https://example.com/webhook", allowed_domains=[]
                )

                # Should log that allowlist is empty
                assert any(
                    "Domain allowlist is empty" in record.message
                    for record in caplog.records
                )


def test_is_ip_in_cidr_block_ipv4_mapped_ipv6_loopback():
    """Verify that is_ip_in_cidr_block correctly matches IPv4-mapped IPv6 localhost address to the loopback CIDR block."""
    assert is_ip_in_cidr_block("::ffff:127.0.0.1", "127.0.0.0/8") is True


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_resolve_hostname_caching_behavior(mock_getaddrinfo):
    """Verify that SSRFProtector._resolve_hostname successfully uses cached DNS entries on repeated lookups."""
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))
    ]

    # First lookup
    ip1 = SSRFProtector._resolve_hostname("example.com")
    assert ip1 == "93.184.216.34"
    assert mock_getaddrinfo.call_count == 1

    # Second lookup (should hit cache)
    ip2 = SSRFProtector._resolve_hostname("example.com")
    assert ip2 == "93.184.216.34"
    assert mock_getaddrinfo.call_count == 1


def test_is_ip_in_cidr_block_strips_padded_ip():
    """Whitespace-padded private IPs must still match the CIDR block."""
    assert is_ip_in_cidr_block(" 127.0.0.1 ", "127.0.0.0/8") is True
