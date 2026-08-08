import socket
from unittest.mock import patch

import pytest

from src.security.ssrf_protector import SSRFProtector, SSRFSecurityException



def test_whitelisted_private_subnet(monkeypatch):
    SSRFProtector.configure_allowed_cidrs(["10.0.0.0/8"])

    monkeypatch.setattr(
        SSRFProtector,
        "_resolve_hostname",
        classmethod(lambda cls, hostname: "10.1.2.3"),
    )




def test_private_subnet_blocked(monkeypatch, caplog):
    SSRFProtector.configure_allowed_cidrs(None)

    monkeypatch.setattr(
        SSRFProtector,
        "_resolve_hostname",
        classmethod(lambda cls, hostname: "10.1.2.3"),
    )

    with pytest.raises(SSRFSecurityException):
        SSRFProtector.validate_webhook_url(
            "https://internal.example.com"
        )
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
def test_validate_webhook_url_loopback(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP: 127.0.0.1"):
        SSRFProtector.validate_webhook_url("https://localhost:8080/hook")
    assert "Blocked SSRF attempt to target URL: https://localhost:8080/hook" in caplog.text


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
    mock_getaddrinfo, blocked_ip, caplog
):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", (blocked_ip, 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked private network IP"):
        SSRFProtector.validate_webhook_url("https://internal.corp.network/webhook")
    assert "Blocked SSRF attempt to target URL: https://internal.corp.network/webhook" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_link_local(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("169.254.169.254", 443))]
    with pytest.raises(
        SSRFSecurityException, match="Blocked link-local IP: 169.254.169.254"
    ):
        SSRFProtector.validate_webhook_url("https://aws-metadata-service.local/data")
    assert "Blocked SSRF attempt to target URL: https://aws-metadata-service.local/data" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_multicast(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("224.0.0.1", 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked multicast IP: 224.0.0.1"):
        SSRFProtector.validate_webhook_url("https://multicast.local/data")
    assert "Blocked SSRF attempt to target URL: https://multicast.local/data" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_loopback(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP: ::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-localhost.local/hook")
    assert "Blocked SSRF attempt to target URL: https://ipv6-localhost.local/hook" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_private(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("fd00::1", 443, 0, 0))]
    with pytest.raises(
        SSRFSecurityException, match="Blocked private network IP: fd00::1"
    ):
        SSRFProtector.validate_webhook_url("https://ipv6-internal.local/hook")
    assert "Blocked SSRF attempt to target URL: https://ipv6-internal.local/hook" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_link_local(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("fe80::1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked link-local IP: fe80::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-link-local.local/hook")
    assert "Blocked SSRF attempt to target URL: https://ipv6-link-local.local/hook" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_multicast(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("ff00::1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked multicast IP: ff00::1"):
        SSRFProtector.validate_webhook_url("https://ipv6-multicast.local/hook")
    assert "Blocked SSRF attempt to target URL: https://ipv6-multicast.local/hook" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_unspecified(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked unspecified IP"):
        SSRFProtector.validate_webhook_url("https://unspecified.local/hook")
    assert "Blocked SSRF attempt to target URL: https://unspecified.local/hook" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_mapped_ipv4_loopback(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::ffff:127.0.0.1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP"):
        SSRFProtector.validate_webhook_url("https://mapped-ipv6-loopback.local/hook")
    assert "Blocked SSRF attempt to target URL: https://mapped-ipv6-loopback.local/hook" in caplog.text


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_mapped_ipv4_private(mock_getaddrinfo, caplog):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("::ffff:10.0.0.1", 443, 0, 0))]
    with pytest.raises(SSRFSecurityException, match="Blocked private network IP"):
        SSRFProtector.validate_webhook_url("https://mapped-ipv6-private.local/hook")
    assert "Blocked SSRF attempt to target URL: https://mapped-ipv6-private.local/hook" in caplog.text


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

@patch("src.security.ssrf_protector.requests.head")
def test_validate_webhook_url_max_redirects_exceeded(mock_head, monkeypatch):
    monkeypatch.setattr(
        SSRFProtector,
        "_resolve_hostname",
        classmethod(lambda cls, hostname: "93.184.216.34"),
    )

    def make_redirect_response(hop_number):
        return type(
            "MockResponse",
            (),
            {
                "status_code": 302,
                "headers": {"Location": f"https://example.com/hop{hop_number}"},
            },
        )()

    # Each hop redirects to a brand-new URL (no repeats), so this
    # exercises the "too many redirects" path rather than the loop path.
    mock_head.side_effect = [make_redirect_response(i) for i in range(1, 6)]

    with pytest.raises(SSRFSecurityException, match="Maximum HTTP redirect depth exceeded"):
        SSRFProtector.validate_webhook_url("https://example.com/start")


@patch("src.security.ssrf_protector.requests.head")
def test_validate_webhook_url_circular_redirect_loop_detected(mock_head, monkeypatch):
    """A -> B -> A should be caught as a circular redirect loop (issue #1496)."""
    monkeypatch.setattr(
        SSRFProtector,
        "_resolve_hostname",
        classmethod(lambda cls, hostname: "93.184.216.34"),
    )

    response_a_to_b = type(
        "MockResponse",
        (),
        {"status_code": 302, "headers": {"Location": "https://example.com/b"}},
    )()
    response_b_to_a = type(
        "MockResponse",
        (),
        {"status_code": 302, "headers": {"Location": "https://example.com/a"}},
    )()
    mock_head.side_effect = [response_a_to_b, response_b_to_a]

    with pytest.raises(
        SSRFSecurityException, match="Circular HTTP redirect loop detected"
    ):
        SSRFProtector.validate_webhook_url("https://example.com/a")
@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_hexadecimal_localhost(mock_getaddrinfo, caplog):
    """
    Hexadecimal representation of localhost (0x7f000001) should resolve
    to 127.0.0.1 and be blocked.
    """
    mock_getaddrinfo.return_value = [
        (2, 1, 6, "", ("127.0.0.1", 443))
    ]

    with pytest.raises(
        SSRFSecurityException,
        match="Blocked loopback IP: 127.0.0.1",
    ):
        SSRFProtector.validate_webhook_url(
            "https://0x7f000001/webhook"
        )

    assert (
        "Blocked SSRF attempt to target URL: https://0x7f000001/webhook"
        in caplog.text
    )
@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_webhook_url_ipv6_loopback_literal(mock_getaddrinfo, caplog):
    """
    IPv6 loopback literal (::1) should always be blocked.
    """
    mock_getaddrinfo.return_value = [
        (10, 1, 6, "", ("::1", 443, 0, 0))
    ]

    with pytest.raises(
        SSRFSecurityException,
        match="Blocked loopback IP: ::1",
    ):
        SSRFProtector.validate_webhook_url(
            "https://[::1]/webhook"
        )

    assert (
        "Blocked SSRF attempt to target URL: https://[::1]/webhook"
        in caplog.text
    )
@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_url_safety_returns_pinned_ip(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]
    validated_url, pinned_ip = SSRFProtector.validate_url_safety(
        "https://discord.com/api/webhooks/123/abc"
    )
    assert validated_url == "https://discord.com/api/webhooks/123/abc"
    assert pinned_ip == "142.250.190.46"

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_url_safety_blocks_private_ip(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("10.0.0.1", 443))]
    with pytest.raises(SSRFSecurityException, match="Blocked private network IP"):
        SSRFProtector.validate_url_safety("https://internal.example.com/webhook")

@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_validate_url_safety_uses_dns_cache(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("142.250.190.46", 443))]
    SSRFProtector.validate_url_safety("https://example.com/a")
    SSRFProtector.validate_url_safety("https://example.com/b")
    assert mock_getaddrinfo.call_count == 1

@patch("src.security.ssrf_protector.requests.request")
@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_pinned_request_uses_ip_with_host_header(mock_getaddrinfo, mock_request):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
    mock_request.return_value = type("Response", (), {"status_code": 200})()
    SSRFProtector.pinned_request(
        "GET", "https://example.com/api/resource", "93.184.216.34", timeout=10,
    )
    call_args, call_kwargs = mock_request.call_args
    assert "93.184.216.34" in call_args[1]
    assert call_kwargs["headers"]["Host"] == "example.com"
    assert call_kwargs["timeout"] == 10

@patch("src.security.ssrf_protector.requests.request")
@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_pinned_request_preserves_path_and_query(mock_getaddrinfo, mock_request):
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
    mock_request.return_value = type("Response", (), {"status_code": 200})()
    SSRFProtector.pinned_request(
        "POST", "https://example.com/api/v1/webhook?token=abc123", "93.184.216.34",
        json={"key": "value"},
    )
    call_args, call_kwargs = mock_request.call_args
    requested_url = call_args[1]
    assert "/api/v1/webhook" in requested_url
    assert "token=abc123" in requested_url
    assert call_kwargs["headers"]["Host"] == "example.com"
    assert call_kwargs["json"] == {"key": "value"}

@patch("src.security.ssrf_protector.requests.request")
@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_pinned_request_ipv6_address(mock_getaddrinfo, mock_request):
    mock_getaddrinfo.return_value = [(10, 1, 6, "", ("2606:2800:220:1:248:1893:25c8:1946", 443, 0, 0))]
    mock_request.return_value = type("Response", (), {"status_code": 200})()
    SSRFProtector.pinned_request("GET", "https://example.com/", "2606:2800:220:1:248:1893:25c8:1946")
    call_args, call_kwargs = mock_request.call_args
    assert "[2606:2800:220:1:248:1893:25c8:1946]" in call_args[1]
    assert call_kwargs["headers"]["Host"] == "example.com"


@patch("src.security.ssrf_protector.requests.head")
@patch("src.security.ssrf_protector.SSRFProtector._resolve_hostname")
def test_redirect_chain_validates_ip_before_request(mock_resolve, mock_head):
    """Verify redirect targets to internal/private IPs are blocked BEFORE making requests.head calls."""
    def side_effect(hostname):
        if hostname == "public.com":
            return "93.184.216.34"
        return "127.0.0.1"

    mock_resolve.side_effect = side_effect
    mock_head.return_value = type("Response", (), {"status_code": 302, "headers": {"Location": "https://internal.local/"}})()

    with pytest.raises(SSRFSecurityException, match="Blocked loopback IP: 127.0.0.1"):
        SSRFProtector._check_redirect_depth("https://public.com/start")

    assert mock_head.call_count == 1
    assert mock_head.call_args[0][0] == "https://public.com/start"
