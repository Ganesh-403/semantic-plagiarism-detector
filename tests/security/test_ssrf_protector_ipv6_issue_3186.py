"""Regression tests for SSRFProtector IPv6 CIDR restrictions (Issue #3186)."""

from unittest.mock import patch

import pytest

from src.security.ssrf_protector import (
    RESTRICTED_IPV6_CIDR_BLOCKS,
    SSRFProtector,
    SSRFSecurityException,
    is_ip_in_cidr_block,
)


def test_restricted_ipv6_cidr_blocks_match_issue_acceptance_criteria():
    assert RESTRICTED_IPV6_CIDR_BLOCKS == (
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )


@pytest.mark.parametrize(
    ("ip_address", "cidr"),
    [
        ("::1", "::1/128"),
        ("fc00::1", "fc00::/7"),
        ("fd12:3456::1", "fc00::/7"),
        ("fe80::1", "fe80::/10"),
        ("febf:ffff::1", "fe80::/10"),
        ("ff00::1", "ff00::/8"),
        ("ffff:ffff::1", "ff00::/8"),
    ],
)
def test_ipv6_restricted_cidr_boundaries(ip_address, cidr):
    assert is_ip_in_cidr_block(ip_address, cidr) is True


@pytest.mark.parametrize(
    ("ip_address", "cidr"),
    [
        ("2001:4860:4860::8888", "::1/128"),
        ("2001:4860:4860::8888", "fc00::/7"),
        ("2001:4860:4860::8888", "fe80::/10"),
        ("2001:4860:4860::8888", "ff00::/8"),
    ],
)
def test_public_ipv6_is_not_in_restricted_cidrs(ip_address, cidr):
    assert is_ip_in_cidr_block(ip_address, cidr) is False


@pytest.mark.parametrize(
    ("ip_address", "message"),
    [
        ("::1", "Blocked loopback IP: ::1"),
        ("fd00::1", "Blocked private network IP: fd00::1"),
        ("fe80::1", "Blocked link-local IP: fe80::1"),
        ("ff00::1", "Blocked multicast IP: ff00::1"),
    ],
)
def test_validate_url_target_blocks_restricted_ipv6_ranges(ip_address, message):
    with patch.object(SSRFProtector, "_resolve_hostname", return_value=ip_address):
        with pytest.raises(SSRFSecurityException, match=message):
            SSRFProtector._validate_url_target(
                "https://ipv6-security-test.example/webhook",
                allowed_domains=[],
            )


def test_validate_url_target_allows_public_ipv6():
    public_ipv6 = "2606:4700:4700::1111"

    with patch.object(SSRFProtector, "_resolve_hostname", return_value=public_ipv6):
        assert (
            SSRFProtector._validate_url_target(
                "https://ipv6-public.example/webhook",
                allowed_domains=[],
            )
            == public_ipv6
        )
