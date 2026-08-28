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

"""Unit tests for configurable SSRF_DNS_CACHE_TTL_SECONDS in SSRFProtector (Issue #3187)."""

import os
import time
from unittest.mock import patch

import pytest

from src.security.ssrf_protector import SSRFProtector


def test_default_dns_cache_ttl():
    """Verify get_dns_cache_ttl returns 300 by default when env var is not set."""
    if "SSRF_DNS_CACHE_TTL_SECONDS" in os.environ:
        del os.environ["SSRF_DNS_CACHE_TTL_SECONDS"]

    assert SSRFProtector.get_dns_cache_ttl() == 300


def test_configurable_dns_cache_ttl_from_env():
    """Verify get_dns_cache_ttl respects SSRF_DNS_CACHE_TTL_SECONDS environment variable."""
    try:
        os.environ["SSRF_DNS_CACHE_TTL_SECONDS"] = "60"
        assert SSRFProtector.get_dns_cache_ttl() == 60

        os.environ["SSRF_DNS_CACHE_TTL_SECONDS"] = "10"
        assert SSRFProtector.get_dns_cache_ttl() == 10
    finally:
        os.environ.pop("SSRF_DNS_CACHE_TTL_SECONDS", None)


def test_invalid_dns_cache_ttl_env_var_fallback():
    """Verify invalid env var value falls back to default 300."""
    try:
        os.environ["SSRF_DNS_CACHE_TTL_SECONDS"] = "not_an_int"
        assert SSRFProtector.get_dns_cache_ttl() == 300
    finally:
        os.environ.pop("SSRF_DNS_CACHE_TTL_SECONDS", None)


@patch("src.security.ssrf_protector.socket.getaddrinfo")
def test_resolve_hostname_respects_custom_ttl(mock_getaddrinfo):
    """Verify _resolve_hostname expires cache according to configured SSRF_DNS_CACHE_TTL_SECONDS."""
    SSRFProtector._dns_cache.clear()
    mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]

    try:
        # Set short TTL of 5 seconds
        os.environ["SSRF_DNS_CACHE_TTL_SECONDS"] = "5"

        hostname = "custom-ttl.example.com"
        ip1 = SSRFProtector._resolve_hostname(hostname)
        assert mock_getaddrinfo.call_count == 1
        assert ip1 == "93.184.216.34"

        # Immediate re-call should hit cache
        ip2 = SSRFProtector._resolve_hostname(hostname)
        assert mock_getaddrinfo.call_count == 1
        assert ip2 == "93.184.216.34"

        # Simulate time passing beyond 5s TTL
        cached_ip, ts = SSRFProtector._dns_cache[hostname]
        SSRFProtector._dns_cache[hostname] = (cached_ip, ts - 10)

        # Calling again should trigger re-resolution
        ip3 = SSRFProtector._resolve_hostname(hostname)
        assert mock_getaddrinfo.call_count == 2
        assert ip3 == "93.184.216.34"

    finally:
        os.environ.pop("SSRF_DNS_CACHE_TTL_SECONDS", None)
        SSRFProtector._dns_cache.clear()
