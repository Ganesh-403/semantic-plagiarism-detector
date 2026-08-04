import ipaddress
import logging
import socket
import time
import urllib.parse
from typing import Dict

import requests

from src.errors import (
    SSRF_BLOCKED_LINK_LOCAL,
    SSRF_BLOCKED_LOOPBACK,
    SSRF_BLOCKED_MULTICAST,
    SSRF_BLOCKED_PRIVATE,
    SSRF_BLOCKED_UNSPECIFIED,
    SSRF_DNS_NO_ADDRESSES,
    SSRF_DNS_RESOLUTION_FAILED,
    SSRF_DOMAIN_NOT_ALLOWED,
    SSRF_MAX_REDIRECTS_EXCEEDED,
    SSRF_WEBHOOK_URL_EMPTY,
    SSRF_INSECURE_SCHEME,
    SSRF_INVALID_IP_FORMAT,
    SSRF_MISSING_HOSTNAME,
)


logger = logging.getLogger(__name__)


class SSRFSecurityException(Exception):
    """Raised when a Webhook URL fails SSRF security checks."""

    pass


class SSRFProtector:
    """
    Core security module designed to prevent Server-Side Request Forgery (SSRF)
    attacks via the Webhook feature. Includes DNS rebinding protection caching.
    """

    # Simple in-memory cache to prevent repeated DNS lookups and mitigate
    # slow-DNS denial of service attacks. (Format: {hostname: (ip_str, timestamp)})
    _dns_cache: Dict[str, tuple[str, float]] = {}
    DNS_CACHE_TTL_SECONDS = 300  # 5 minutes
    BLOCKED_PRIVATE_IPV4_SUBNETS = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    ALLOWED_CIDRS: tuple[ipaddress._BaseNetwork, ...] = ()

    @classmethod
    def _resolve_hostname(cls, hostname: str) -> str:
        """
        Resolves a hostname to an IP address with a caching layer.
        """
        current_time = time.time()

        # Check cache first
        if hostname in cls._dns_cache:
            cached_ip, timestamp = cls._dns_cache[hostname]
            if current_time - timestamp < cls.DNS_CACHE_TTL_SECONDS:
                return cached_ip

        # Cache miss or expired, perform DNS resolution
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            if not addr_info:
                raise SSRFSecurityException(
                    SSRF_DNS_NO_ADDRESSES.format(hostname=hostname)
                )

            ip_str = addr_info[0][4][0]
            cls._dns_cache[hostname] = (ip_str, current_time)
            return ip_str

        except socket.gaierror as e:
            raise SSRFSecurityException(
                SSRF_DNS_RESOLUTION_FAILED.format(hostname=hostname, error=e)
            )

    @classmethod
    def _check_redirect_depth(cls, url: str, max_redirects: int = 3) -> None:
        """
        Follows HTTP redirects (301/302/303/307/308) one hop at a time and
        raises if the chain goes deeper than max_redirects.
        """
        current_url = url
        redirect_count = 0
        while True:
            response = requests.head(current_url, allow_redirects=False, timeout=5)
            if response.status_code not in (301, 302, 303, 307, 308):
                return
            redirect_count += 1
            if redirect_count > max_redirects:
                raise SSRFSecurityException(SSRF_MAX_REDIRECTS_EXCEEDED)
            location = response.headers.get("Location")
            if not location:
                return
            current_url = urllib.parse.urljoin(current_url, location)

    @classmethod
    def validate_webhook_url(
        cls,
        url: str,
        allowed_domains: list[str] | None = None,
        max_redirects: int = 3,
    ) -> bool:
        """
        Validates that a provided webhook URL is safe to dispatch.
        Ensures the URL uses HTTPS, its domain is in ALLOWED_WEBHOOK_DOMAINS (if configured),
        and does not resolve to any internal network IP.

        Args:
            url: The webhook URL string
            allowed_domains: Optional list of allowed domain hostnames. If None,
                fetches configured domains via ``get_allowed_webhook_domains()``.

        Returns:
            True if the URL is strictly safe.

        Raises:
            SSRFSecurityException: If the URL is malicious or unapproved.
        """
        if not url:
            raise SSRFSecurityException(SSRF_WEBHOOK_URL_EMPTY)

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise SSRFSecurityException(
                SSRF_INSECURE_SCHEME.format(scheme=parsed.scheme)
            )

        hostname = parsed.hostname
        if not hostname:
            raise SSRFSecurityException(SSRF_MISSING_HOSTNAME)

        # Domain whitelist validation
        if allowed_domains is None:
            from src.core.app_config import get_allowed_webhook_domains

            allowed_domains = get_allowed_webhook_domains()

        if allowed_domains:
            host_lower = hostname.lower()
            allowed = False
            for domain in allowed_domains:
                dom_lower = domain.lower()
                if host_lower == dom_lower or host_lower.endswith("." + dom_lower):
                    allowed = True
                    break
            if not allowed:
                raise SSRFSecurityException(
                    SSRF_DOMAIN_NOT_ALLOWED.format(hostname=hostname)
                )

        # 2. DNS Resolution
        ip_str = cls._resolve_hostname(hostname)

        try:
            ip = ipaddress.ip_address(ip_str)
            if cls.ALLOWED_CIDRS:
                for network in cls.ALLOWED_CIDRS:
                    if ip in network:
                        logger.debug(
                            "SSRF whitelist matched %s in %s",
                            ip_str,
                            network,
                        )
                        return True
        except ValueError as e:
            raise SSRFSecurityException(SSRF_INVALID_IP_FORMAT.format(error=e))

        if isinstance(ip, ipaddress.IPv4Address):
            for subnet in cls.BLOCKED_PRIVATE_IPV4_SUBNETS:
                if ip in subnet:
                    logger.warning("Blocked SSRF attempt to target URL: %s", url)
                    raise SSRFSecurityException(SSRF_BLOCKED_PRIVATE.format(ip=ip_str))
        if ip.is_loopback:
            logger.warning("Blocked SSRF attempt to target URL: %s", url)
            raise SSRFSecurityException(SSRF_BLOCKED_LOOPBACK.format(ip=ip_str))
        if ip.is_link_local:
            logger.warning("Blocked SSRF attempt to target URL: %s", url)
            raise SSRFSecurityException(SSRF_BLOCKED_LINK_LOCAL.format(ip=ip_str))
        if ip.is_multicast:
            logger.warning("Blocked SSRF attempt to target URL: %s", url)
            raise SSRFSecurityException(SSRF_BLOCKED_MULTICAST.format(ip=ip_str))
        if ip.is_unspecified:
            logger.warning("Blocked SSRF attempt to target URL: %s", url)
            raise SSRFSecurityException(SSRF_BLOCKED_UNSPECIFIED.format(ip=ip_str))
        if ip.is_private:
            logger.warning("Blocked SSRF attempt to target URL: %s", url)
            raise SSRFSecurityException(SSRF_BLOCKED_PRIVATE.format(ip=ip_str))

        # Guard against redirect loops/deep redirect chains before declaring safe
        cls._check_redirect_depth(url, max_redirects)

        # If it passed all checks, it's considered safe (public routable IP)
        return True

    @classmethod
    def configure_allowed_cidrs(
        cls,
        allowed_cidrs: list[str] | None = None,
    ) -> None:
        """
        Configure CIDR ranges that are allowed even if they are private.
        """
        cls.ALLOWED_CIDRS = (
            tuple(ipaddress.ip_network(cidr) for cidr in allowed_cidrs)
            if allowed_cidrs
            else ()
        )
