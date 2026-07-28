import ipaddress
import logging
import socket
import time
import urllib.parse
from typing import Dict


from src import errors

from src.errors import (
    SSRF_EMPTY_URL,
    SSRF_INSECURE_SCHEME,
    SSRF_MISSING_HOSTNAME,
    SSRF_NO_ADDRESSES,
    SSRF_DNS_RESOLUTION_FAILED,
    SSRF_INVALID_IP,
    SSRF_BLOCKED_LOOPBACK,
    SSRF_BLOCKED_PRIVATE,
    SSRF_BLOCKED_LINK_LOCAL,
    SSRF_BLOCKED_MULTICAST,
    SSRF_BLOCKED_UNSPECIFIED,
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
    DNS_CACHE_TTL_SECONDS = 300 # 5 minutes
    BLOCKED_PRIVATE_IPV4_SUBNETS = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )

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
            # socket.getaddrinfo is used to support both IPv4 and IPv6 resolution safely
            addr_info = socket.getaddrinfo(hostname, None)
            if not addr_info:


                raise SSRFSecurityException(SSRF_NO_ADDRESSES.format(hostname=hostname))


            # Extract the first resolved IP
            ip_str = addr_info[0][4][0]

            # Store in cache
            cls._dns_cache[hostname] = (ip_str, current_time)
            return ip_str

        except socket.gaierror as e:


            raise SSRFSecurityException(
                SSRF_DNS_RESOLUTION_FAILED.format(hostname=hostname, error=e)
            )


    @classmethod
    def validate_webhook_url(cls, url: str) -> bool:
        """
        Validates that a provided webhook URL is safe to dispatch.
        Ensures the URL uses HTTPS and does not resolve to any internal network IP.

        Args:
            url: The webhook URL string

        Returns:
            True if the URL is strictly safe.

        Raises:
            SSRFSecurityException: If the URL is malicious.
        """
        if not url:
            raise SSRFSecurityException(SSRF_EMPTY_URL)

        # 1. Scheme Validation
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise SSRFSecurityException(
                SSRF_INSECURE_SCHEME.format(scheme=parsed.scheme)
            )

        hostname = parsed.hostname
        if not hostname:
            raise SSRFSecurityException(SSRF_MISSING_HOSTNAME)


        # 2. DNS Resolution
        ip_str = cls._resolve_hostname(hostname)

        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as e:


            raise SSRFSecurityException(SSRF_INVALID_IP.format(error=e))
            

        # 3. Block explicit RFC1918 private IPv4 subnets using CIDR checks
        if isinstance(ip, ipaddress.IPv4Address):
            for subnet in cls.BLOCKED_PRIVATE_IPV4_SUBNETS:
                if ip in subnet:
                    raise SSRFSecurityException(SSRF_BLOCKED_PRIVATE.format(ip=ip_str))
        # 4. Block private IPv6 addresses and special-purpose ranges
        if ip.is_loopback:
            raise SSRFSecurityException(SSRF_BLOCKED_LOOPBACK.format(ip=ip_str))
        if ip.is_private:
            raise SSRFSecurityException(SSRF_BLOCKED_PRIVATE.format(ip=ip_str))
        if ip.is_link_local:
            raise SSRFSecurityException(SSRF_BLOCKED_LINK_LOCAL.format(ip=ip_str))
        if ip.is_multicast:
            raise SSRFSecurityException(SSRF_BLOCKED_MULTICAST.format(ip=ip_str))
        if ip.is_unspecified:
            raise SSRFSecurityException(SSRF_BLOCKED_UNSPECIFIED.format(ip=ip_str))

        if ip.is_private:
            raise SSRFSecurityException(SSRF_BLOCKED_PRIVATE.format(ip=ip_str))
        if ip.is_link_local:
            raise SSRFSecurityException(SSRF_BLOCKED_LINK_LOCAL.format(ip=ip_str))
        if ip.is_multicast:
            raise SSRFSecurityException(SSRF_BLOCKED_MULTICAST.format(ip=ip_str))

        if ip.is_unspecified:
            raise SSRFSecurityException(SSRF_BLOCKED_UNSPECIFIED.format(ip=ip_str))


        # If it passed all checks, it's considered safe (public routable IP)
        logger.debug(f"SSRF Check passed for {url} -> {ip_str}")
        return True
