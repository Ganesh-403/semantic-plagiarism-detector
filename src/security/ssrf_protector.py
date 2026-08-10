import ipaddress
import logging
import socket
import time
import urllib.parse
from typing import Dict


from src.errors import (
    SSRF_BLOCKED_LINK_LOCAL,
    SSRF_BLOCKED_LOOPBACK,
    SSRF_BLOCKED_MULTICAST,
    SSRF_BLOCKED_PRIVATE,
    SSRF_BLOCKED_UNSPECIFIED,
    SSRF_DNS_NO_ADDRESSES,
    SSRF_DNS_RESOLUTION_FAILED,
    SSRF_DOMAIN_NOT_ALLOWED,
    SSRF_WEBHOOK_URL_EMPTY,
    SSRF_INSECURE_SCHEME,
    SSRF_INVALID_IP_FORMAT,
    SSRF_MISSING_HOSTNAME,
)


logger = logging.getLogger(__name__)



RESTRICTED_IPV4_CIDR_BLOCKS = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
)


def is_ip_in_cidr_block(
    ip_str: str,
    cidr_block: str,
) -> bool:
    """Return whether an IP address belongs to a CIDR network.

    Invalid addresses, malformed CIDR values, and IP-version mismatches return
    ``False`` rather than leaking ``ipaddress`` parsing errors into callers.

    Args:
        ip_str: IPv4 or IPv6 address string.
        cidr_block: IPv4 or IPv6 network in CIDR notation.

    Returns:
        ``True`` when the address is contained in the network; otherwise
        ``False``.
    """
    try:
        ip_address = ipaddress.ip_address(ip_str)
        network = ipaddress.ip_network(
            cidr_block,
            strict=False,
        )
    except (TypeError, ValueError):
        return False

    if ip_address.version != network.version:
        return False

    return ip_address in network


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
    RESTRICTED_IPV4_CIDR_BLOCKS = RESTRICTED_IPV4_CIDR_BLOCKS

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
    def validate_webhook_url(
        cls,
        url: str,
        allowed_domains: list[str] | None = None,
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
        except ValueError as e:
            raise SSRFSecurityException(SSRF_INVALID_IP_FORMAT.format(error=e))

        if isinstance(ip, ipaddress.IPv4Address):
            for cidr_block in cls.RESTRICTED_IPV4_CIDR_BLOCKS:
                if is_ip_in_cidr_block(ip_str, cidr_block):
                    if is_ip_in_cidr_block(
                        ip_str,
                        "127.0.0.0/8",
                    ):
                        raise SSRFSecurityException(
                            SSRF_BLOCKED_LOOPBACK.format(ip=ip_str)
                        )
                    raise SSRFSecurityException(
                        SSRF_BLOCKED_PRIVATE.format(ip=ip_str)
                    )
        if ip.is_loopback:
            raise SSRFSecurityException(SSRF_BLOCKED_LOOPBACK.format(ip=ip_str))
        if ip.is_link_local:
            raise SSRFSecurityException(SSRF_BLOCKED_LINK_LOCAL.format(ip=ip_str))
        if ip.is_multicast:
            raise SSRFSecurityException(SSRF_BLOCKED_MULTICAST.format(ip=ip_str))
        if ip.is_unspecified:
            raise SSRFSecurityException(SSRF_BLOCKED_UNSPECIFIED.format(ip=ip_str))
        if ip.is_private:
            raise SSRFSecurityException(SSRF_BLOCKED_PRIVATE.format(ip=ip_str))

        # If it passed all checks, it's considered safe (public routable IP)
        logger.debug(f"SSRF Check passed for {url} -> {ip_str}")
        return True

+--- a/src/security/ssrf_protector.py
+@@ -20,6 +20,14 @@
+ import socket
+ from urllib.parse import urlparse
+ 
++# CIDR Subnet Block Filter Configuration
++CIDR_SUBNET_FILTER = [
++    '127.0.0.0/8',  # Loopback addresses
++    '10.0.0.0/8',   # Private network
++    '172.16.0.0/12',# Private network
++    '192.168.0.0/16'  # Private network
++]
++
+ 
+ class SSRFProtector:
+     def __init__(self):
+@@ -45,7 +53,15 @@ class SSRFProtector:
+         if not self.is_valid_url(url):
+             return False
+         
+-        parsed_url = urlparse(url)
++        try:
++            parsed_url = urlparse(url)
++            ip_address = socket.gethostbyname(parsed_url.hostname)
++        except (socket.gaierror, socket.herror):
++            return False  # Invalid hostname
+ 
++        if not self.is_valid_ip(ip_address):
++            return False
++
+         return True
+ 
+     def is_valid_url(self, url):
+@@ -54,6 +70,22 @@ class SSRFProtector:
+         try:
+             result = urlparse(url)
+             return all([result.scheme, result.netloc])
++        except ValueError:
++            return False  # Invalid URL format
++
++    def is_valid_ip(self, ip_address):
++        try:
++            socket.inet_aton(ip_address)
++        except socket.error:
++            return False  # Invalid IP address
++
++        for subnet in CIDR_SUBNET_FILTER:
++            if self.is_ip_in_subnet(ip_address, subnet):
++                return True
++
++        return False
++
++    def is_ip_in_subnet(self, ip_address, subnet):
++        ip_parts = [int(part) for part in ip_address.split('.')]
++        subnet_parts = [int(part) for part in subnet.split('/')[0].split('.')]
++        mask_length = int(subnet.split('/')[1])
+ 
+         return True