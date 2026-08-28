"""
src/security/ssrf_protector.py
------------------------------
Server-Side Request Forgery (SSRF) protection module for webhook validation.

Provides comprehensive URL validation including:
- HTTPS scheme enforcement
- Domain allowlist verification
- DNS resolution with caching
- Private/loopback/link-local IP blocking
- Redirect chain validation with depth limits
- Explicit User-Agent header attachment (Issue #2212)
- Strict MyPy type checking compliance (Issue #3250)
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import threading
import time
import urllib.parse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import idna
import requests

from src.errors import (
    SSRF_BLOCKED_LINK_LOCAL,
    SSRF_BLOCKED_LOOPBACK,
    SSRF_BLOCKED_MULTICAST,
    SSRF_BLOCKED_PRIVATE,
    SSRF_BLOCKED_UNSPECIFIED,
    SSRF_CIRCULAR_REDIRECT_LOOP,
    SSRF_DNS_NO_ADDRESSES,
    SSRF_DNS_RESOLUTION_FAILED,
    SSRF_DOMAIN_NOT_ALLOWED,
    SSRF_INSECURE_SCHEME,
    SSRF_INVALID_HOSTNAME,
    SSRF_INVALID_IP_FORMAT,
    SSRF_MAX_REDIRECTS_EXCEEDED,
    SSRF_MISSING_HOSTNAME,
    SSRF_PORT_NOT_ALLOWED,
    SSRF_WEBHOOK_URL_EMPTY,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT: str = "SemanticPlagiarismDetector/1.0"
DEFAULT_REQUEST_TIMEOUT: float = 5.0
REDIRECT_STATUS_CODES: tuple[int, ...] = (301, 302, 303, 307, 308)
HEAD_REJECTION_STATUS_CODES: tuple[int, ...] = (405, 501)

RESTRICTED_IPV4_CIDR_BLOCKS: tuple[str, ...] = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
)

RESTRICTED_IPV6_CIDR_BLOCKS: tuple[str, ...] = (
    "::1/128",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
)


def is_ip_in_cidr_block(ip_str: str, cidr_block: str) -> bool:
    """Return whether an IP address belongs to a CIDR network under strict typing."""
    ip_str = ip_str.strip()

    try:
        ip_address = ipaddress.ip_address(ip_str)
        network = ipaddress.ip_network(cidr_block, strict=False)
    except (TypeError, ValueError):
        return False

    if ip_address.version != network.version:
        if network.version == 4 and ip_address.version == 6:
            mapped = getattr(ip_address, "ipv4_mapped", None)
            if mapped is not None:
                ip_address = mapped
            else:
                return False
        else:
            return False

    return ip_address in network


def get_allowed_webhook_domains() -> list[str]:
    """Return the deployment's configured webhook domain allowlist.

    Thin wrapper over :func:`src.core.app_config.get_allowed_webhook_domains`.
    The import stays inside the function because ``src.core`` pulls in
    ``src.core.webhook``, which imports this module -- a module-level import
    would close that cycle. Having the accessor live here also gives tests a
    single name to patch.

    Returns:
        The configured domains, or an empty list when the configuration module
        is unavailable.
    """
    try:
        from src.core.app_config import (
            get_allowed_webhook_domains as _configured_domains,
        )
    except ImportError:
        logger.warning(
            "Could not load the webhook domain allowlist; "
            "falling back to IP-level restrictions only."
        )
        return []

    return _configured_domains()


def get_user_agent(user_agent: Optional[str] = None) -> str:
    """Return a validated User-Agent for outbound SSRF validation requests.

    An explicit ``user_agent`` takes precedence. When it is omitted, the
    ``SSRF_USER_AGENT`` environment variable is used, falling back to
    ``DEFAULT_USER_AGENT``. CR/LF characters are rejected to prevent HTTP
    header injection.
    """
    configured_user_agent = (
        user_agent
        if user_agent is not None
        else os.getenv("SSRF_USER_AGENT", DEFAULT_USER_AGENT)
    )

    if "\r" in configured_user_agent or "\n" in configured_user_agent:
        raise SSRFSecurityException(
            "User-Agent must not contain carriage return or line feed characters"
        )

    return configured_user_agent


class SSRFSecurityException(Exception):
    """Raised when a Webhook URL fails SSRF security checks."""

    pass


class SSRFProtector:
    """Core security module designed to prevent SSRF attacks under strict typing."""

    _dns_cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
    _cache_lock: threading.Lock = threading.Lock()
    DNS_CACHE_TTL_SECONDS: int = 300
    DNS_CACHE_MAX_SIZE: int = 1000
    RESTRICTED_IPV4_CIDR_BLOCKS: tuple[str, ...] = RESTRICTED_IPV4_CIDR_BLOCKS
    RESTRICTED_IPV6_CIDR_BLOCKS: tuple[str, ...] = RESTRICTED_IPV6_CIDR_BLOCKS
    MAX_REDIRECT_DEPTH: int = 5
    DEFAULT_USER_AGENT: str = DEFAULT_USER_AGENT

    @classmethod
    def clear_dns_cache(cls) -> None:
        """Clear all cached DNS resolution entries in a thread-safe manner."""
        with cls._cache_lock:
            cls._dns_cache.clear()

    @classmethod
    def get_dns_cache_ttl(cls) -> int:
        """Return the configured SSRF DNS cache TTL in seconds.

        Reads SSRF_DNS_CACHE_TTL_SECONDS environment variable,
        falling back to cls.DNS_CACHE_TTL_SECONDS (default 300).
        """
        env_ttl = os.getenv("SSRF_DNS_CACHE_TTL_SECONDS")
        if env_ttl is not None:
            try:
                val = int(env_ttl)
                if val >= 0:
                    return val
            except ValueError:
                logger.warning(
                    "Invalid SSRF_DNS_CACHE_TTL_SECONDS value '%s', falling back to default.",
                    env_ttl,
                )
        return getattr(cls, "DNS_CACHE_TTL_SECONDS", 300)

    @classmethod
    def get_allowed_webhook_ports(cls) -> set[int]:
        """Return the set of allowed webhook target ports.

        Reads ALLOWED_WEBHOOK_PORTS environment variable (comma-separated integers),
        defaulting to {443}. Restricts allowed ports to prevent port scanning attacks (Issue #3190).
        """
        env_ports = os.getenv("ALLOWED_WEBHOOK_PORTS")
        if env_ports is not None and env_ports.strip():
            allowed = set()
            for item in env_ports.split(","):
                item_str = item.strip()
                if item_str.isdigit():
                    port_num = int(item_str)
                    if 1 <= port_num <= 65535:
                        allowed.add(port_num)
            if allowed:
                return allowed
        return {443}

    @classmethod
    def _resolve_hostname(cls, hostname: str) -> str:
        current_time = float(time.time())
        ttl_seconds = cls.get_dns_cache_ttl()

        with cls._cache_lock:
            if hostname in cls._dns_cache:
                cached_ip, timestamp = cls._dns_cache[hostname]
                if current_time - timestamp < ttl_seconds:
                    cls._dns_cache.move_to_end(hostname)
                    return cached_ip
                del cls._dns_cache[hostname]

        try:
            addr_info = socket.getaddrinfo(hostname, None)
            if not addr_info:
                raise SSRFSecurityException(
                    SSRF_DNS_NO_ADDRESSES.format(hostname=hostname)
                )

            ip_str = str(addr_info[0][4][0])
            with cls._cache_lock:
                cls._dns_cache[hostname] = (ip_str, current_time)
                cls._dns_cache.move_to_end(hostname)
                if len(cls._dns_cache) > cls.DNS_CACHE_MAX_SIZE:
                    cls._dns_cache.popitem(last=False)
            return ip_str

        except socket.gaierror as e:
            raise SSRFSecurityException(
                SSRF_DNS_RESOLUTION_FAILED.format(hostname=hostname, error=e)
            )

    @classmethod
    def _validate_url_target(
        cls,
        url: str,
        allowed_domains: Optional[list[str]] = None,
    ) -> str:
        """Run every network-free safety check for a single URL.

        Covers scheme, hostname encoding, the domain allowlist, DNS
        resolution, and the private/loopback/link-local/multicast IP checks.
        Makes no outbound HTTP request, so it is safe to call before deciding
        whether a URL is worth contacting at all.

        Args:
            url: The URL to validate.
            allowed_domains: Domain allowlist to enforce. ``None`` -- the
                default -- means "use the deployment's configured allowlist",
                read from ``ALLOWED_WEBHOOK_DOMAINS`` via
                :func:`get_allowed_webhook_domains`. An explicit empty list is
                the opt-out: the domain check is skipped and any external
                domain is permitted, subject to the IP-level restrictions
                below. Entries are compared after Punycode encoding, and a
                host matches an entry exactly or as a subdomain of it.

        Returns:
            The resolved IP address, as a string.

        Raises:
            SSRFSecurityException: If the URL is empty, is not HTTPS, has no
                hostname, has a hostname that cannot be IDNA-encoded, is
                outside the allowlist, fails DNS resolution, or resolves to a
                restricted address.

        Security note:
            An empty allowlist permits any external domain by design, for
            deployments that rely on IP-level restrictions alone. It is not a
            bypass: private, loopback, link-local, multicast and unspecified
            addresses are still blocked regardless of allowlist state.
        """
        if not url:
            raise SSRFSecurityException(SSRF_WEBHOOK_URL_EMPTY)

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise SSRFSecurityException(
                SSRF_INSECURE_SCHEME.format(scheme=parsed.scheme)
            )

        # Restrict allowed destination ports to prevent internal port scanning (Issue #3190)
        port = parsed.port if parsed.port is not None else 443
        allowed_ports = cls.get_allowed_webhook_ports()
        if port not in allowed_ports:
            raise SSRFSecurityException(
                SSRF_PORT_NOT_ALLOWED.format(
                    port=port, allowed_ports=sorted(list(allowed_ports))
                )
            )

        hostname: Optional[str] = parsed.hostname
        if not hostname:
            raise SSRFSecurityException(SSRF_MISSING_HOSTNAME)

        # Encode hostname to ASCII (Punycode) to prevent IDN homograph attacks
        try:
            hostname = idna.encode(hostname).decode("ascii")
        except idna.IDNAError:
            raise SSRFSecurityException(SSRF_INVALID_HOSTNAME)

        # A caller that passes nothing wants the deployment's configured
        # allowlist. An explicit empty list is the documented opt-out and is
        # left alone, so IP-level restrictions remain the only gate.
        if allowed_domains is None:
            allowed_domains = get_allowed_webhook_domains()

        # Compare against Punycode, not the raw configuration, so an allowlist
        # written as 'bücher.example' matches the 'xn--bcher-kva.example' the
        # hostname was just encoded to.
        normalized_allowed_domains = []
        for domain in allowed_domains:
            try:
                normalized_allowed_domains.append(idna.encode(domain).decode("ascii"))
            except idna.IDNAError:
                logger.warning(
                    "Ignoring un-encodable webhook allowlist entry: %s", domain
                )
                continue

        # Gate on what was *supplied*, and compare against what survived
        # normalisation. A list whose every entry is malformed therefore
        # rejects rather than falling through to "no allowlist configured" --
        # an allowlist that quietly evaluates to empty is the failure this
        # whole block exists to prevent.
        if allowed_domains:
            host_lower = hostname.lower()
            allowed = False
            for domain in normalized_allowed_domains:
                dom_lower = domain.lower()
                if host_lower == dom_lower or host_lower.endswith("." + dom_lower):
                    allowed = True
                    break
            if not allowed:
                raise SSRFSecurityException(
                    SSRF_DOMAIN_NOT_ALLOWED.format(hostname=hostname)
                )
        else:
            logger.debug(
                "Domain allowlist is empty. Permitting all external domains for %s.",
                hostname,
            )

        ip_str = cls._resolve_hostname(hostname)

        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as e:
            raise SSRFSecurityException(SSRF_INVALID_IP_FORMAT.format(error=e))

        if ip.version == 6:
            for cidr_block in cls.RESTRICTED_IPV6_CIDR_BLOCKS:
                if is_ip_in_cidr_block(ip_str, cidr_block):
                    if cidr_block == "::1/128":
                        raise SSRFSecurityException(
                            SSRF_BLOCKED_LOOPBACK.format(ip=ip_str)
                        )
                    if cidr_block == "fc00::/7":
                        raise SSRFSecurityException(
                            SSRF_BLOCKED_PRIVATE.format(ip=ip_str)
                        )
                    if cidr_block == "fe80::/10":
                        raise SSRFSecurityException(
                            SSRF_BLOCKED_LINK_LOCAL.format(ip=ip_str)
                        )
                    if cidr_block == "ff00::/8":
                        raise SSRFSecurityException(
                            SSRF_BLOCKED_MULTICAST.format(ip=ip_str)
                        )

        if ip.version == 6:
            mapped = getattr(ip, "ipv4_mapped", None)
            if mapped is not None:
                ip = mapped
                ip_str = str(ip)

        if isinstance(ip, ipaddress.IPv4Address):
            for cidr_block in cls.RESTRICTED_IPV4_CIDR_BLOCKS:
                if is_ip_in_cidr_block(ip_str, cidr_block):
                    if is_ip_in_cidr_block(ip_str, "127.0.0.0/8"):
                        raise SSRFSecurityException(
                            SSRF_BLOCKED_LOOPBACK.format(ip=ip_str)
                        )
                    raise SSRFSecurityException(SSRF_BLOCKED_PRIVATE.format(ip=ip_str))

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

        logger.debug(f"SSRF Check passed for {url} -> {ip_str}")
        return ip_str

    @classmethod
    def _make_validation_request(
        cls,
        url: str,
        user_agent: str,
        timeout: float,
    ) -> requests.Response:
        headers: dict[str, str] = {"User-Agent": user_agent}

        logger.debug(
            "Making validation request to %s with User-Agent: %s", url, user_agent
        )

        try:
            response = requests.head(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
            )

            if response.status_code in HEAD_REJECTION_STATUS_CODES:
                logger.debug(
                    "Server rejected HEAD with %d, falling back to GET for %s",
                    response.status_code,
                    url,
                )
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=False,
                    stream=True,
                )

            return response

        except requests.RequestException as exc:
            logger.debug(f"Validation request failed for {url}: {exc}")
            raise

    @classmethod
    def validate_webhook_url(
        cls,
        url: str,
        allowed_domains: Optional[list[str]] = None,
        user_agent: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> bool:
        resolved_user_agent = get_user_agent(user_agent)
        cls._validate_url_target(url, allowed_domains=allowed_domains)

        try:
            cls._make_validation_request(url, resolved_user_agent, timeout)
        except Exception as e:
            logger.debug(f"Outgoing HTTP validation request failed for {url}: {e}")

        return True

    @classmethod
    def _check_redirect_depth(
        cls,
        current_url: str,
        allowed_domains: Optional[list[str]] = None,
        user_agent: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> Optional[str]:
        resolved_user_agent = get_user_agent(user_agent)
        cls._validate_url_target(current_url, allowed_domains=allowed_domains)
        response = cls._make_validation_request(current_url, resolved_user_agent, timeout)

        if response.status_code in REDIRECT_STATUS_CODES:
            location = response.headers.get("Location")
            if location:
                return str(urllib.parse.urljoin(current_url, location))
        return None

    @classmethod
    def validate_url_safety(
        cls,
        url: str,
        allowed_domains: Optional[list[str]] = None,
        max_redirects: Optional[int] = None,
        user_agent: Optional[str] = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> tuple[str, str]:
        resolved_user_agent = get_user_agent(user_agent)
        if max_redirects is None:
            max_redirects = cls.MAX_REDIRECT_DEPTH

        current_url = url
        pinned_ip = cls._validate_url_target(url, allowed_domains=allowed_domains)

        hops = 0
        seen_urls = {current_url}
        while True:
            next_url = cls._check_redirect_depth(
                current_url,
                allowed_domains=allowed_domains,
                user_agent=resolved_user_agent,
                timeout=timeout,
            )
            if next_url is None:
                break

            hops += 1
            if hops > max_redirects:
                raise SSRFSecurityException(SSRF_MAX_REDIRECTS_EXCEEDED)

            if next_url in seen_urls:
                raise SSRFSecurityException(SSRF_CIRCULAR_REDIRECT_LOOP)
            seen_urls.add(next_url)

            current_url = next_url
            pinned_ip = cls._validate_url_target(
                current_url,
                allowed_domains=allowed_domains,
            )

        return current_url, pinned_ip
