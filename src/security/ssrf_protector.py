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

Recent Additions (Issue #2212):
- Ensured DEFAULT_USER_AGENT is explicitly attached to all outgoing HEAD/GET requests
- Added GET fallback when servers reject HEAD requests with 405 Method Not Allowed
- Enhanced logging for User-Agent header inspection
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
import urllib.parse
from collections import OrderedDict
from typing import Tuple

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
    SSRF_INVALID_IP_FORMAT,
    SSRF_MAX_REDIRECTS_EXCEEDED,
    SSRF_MISSING_HOSTNAME,
    SSRF_WEBHOOK_URL_EMPTY,
)

logger = logging.getLogger(__name__)

# Default User-Agent header for all outgoing validation requests (Issue #2212)
# Explicitly identifying the application prevents remote servers from blocking
# requests due to missing or generic User-Agent headers.
DEFAULT_USER_AGENT = "SemanticPlagiarismDetector/1.0"

# Default per-request timeout, in seconds, for outgoing validation requests.
DEFAULT_REQUEST_TIMEOUT = 5.0

# HTTP status codes that carry a Location header worth following.
REDIRECT_STATUS_CODES = (301, 302, 303, 307, 308)

# HTTP status codes that indicate the server rejected the HEAD method
# and we should fall back to GET for validation (Issue #2212)
HEAD_REJECTION_STATUS_CODES = (405, 501)

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
    ip_str = ip_str.strip()

    try:
        ip_address = ipaddress.ip_address(ip_str)
        network = ipaddress.ip_network(
            cidr_block,
            strict=False,
        )
    except (TypeError, ValueError):
        return False

    if ip_address.version != network.version:
        # Handle IPv4-mapped IPv6 address (e.g. ::ffff:127.0.0.1) when checked against IPv4 CIDRs
        if (
            network.version == 4
            and ip_address.version == 6
            and getattr(ip_address, "ipv4_mapped", None) is not None
        ):
            ip_address = ip_address.ipv4_mapped
        else:
            return False

    return ip_address in network


class SSRFSecurityException(Exception):
    """Raised when a Webhook URL fails SSRF security checks."""

    pass


class SSRFProtector:
    """
    Core security module designed to prevent Server-Side Request Forgery (SSRF)
    attacks via the Webhook feature. Includes DNS rebinding protection caching.

    Security Model:
        This module implements a multi-layer defense-in-depth approach:

        1. **Scheme Enforcement**: Only HTTPS URLs are permitted
        2. **Domain Allowlist** (Optional): When configured, restricts to approved domains
        3. **DNS Resolution**: Resolves hostnames to IP addresses with caching
        4. **IP Validation**: Blocks private, loopback, link-local, and multicast ranges
        5. **Redirect Chain Validation**: Follows and validates each redirect hop

    Allowlist Behavior (Critical Security Note):
        When `allowed_domains` is empty or None, the domain allowlist check is
        SKIPPED, permitting ANY external domain. This is intentional to support
        deployments that rely solely on IP-level restrictions (private/loopback
        blocking) without maintaining a domain whitelist.

        **Security Implications:**
        - Empty allowlist = Permit all external domains (subject to IP checks)
        - Non-empty allowlist = Restrict to explicitly listed domains only
        - Private IP ranges (10.x, 172.16-31.x, 192.168.x) are ALWAYS blocked
        - Loopback (127.x) and link-local (169.254.x) are ALWAYS blocked

        **Recommendation:**
        Production deployments SHOULD configure `ALLOWED_WEBHOOK_DOMAINS` to
        restrict webhooks to known, trusted endpoints (e.g., Slack, Discord).
        Empty allowlists should only be used in development/testing environments
        or when IP-level restrictions are deemed sufficient.
    """

    # Bounded LRU cache to prevent repeated DNS lookups and mitigate
    # slow-DNS denial of service attacks. (Format: {hostname: (ip_str, timestamp)})
    #
    # Backed by OrderedDict rather than a plain dict: a plain dict grows
    # without limit, so a malicious actor hitting webhook validation with
    # thousands of unique randomly generated subdomains could exhaust
    # memory over time. OrderedDict lets us cheaply track recency
    # (move_to_end on every hit/write) and evict the least-recently-used
    # entry once the cache exceeds DNS_CACHE_MAX_SIZE, capping memory
    # usage regardless of how many distinct hostnames are ever queried.
    _dns_cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
    DNS_CACHE_TTL_SECONDS = 300  # 5 minutes
    DNS_CACHE_MAX_SIZE = 1000
    RESTRICTED_IPV4_CIDR_BLOCKS = RESTRICTED_IPV4_CIDR_BLOCKS
    MAX_REDIRECT_DEPTH = 5
    DEFAULT_USER_AGENT = DEFAULT_USER_AGENT

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
                # Mark as most-recently-used so it survives future evictions.
                cls._dns_cache.move_to_end(hostname)
                return cached_ip
            # Entry expired -- drop it so the lookup below writes a fresh
            # value with a clean recency position.
            del cls._dns_cache[hostname]

        # Cache miss or expired, perform DNS resolution
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            if not addr_info:
                raise SSRFSecurityException(
                    SSRF_DNS_NO_ADDRESSES.format(hostname=hostname)
                )

            ip_str = addr_info[0][4][0]
            cls._dns_cache[hostname] = (ip_str, current_time)
            cls._dns_cache.move_to_end(hostname)
            if len(cls._dns_cache) > cls.DNS_CACHE_MAX_SIZE:
                # Evict the least-recently-used entry to keep the cache
                # bounded no matter how many distinct hostnames are seen.
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
        allowed_domains: list[str] | None = None,
    ) -> str:
        """Run every network-free safety check for a single URL.

        Covers scheme, hostname, domain allow-list, DNS resolution, and the
        internal/private/loopback/link-local/multicast IP checks. Makes no
        outbound HTTP request, so it is safe to call before deciding whether
        a URL may be contacted at all.

        Args:
            url: The URL to validate.
            allowed_domains: Optional list of allowed domain hostnames. When
                           None or empty, the domain allowlist check is skipped,
                           permitting any external domain (subject to IP-level
                           restrictions). This supports flexible deployment
                           scenarios where domain whitelisting is not required.

        Returns:
            The resolved IP address of the URL's host.

        Raises:
            SSRFSecurityException: If the URL is malicious or unapproved.

        Security Note:
            An empty `allowed_domains` list intentionally permits all external
            domains. This is NOT a security bypass - private IP ranges, loopback
            addresses, and link-local addresses are still blocked regardless of
            the allowlist configuration.
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

        # Domain whitelist validation (Issue #2434: Documented empty allowlist behavior)
        # CRITICAL: When allowed_domains is None or empty, this check is SKIPPED,
        # permitting ANY external domain. This is intentional for flexible deployments.
        # Private IP ranges are still enforced below regardless of allowlist state.
        if allowed_domains is None:
            from src.core.app_config import get_allowed_webhook_domains

            allowed_domains = get_allowed_webhook_domains()

        if allowed_domains:
            # Non-empty allowlist: Restrict to explicitly approved domains only
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
        else:
            # Empty allowlist: Permit all external domains (IP restrictions still apply)
            # This supports deployments that don't require domain whitelisting
            logger.debug(
                "Domain allowlist is empty. Permitting all external domains for %s. "
                "Private IP ranges will still be enforced.",
                hostname,
            )

        # DNS Resolution
        ip_str = cls._resolve_hostname(hostname)

        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as e:
            raise SSRFSecurityException(SSRF_INVALID_IP_FORMAT.format(error=e))

        # Handle IPv4-mapped IPv6 address (e.g., ::ffff:127.0.0.1)
        if ip.version == 6 and getattr(ip, "ipv4_mapped", None) is not None:
            ip = ip.ipv4_mapped
            ip_str = str(ip)

        # IP-level restrictions (ALWAYS enforced, regardless of allowlist state)
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

        # Passed every check: a public, routable address.
        logger.debug(f"SSRF Check passed for {url} -> {ip_str}")
        return ip_str

    @classmethod
    def _make_validation_request(
        cls,
        url: str,
        user_agent: str,
        timeout: float,
    ) -> requests.Response:
        """Make an outgoing HTTP validation request with User-Agent header.

        Attempts a HEAD request first for efficiency. If the server rejects
        HEAD with 405 Method Not Allowed or 501 Not Implemented, falls back
        to a GET request (Issue #2212).

        Args:
            url: The URL to validate.
            user_agent: User-Agent header value to attach.
            timeout: Request timeout in seconds.

        Returns:
            The HTTP response object.

        Raises:
            requests.RequestException: If both HEAD and GET fail.
        """
        headers = {"User-Agent": user_agent}

        logger.debug(
            "Making validation request to %s with User-Agent: %s", url, user_agent
        )

        try:
            # Try HEAD first (more efficient, no body download)
            response = requests.head(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
            )

            # If server rejects HEAD, fall back to GET
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
                    stream=True,  # Don't download body
                )

            return response

        except requests.RequestException as exc:
            logger.debug(f"Validation request failed for {url}: {exc}")
            raise

    @classmethod
    def validate_webhook_url(
        cls,
        url: str,
        allowed_domains: list[str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> bool:
        """
        Validates that a provided webhook URL is safe to dispatch.

        Ensures the URL uses HTTPS, its domain is in ALLOWED_WEBHOOK_DOMAINS,
        does not resolve to any internal network IP, and sends an outgoing
        HTTP validation check with explicit User-Agent header (Issue #2212).

        Args:
            url: The webhook URL string
            allowed_domains: Optional list of allowed domain hostnames.
            user_agent: Custom User-Agent header for validation requests.
            timeout: Timeout in seconds for the outgoing validation request.

        Returns:
            True if the URL is strictly safe.

        Raises:
            SSRFSecurityException: If the URL is malicious or unapproved.
        """
        cls._validate_url_target(url, allowed_domains=allowed_domains)

        # Outgoing HTTP validation request with User-Agent header (Issue #2212)
        try:
            cls._make_validation_request(url, user_agent, timeout)
        except Exception as e:
            # Network errors during validation are logged but don't fail validation
            # The URL passed all security checks, server might just be temporarily down
            logger.debug(f"Outgoing HTTP validation request failed for {url}: {e}")

        return True

    @classmethod
    def _check_redirect_depth(
        cls,
        current_url: str,
        allowed_domains: list[str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> str | None:
        """
        Inspects a single hop of a redirect chain.

        The URL is fully re-validated BEFORE any outbound HTTP request is made.

        Returns:
            The next URL in the chain if a redirect is present, else None.
        """
        cls._validate_url_target(current_url, allowed_domains=allowed_domains)

        response = cls._make_validation_request(current_url, user_agent, timeout)

        if response.status_code in REDIRECT_STATUS_CODES:
            location = response.headers.get("Location")
            if location:
                return urllib.parse.urljoin(current_url, location)
        return None

    @classmethod
    def validate_url_safety(
        cls,
        url: str,
        allowed_domains: list[str] | None = None,
        max_redirects: int | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> Tuple[str, str]:
        """
        Validates a URL and every hop of any redirect chain it produces.

        Each hop is fully re-validated before it is requested. Attaches explicit
        User-Agent header to all outgoing requests (Issue #2212).

        Args:
            url: Target URL to validate.
            allowed_domains: Optional domain whitelist. When None, the
                configured allow-list is used.
            max_redirects: Maximum number of hops to follow. Defaults to
                ``MAX_REDIRECT_DEPTH``.
            user_agent: User-Agent header sent with validation requests.
            timeout: Per-request timeout in seconds.

        Returns:
            Tuple of ``(final_validated_url, pinned_ip)``. The pinned IP
            belongs to the *final* host in the chain, not the first one.

        Raises:
            SSRFSecurityException: If the URL, or any hop it redirects to,
                fails validation, or if the chain exceeds ``max_redirects``.
        """
        if max_redirects is None:
            max_redirects = cls.MAX_REDIRECT_DEPTH

        current_url = url
        # Network-free validation of the starting URL. Each hop is validated
        # again inside _check_redirect_depth() before it is requested.
        pinned_ip = cls._validate_url_target(url, allowed_domains=allowed_domains)

        hops = 0
        seen_urls = {current_url}
        while True:
            next_url = cls._check_redirect_depth(
                current_url,
                allowed_domains=allowed_domains,
                user_agent=user_agent,
                timeout=timeout,
            )
            if next_url is None:
                # Not a redirect: this is the end of the chain.
                break

            hops += 1
            if hops > max_redirects:
                raise SSRFSecurityException(SSRF_MAX_REDIRECTS_EXCEEDED)

            if next_url in seen_urls:
                raise SSRFSecurityException(SSRF_CIRCULAR_REDIRECT_LOOP)
            seen_urls.add(next_url)

            current_url = next_url
            # Re-pin to the new host, validating it here so the returned IP
            # is never one we would have refused to contact.
            pinned_ip = cls._validate_url_target(
                current_url,
                allowed_domains=allowed_domains,
            )

        return current_url, pinned_ip
