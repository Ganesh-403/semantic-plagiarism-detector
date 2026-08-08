import ipaddress
import logging
import requests
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

    _dns_cache: Dict[str, tuple[str, float]] = {}
    DNS_CACHE_TTL_SECONDS = 300  # 5 minutes
    RESTRICTED_IPV4_CIDR_BLOCKS = RESTRICTED_IPV4_CIDR_BLOCKS
    BLOCKED_PRIVATE_IPV4_SUBNETS: tuple[ipaddress.IPv4Network, ...] = (
        ipaddress.IPv4Network("10.0.0.0/8"),
        ipaddress.IPv4Network("172.16.0.0/12"),
        ipaddress.IPv4Network("192.168.0.0/16"),
        ipaddress.IPv4Network("127.0.0.0/8"),
        ipaddress.IPv4Network("169.254.0.0/16"),
    )
    ALLOWED_CIDRS: tuple[ipaddress._BaseNetwork, ...] = ()

    @classmethod
    def _resolve_hostname(cls, hostname: str) -> str:
        """
        Resolves a hostname to an IP address with a caching layer.
        """
        current_time = time.time()

        if hostname in cls._dns_cache:
            cached_ip, timestamp = cls._dns_cache[hostname]
            if current_time - timestamp < cls.DNS_CACHE_TTL_SECONDS:
                return cached_ip

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
    def _validate_ip_safety(cls, ip_str: str, url: str) -> None:
        """Validates that a resolved IP address is not private, loopback, or link-local."""
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
                        return
        except ValueError as e:
            raise SSRFSecurityException(SSRF_INVALID_IP_FORMAT.format(error=e))

        if isinstance(ip, ipaddress.IPv4Address):
            for cidr_block in cls.RESTRICTED_IPV4_CIDR_BLOCKS:
                if is_ip_in_cidr_block(ip_str, cidr_block):
                    if is_ip_in_cidr_block(ip_str, "127.0.0.0/8"):
                        logger.warning("Blocked SSRF attempt to target URL: %s", url)
                        raise SSRFSecurityException(SSRF_BLOCKED_LOOPBACK.format(ip=ip_str))
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

    @classmethod
    def _check_redirect_depth(cls, url: str, max_redirects: int = 3, skip_initial_dns: bool = False) -> None:
        """
        Follows HTTP redirects (301/302/303/307/308) one hop at a time,
        validating IP safety and scheme before making every HTTP request.
        """
        current_url = url
        visited_urls = {current_url}
        redirect_count = 0
        while True:
            parsed = urllib.parse.urlparse(current_url)
            if parsed.scheme != "https":
                raise SSRFSecurityException(
                    SSRF_INSECURE_SCHEME.format(scheme=parsed.scheme)
                )
            if not parsed.hostname:
                raise SSRFSecurityException(SSRF_MISSING_HOSTNAME)

            if redirect_count > 0 or not skip_initial_dns:
                ip_str = cls._resolve_hostname(parsed.hostname)
                cls._validate_ip_safety(ip_str, current_url)

            try:
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
                if current_url in visited_urls:
                    raise SSRFSecurityException(SSRF_CIRCULAR_REDIRECT_LOOP)
                visited_urls.add(current_url)
            except requests.RequestException:
                return

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

        # 2. DNS Resolution & Safety Checks
        ip_str = cls._resolve_hostname(hostname)
        cls._validate_ip_safety(ip_str, url)

        # Guard against redirect loops/deep redirect chains before declaring safe
        cls._check_redirect_depth(url, max_redirects, skip_initial_dns=True)

        logger.debug(f"SSRF Check passed for {url} -> {ip_str}")
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

    @classmethod
    def validate_url_safety(
        cls,
        url: str,
        allowed_domains: list[str] | None = None,
        max_redirects: int = 3,
    ) -> tuple[str, str]:
        """
        Validates that a provided URL is safe to dispatch and pins the DNS resolution.
        """
        cls.validate_webhook_url(url, allowed_domains, max_redirects)
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        ip_str, _ = cls._dns_cache[hostname]
        return url, ip_str

    @classmethod
    def pinned_request(
        cls,
        method: str,
        url: str,
        pinned_ip: str,
        **kwargs,
    ) -> requests.Response:
        """
        Executes an HTTP request pinned to a previously validated IP address.
        """
        parsed = urllib.parse.urlparse(url)
        headers = kwargs.pop("headers", {})
        headers["Host"] = parsed.hostname
        target_url = urllib.parse.urlunparse(
            (parsed.scheme, f"{pinned_ip}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}", parsed.path, parsed.params, parsed.query, parsed.fragment)
        )
        return requests.request(method, target_url, headers=headers, **kwargs)
