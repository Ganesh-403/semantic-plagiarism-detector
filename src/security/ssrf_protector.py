import ipaddress
import logging
import socket
import time
import urllib.parse
from typing import Dict, Optional

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
    DNS_CACHE_TTL_SECONDS = 300 # 5 minutes

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
                raise SSRFSecurityException(f"No addresses found for hostname '{hostname}'")
            
            # Extract the first resolved IP
            ip_str = addr_info[0][4][0]
            
            # Store in cache
            cls._dns_cache[hostname] = (ip_str, current_time)
            return ip_str
            
        except socket.gaierror as e:
            raise SSRFSecurityException(f"DNS resolution failed for hostname '{hostname}': {e}")

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
            raise SSRFSecurityException("Webhook URL cannot be empty.")
            
        # 1. Scheme Validation
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise SSRFSecurityException(f"Insecure scheme '{parsed.scheme}'. Webhooks must use 'https'.")
            
        hostname = parsed.hostname
        if not hostname:
            raise SSRFSecurityException("Invalid URL: missing hostname.")
            
        # 2. DNS Resolution
        ip_str = cls._resolve_hostname(hostname)
            
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as e:
            raise SSRFSecurityException(f"Resolved invalid IP address format: {e}")
            
        # 3. Block Private, Loopback, and Unspecified IP ranges
        if ip.is_loopback:
            raise SSRFSecurityException(f"Blocked loopback IP: {ip_str}")
            
        if ip.is_private:
            raise SSRFSecurityException(f"Blocked private network IP: {ip_str}")
            
        if ip.is_link_local:
            raise SSRFSecurityException(f"Blocked link-local IP: {ip_str}")
            
        if ip.is_multicast:
            raise SSRFSecurityException(f"Blocked multicast IP: {ip_str}")
            
        if ip.is_unspecified:
            raise SSRFSecurityException(f"Blocked unspecified IP: {ip_str}")
            
        # If it passed all checks, it's considered safe (public routable IP)
        logger.debug(f"SSRF Check passed for {url} -> {ip_str}")
        return True
