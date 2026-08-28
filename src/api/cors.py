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

# src/api/cors.py
import re
from urllib.parse import urlparse


def validate_and_parse_origin(origin_str: str) -> tuple[str | None, str | None]:
    """
    Validates a CORS origin string.
    Returns a tuple of (exact_origin, regex_pattern).
    - If it's a simple wildcard '*', returns ('*', None)
    - If it contains a wildcard subdomain (e.g. https://*.university.edu),
      returns a secure regex pattern.
    - If it's a standard URL, validates and returns it as an exact origin.
    - Raises ValueError if invalid.
    """
    origin_str = origin_str.strip()
    if not origin_str:
        raise ValueError("Origin cannot be empty.")

    if origin_str == "*":
        return "*", None

    # Handle wildcard subdomains securely, e.g., https://*.university.edu
    if "*" in origin_str:
        if origin_str.count("*") != 1 or "*.not-supported" in origin_str:
            raise ValueError(f"Invalid wildcard pattern in origin: {origin_str}")

        # Escape special regex characters except '*', then replace '*' with valid subdomain regex '[a-zA-Z0-9_-]+'
        # e.g., https://*.university.edu -> ^https://[a-zA-Z0-9_-]+\.university\.edu$
        parts = origin_str.split("*")
        if len(parts) == 2 and parts[0].endswith("://"):
            escaped_prefix = re.escape(parts[0])
            escaped_suffix = re.escape(parts[1])
            pattern = f"^{escaped_prefix}[a-zA-Z0-9_-]+{escaped_suffix}$"
            # Verify regex compilation works
            re.compile(pattern)
            return None, pattern
        else:
            raise ValueError(
                f"Wildcard '*' is only supported in the subdomain position (e.g., https://*.example.com): {origin_str}"
            )

    # Standard URL validation
    parsed = urlparse(origin_str)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL format for CORS origin: {origin_str}")

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported scheme '{parsed.scheme}' in CORS origin: {origin_str}"
        )

    if parsed.path or parsed.params or parsed.query or parsed.fragment:
        raise ValueError(
            f"CORS origin cannot contain paths or query parameters: {origin_str}"
        )

    return origin_str, None
