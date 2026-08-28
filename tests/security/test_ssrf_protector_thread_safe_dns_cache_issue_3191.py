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

"""
test_ssrf_protector_thread_safe_dns_cache_issue_3191.py
-------------------------------------------------------
Unit test suite for Issue #3191:
Validates that SSRFProtector uses a class-level threading.Lock (_cache_lock) to synchronize
all reads, updates, deletions, and evictions on _dns_cache across concurrent threads.
"""

import threading
from unittest.mock import patch

from src.security.ssrf_protector import SSRFProtector


def test_ssrf_protector_cache_lock_exists():
    """Verify SSRFProtector has class-level _cache_lock of type threading.Lock."""
    assert hasattr(SSRFProtector, "_cache_lock")
    assert isinstance(SSRFProtector._cache_lock, type(threading.Lock()))


def test_concurrent_dns_cache_resolution_thread_safety():
    """Verify concurrent DNS cache reads/updates/evictions execute safely under lock."""
    SSRFProtector.clear_dns_cache()

    errors = []

    def worker(domain_idx: int):
        domain = f"sub{domain_idx}.example.com"
        try:
            with patch(
                "socket.getaddrinfo",
                return_value=[
                    (None, None, None, None, (f"93.184.216.{domain_idx % 250}", 0))
                ],
            ):
                for _ in range(50):
                    SSRFProtector._resolve_hostname(domain)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent DNS cache access produced errors: {errors}"
    SSRFProtector.clear_dns_cache()
