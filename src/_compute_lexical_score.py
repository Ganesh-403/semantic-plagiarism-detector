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

import hashlib

# ... existing code ...


def _compute_lexical_score(text_a: str, text_b: str) -> float:
    """
    Computes the lexical similarity score between two texts.
    Uses deterministic MD5 hashing for cache keys to ensure compatibility
    across multiple Gunicorn workers or externalized caches.
    """
    # Generate deterministic hashes
    hash_a = hashlib.md5(text_a.encode("utf-8")).hexdigest()  # nosec
    hash_b = hashlib.md5(text_b.encode("utf-8")).hexdigest()  # nosec

    # Example cache key or cache lookup implementation
    cache_key = f"lexical_score:{hash_a}:{hash_b}"

    # ... rest of your scoring logic ...
