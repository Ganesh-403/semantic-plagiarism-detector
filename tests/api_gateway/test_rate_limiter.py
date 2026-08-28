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

"""Unit tests for RateLimiter."""

import pytest

from src.api_gateway.rate_limiter import RateLimiter


@pytest.fixture
def limiter():
    return RateLimiter(default_limit=3, window_seconds=60)


def test_requests_under_limit_succeed(limiter):
    key = "user_1"
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True


def test_request_over_limit_fails(limiter):
    key = "user_2"
    # Exhaust allowed count of 3
    for _ in range(3):
        assert limiter.allow(key) is True

    # 4th request exceeds limit
    assert limiter.allow(key) is False


def test_separate_keys_have_independent_limits(limiter):
    key_a = "key_a"
    key_b = "key_b"

    # Exhaust key_a
    for _ in range(3):
        limiter.allow(key_a)
    assert limiter.allow(key_a) is False

    # key_b still allowed
    assert limiter.allow(key_b) is True
    assert limiter.allow(key_b) is True
