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
tests/security/test_rate_limiter.py
-----------------------------------
Unit tests for the Redis-backed API rate limiter.

The counting tests were rewritten for issue #2178: the limiter used to seed a
new window with SETEX(key, 1) and then fall through to an unconditional INCR,
leaving Redis at 2 after a single request and serving only ``limit - 1``
requests per window. It now increments atomically and treats the returned
post-increment count as authoritative.
"""

from unittest.mock import MagicMock, call, patch

import pytest
import redis

from src.security.rate_limiter import RateLimiter, get_rate_limit_headers


class FakeRedis:
    """A minimal in-memory stand-in for the Redis commands the limiter uses.

    MagicMock cannot model INCR's read-modify-write behaviour, and the whole
    point of this fix is that the count Redis returns is authoritative. This
    fake keeps real state so a sequence of calls can be asserted end to end.
    """

    def __init__(self):
        self.store = {}
        self.expiries = {}
        self.commands = []

    # ── commands used by RateLimiter ──────────────────────────────────────────

    def incr(self, key):
        self.commands.append(("incr", key))
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    def ttl(self, key):
        self.commands.append(("ttl", key))
        if key not in self.store:
            return -2
        return self.expiries.get(key, -1)

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        if key not in self.store:
            return False
        self.expiries[key] = seconds
        return True

    def exists(self, key):
        self.commands.append(("exists", key))
        return 1 if key in self.store else 0

    def setex(self, key, seconds, value):
        self.commands.append(("setex", key, seconds, value))
        self.store[key] = value
        self.expiries[key] = seconds
        return True

    def delete(self, key):
        self.commands.append(("delete", key))
        self.store.pop(key, None)
        self.expiries.pop(key, None)
        return True

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    """Queues commands and applies them against the parent FakeRedis."""

    def __init__(self, client):
        self.client = client
        self.queued = []

    def incr(self, key):
        self.queued.append(("incr", (key,)))
        return self

    def ttl(self, key):
        self.queued.append(("ttl", (key,)))
        return self

    def expire(self, key, seconds):
        self.queued.append(("expire", (key, seconds)))
        return self

    def setex(self, key, seconds, value):
        self.queued.append(("setex", (key, seconds, value)))
        return self

    def delete(self, key):
        self.queued.append(("delete", (key,)))
        return self

    def execute(self):
        results = [getattr(self.client, name)(*args) for name, args in self.queued]
        self.queued = []
        return results


@pytest.fixture
def fake_redis():
    """Stateful in-memory Redis substitute."""
    return FakeRedis()


@pytest.fixture
def mock_redis():
    """Call-recording Redis mock, for tests that only assert on interactions."""
    redis_mock = MagicMock(spec=redis.Redis)
    redis_mock.pipeline.return_value = MagicMock()
    return redis_mock


@pytest.fixture
def rate_limiter(fake_redis):
    """A RateLimiter with limit=5, window=60, block_duration=300."""
    return RateLimiter(
        redis_client=fake_redis,
        limit=5,
        window=60,
        block_duration=300,
        prefix="test_limit",
    )


# ── constructor validation ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": -1},
        {"window": 0},
        {"window": -60},
        {"block_duration": 0},
    ],
)
def test_constructor_rejects_non_positive_configuration(fake_redis, kwargs):
    with pytest.raises(ValueError):
        RateLimiter(redis_client=fake_redis, **kwargs)


# ── counting behaviour (issue #2178) ───────────────────────────────────────────


def test_first_request_increments_counter_exactly_once(rate_limiter, fake_redis):
    """Regression: the first request used to leave the counter at 2."""
    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is True
    assert fake_redis.store["test_limit:192.168.1.1"] == 1
    assert headers["X-RateLimit-Remaining"] == "4"


def test_exactly_limit_requests_are_served(rate_limiter, fake_redis):
    """Regression: only limit-1 requests used to get through."""
    decisions = [rate_limiter.check_rate_limit("192.168.1.1")[0] for _ in range(5)]

    assert decisions == [True, True, True, True, True]
    assert fake_redis.store["test_limit:192.168.1.1"] == 5


def test_request_after_the_limit_is_rejected(rate_limiter, fake_redis):
    for _ in range(5):
        rate_limiter.check_rate_limit("192.168.1.1")

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is False
    assert headers["X-RateLimit-Remaining"] == "0"
    assert headers["Retry-After"] == "300"
    assert "test_limit:blocked:192.168.1.1" in fake_redis.store


def test_remaining_counts_down_accurately(rate_limiter):
    """The advertised remaining quota must match what is actually left."""
    remaining = []
    for _ in range(5):
        _, headers = rate_limiter.check_rate_limit("192.168.1.1")
        remaining.append(headers["X-RateLimit-Remaining"])

    assert remaining == ["4", "3", "2", "1", "0"]


def test_last_allowed_request_reports_zero_remaining(rate_limiter):
    for _ in range(4):
        rate_limiter.check_rate_limit("192.168.1.1")

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is True
    assert headers["X-RateLimit-Remaining"] == "0"


def test_separate_identifiers_have_separate_budgets(rate_limiter):
    for _ in range(5):
        rate_limiter.check_rate_limit("10.0.0.1")

    is_allowed, _ = rate_limiter.check_rate_limit("10.0.0.2")

    assert is_allowed is True


# ── atomicity ──────────────────────────────────────────────────────────────────


def test_increment_and_ttl_are_issued_in_one_pipeline(rate_limiter, fake_redis):
    """INCR must not be preceded by a separate GET round trip."""
    rate_limiter.check_rate_limit("192.168.1.1")

    names = [entry[0] for entry in fake_redis.commands]
    assert "get" not in names
    assert names.index("incr") < names.index("ttl")


def test_concurrent_callers_cannot_both_consume_the_last_slot(rate_limiter):
    """Two callers racing on the final slot: exactly one wins.

    With the old GET-then-INCR sequence both callers could read the same
    pre-increment value and both be admitted.
    """
    for _ in range(4):
        rate_limiter.check_rate_limit("192.168.1.1")

    first, _ = rate_limiter.check_rate_limit("192.168.1.1")
    second, _ = rate_limiter.check_rate_limit("192.168.1.1")

    assert [first, second] == [True, False]


# ── window expiry handling ─────────────────────────────────────────────────────


def test_expire_is_set_only_when_the_window_opens(rate_limiter, fake_redis):
    """A sliding EXPIRE on every hit would extend the window indefinitely."""
    for _ in range(3):
        rate_limiter.check_rate_limit("192.168.1.1")

    expire_calls = [c for c in fake_redis.commands if c[0] == "expire"]
    assert len(expire_calls) == 1
    assert expire_calls[0] == ("expire", "test_limit:192.168.1.1", 60)


def test_counter_without_ttl_is_re_armed(rate_limiter, fake_redis):
    """A key that lost its expiry must not lock the client out forever."""
    fake_redis.store["test_limit:192.168.1.1"] = 2  # no entry in expiries -> TTL -1

    rate_limiter.check_rate_limit("192.168.1.1")

    assert fake_redis.expiries["test_limit:192.168.1.1"] == 60


def test_reset_header_uses_remaining_ttl_not_a_full_window(rate_limiter, fake_redis):
    rate_limiter.check_rate_limit("192.168.1.1")
    fake_redis.expiries["test_limit:192.168.1.1"] = 20  # 20s left in the window

    with patch("time.time", return_value=1600000000.0):
        _, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert headers["Retry-After"] == "20"
    assert headers["X-RateLimit-Reset"] == "1600000020"


# ── blocking ───────────────────────────────────────────────────────────────────


def test_blocked_client_is_denied_using_block_ttl(rate_limiter, fake_redis):
    fake_redis.store["test_limit:blocked:192.168.1.1"] = "1"
    fake_redis.expiries["test_limit:blocked:192.168.1.1"] = 150

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is False
    assert headers["X-RateLimit-Remaining"] == "0"
    assert headers["Retry-After"] == "150"


def test_blocked_client_does_not_consume_quota(rate_limiter, fake_redis):
    """A blocked client short-circuits before INCR."""
    fake_redis.store["test_limit:blocked:192.168.1.1"] = "1"
    fake_redis.expiries["test_limit:blocked:192.168.1.1"] = 150
    fake_redis.commands.clear()

    rate_limiter.check_rate_limit("192.168.1.1")

    assert not any(c[0] == "incr" for c in fake_redis.commands)


def test_retry_after_has_a_one_second_floor(rate_limiter, fake_redis):
    fake_redis.store["test_limit:blocked:192.168.1.1"] = "1"
    fake_redis.expiries["test_limit:blocked:192.168.1.1"] = 0

    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is False
    assert headers["Retry-After"] == "1"


def test_block_key_is_set_with_block_duration(rate_limiter, fake_redis):
    for _ in range(6):
        rate_limiter.check_rate_limit("192.168.1.1")

    assert ("setex", "test_limit:blocked:192.168.1.1", 300, "1") in fake_redis.commands


# ── reset ──────────────────────────────────────────────────────────────────────


def test_reset_limit(mock_redis):
    """Resetting deletes both the counter key and the block key."""
    limiter = RateLimiter(
        redis_client=mock_redis,
        limit=5,
        window=60,
        block_duration=300,
        prefix="test_limit",
    )

    assert limiter.reset_limit("192.168.1.1") is True

    pipeline_mock = mock_redis.pipeline.return_value
    pipeline_mock.delete.assert_has_calls(
        [call("test_limit:192.168.1.1"), call("test_limit:blocked:192.168.1.1")],
        any_order=True,
    )


def test_reset_limit_restores_full_quota(rate_limiter):
    for _ in range(5):
        rate_limiter.check_rate_limit("192.168.1.1")

    rate_limiter.reset_limit("192.168.1.1")
    is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.1")

    assert is_allowed is True
    assert headers["X-RateLimit-Remaining"] == "4"


# ── headers / convenience wrapper ──────────────────────────────────────────────


def test_get_rate_limit_headers_convenience_function(fake_redis):
    is_allowed, headers = get_rate_limit_headers(
        redis_client=fake_redis, identifier="10.0.0.1", limit=10, window=120
    )

    assert is_allowed is True
    assert headers["X-RateLimit-Limit"] == "10"
    assert headers["X-RateLimit-Remaining"] == "9"
    assert headers["Retry-After"] == "120"


def test_header_format_compliance(rate_limiter):
    """All header values must be strings, with the standard names."""
    _, headers = rate_limiter.check_rate_limit("192.168.1.1")

    for name, value in headers.items():
        assert isinstance(value, str), f"Header {name} must be a string"

    assert set(headers.keys()) == {
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    }


def test_rate_limit_header_exact_values(rate_limiter):
    rate_limiter.check_rate_limit("192.168.1.2")  # count -> 1

    with patch("time.time", return_value=1600000000.0):
        is_allowed, headers = rate_limiter.check_rate_limit("192.168.1.2")

    assert is_allowed is True
    assert headers["X-RateLimit-Limit"] == "5"
    assert headers["X-RateLimit-Remaining"] == "3"
    assert headers["X-RateLimit-Reset"] == "1600000060"
    assert headers["Retry-After"] == "60"
