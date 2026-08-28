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

"""Regression tests for redirect-chain SSRF validation (Issue #2199).

``SSRFProtector.validate_url_safety`` was defined twice in the class body.
Python keeps the last definition, so the second one silently replaced the
first — and the second one did not follow redirects at all. That quietly
reverted the protections added for #1706 (re-validate every hop before
requesting it) and #1395 (maximum redirect depth), leaving
``_check_redirect_depth()`` and ``MAX_REDIRECT_DEPTH`` as unreachable code.

The attack this restores protection against: a host on an allow-listed
domain passes validation, then answers with ``302 Location:
http://169.254.169.254/...``. Under the surviving definition the "validated"
URL handed back to the caller was the original one, and nothing had checked
where it pointed.
"""

from __future__ import annotations

import ast
import inspect
import socket
from unittest.mock import Mock, patch

import pytest

from src.security import ssrf_protector
from src.security.ssrf_protector import (
    DEFAULT_USER_AGENT,
    SSRFProtector,
    SSRFSecurityException,
)

PUBLIC_IP = "93.184.216.34"
INTERNAL_IPS = {
    "metadata.evil.example": "169.254.169.254",
    "localhost.evil.example": "127.0.0.1",
    "intranet.evil.example": "10.0.0.5",
}


@pytest.fixture(autouse=True)
def clear_dns_cache():
    SSRFProtector._dns_cache.clear()
    yield
    SSRFProtector._dns_cache.clear()


@pytest.fixture
def resolver():
    """Resolve hosts in INTERNAL_IPS to their internal IP, others to public."""

    def fake_getaddrinfo(hostname, *_args, **_kwargs):
        ip = INTERNAL_IPS.get(hostname, PUBLIC_IP)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]

    with patch(
        "src.security.ssrf_protector.socket.getaddrinfo",
        side_effect=fake_getaddrinfo,
    ) as mock:
        yield mock


def head_responder(chain: dict[str, str]):
    """Build a ``requests.head`` stub driven by a {url: redirect_target} map."""

    def _head(url, *_args, **_kwargs):
        response = Mock()
        if url in chain:
            response.status_code = 302
            response.headers = {"Location": chain[url]}
        else:
            response.status_code = 200
            response.headers = {}
        return response

    return _head


# ── The duplicate definition itself ──────────────────────────────────────────


def test_validate_url_safety_is_defined_once():
    """A duplicate definition silently shadows the earlier one."""
    source = inspect.getsource(SSRFProtector)
    tree = ast.parse("class _Stub:\n" + "\n".join(source.splitlines()[1:]))

    class_body = tree.body[0].body
    names = [
        node.name
        for node in class_body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert not duplicates, f"methods defined more than once: {duplicates}"


def test_signature_exposes_redirect_and_transport_options():
    """The merged method must keep parameters from both definitions."""
    parameters = inspect.signature(SSRFProtector.validate_url_safety).parameters

    for name in ("url", "allowed_domains", "max_redirects", "user_agent", "timeout"):
        assert name in parameters, f"validate_url_safety lost the {name!r} parameter"


def test_redirect_helpers_are_reachable():
    """``_check_redirect_depth`` was dead code while the shadow was in place."""
    source = inspect.getsource(SSRFProtector.validate_url_safety)

    assert "_check_redirect_depth" in source
    assert "max_redirects" in source


# ── Redirects to internal addresses ──────────────────────────────────────────


@pytest.mark.parametrize("internal_host", sorted(INTERNAL_IPS))
def test_redirect_to_internal_address_is_blocked(resolver, internal_host):
    """The core SSRF bypass: allow-listed host redirects somewhere internal."""
    start = "https://public.example/webhook"
    chain = {start: f"https://{internal_host}/latest/meta-data/"}

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder(chain),
    ):
        with pytest.raises(SSRFSecurityException):
            SSRFProtector.validate_url_safety(start)


def test_redirect_to_internal_address_is_blocked_before_it_is_requested(resolver):
    """The internal URL must never be contacted, not merely rejected after."""
    start = "https://public.example/webhook"
    internal = "https://metadata.evil.example/latest/meta-data/"
    requested: list[str] = []

    responder = head_responder({start: internal})

    def recording_head(url, *args, **kwargs):
        requested.append(url)
        return responder(url, *args, **kwargs)

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=recording_head,
    ):
        with pytest.raises(SSRFSecurityException):
            SSRFProtector.validate_url_safety(start)

    assert internal not in requested, "the internal address was contacted"


def test_multi_hop_redirect_to_internal_address_is_blocked(resolver):
    """A late hop must be checked as strictly as the first."""
    start = "https://public.example/webhook"
    chain = {
        start: "https://hop1.example/a",
        "https://hop1.example/a": "https://hop2.example/b",
        "https://hop2.example/b": "https://intranet.evil.example/admin",
    }

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder(chain),
    ):
        with pytest.raises(SSRFSecurityException):
            SSRFProtector.validate_url_safety(start)


# ── Depth and loop guards ────────────────────────────────────────────────────


def test_redirect_depth_limit_is_enforced(resolver):
    """``MAX_REDIRECT_DEPTH`` was unreachable while the shadow was in place."""
    chain = {
        f"https://hop{index}.example/": f"https://hop{index + 1}.example/"
        for index in range(20)
    }

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder(chain),
    ):
        with pytest.raises(SSRFSecurityException, match="redirect"):
            SSRFProtector.validate_url_safety("https://hop0.example/")


def test_custom_max_redirects_is_honoured(resolver):
    """``max_redirects`` is a parameter of the merged method again."""
    chain = {
        f"https://hop{index}.example/": f"https://hop{index + 1}.example/"
        for index in range(20)
    }

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder(chain),
    ):
        with pytest.raises(SSRFSecurityException, match="redirect"):
            SSRFProtector.validate_url_safety(
                "https://hop0.example/",
                max_redirects=2,
            )


def test_circular_redirect_loop_is_detected(resolver):
    """A -> B -> A must be reported as a loop, not walked until the depth cap."""
    chain = {
        "https://a.example/": "https://b.example/",
        "https://b.example/": "https://a.example/",
    }

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder(chain),
    ):
        with pytest.raises(SSRFSecurityException, match="loop"):
            SSRFProtector.validate_url_safety("https://a.example/")


# ── Successful chains ────────────────────────────────────────────────────────


def test_returns_final_url_and_its_pinned_ip(resolver):
    """The pinned IP must belong to the final host, not the first one."""
    start = "https://public.example/webhook"
    final = "https://final.example/webhook"

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder({start: final}),
    ):
        validated_url, pinned_ip = SSRFProtector.validate_url_safety(start)

    assert validated_url == final
    assert pinned_ip == PUBLIC_IP


def test_relative_redirect_location_is_resolved(resolver):
    """A relative ``Location`` header must be joined against the current URL."""
    start = "https://public.example/webhook"

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder({start: "/redirected"}),
    ):
        validated_url, _ = SSRFProtector.validate_url_safety(start)

    assert validated_url == "https://public.example/redirected"


def test_non_redirecting_url_is_returned_unchanged(resolver):
    """The common case must not regress."""
    url = "https://public.example/webhook"

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder({}),
    ):
        validated_url, pinned_ip = SSRFProtector.validate_url_safety(url)

    assert validated_url == url
    assert pinned_ip == PUBLIC_IP


def test_non_redirecting_url_costs_exactly_one_request(resolver):
    """Validation must not issue a duplicate HEAD for the same URL."""
    url = "https://public.example/webhook"

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder({}),
    ) as mock_head:
        SSRFProtector.validate_url_safety(url)

    assert mock_head.call_count == 1


# ── Transport options are threaded through every hop ─────────────────────────


def test_user_agent_and_timeout_apply_to_every_hop(resolver):
    """The second definition accepted ``timeout`` and then ignored it."""
    start = "https://public.example/webhook"
    final = "https://final.example/webhook"

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder({start: final}),
    ) as mock_head:
        SSRFProtector.validate_url_safety(
            start,
            user_agent="CustomBot/2.0",
            timeout=1.5,
        )

    assert mock_head.call_count == 2
    for call in mock_head.call_args_list:
        assert call.kwargs["headers"] == {"User-Agent": "CustomBot/2.0"}
        assert call.kwargs["timeout"] == 1.5
        assert call.kwargs["allow_redirects"] is False


def test_default_user_agent_and_timeout_are_used(resolver):
    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder({}),
    ) as mock_head:
        SSRFProtector.validate_url_safety("https://public.example/webhook")

    call = mock_head.call_args_list[0]
    assert call.kwargs["headers"] == {"User-Agent": DEFAULT_USER_AGENT}
    assert call.kwargs["timeout"] == ssrf_protector.DEFAULT_REQUEST_TIMEOUT


def test_allow_list_is_enforced_on_every_hop(resolver):
    """A redirect off the allow-listed domain must be rejected."""
    start = "https://public.example/webhook"
    chain = {start: "https://elsewhere.example/webhook"}

    with patch(
        "src.security.ssrf_protector.requests.head",
        side_effect=head_responder(chain),
    ):
        with pytest.raises(SSRFSecurityException):
            SSRFProtector.validate_url_safety(
                start,
                allowed_domains=["public.example"],
            )


# ── The network-free validation helper ───────────────────────────────────────


def test_validate_url_target_makes_no_outbound_request(resolver):
    """``_validate_url_target`` is what makes per-hop pre-validation safe."""
    with patch("src.security.ssrf_protector.requests.head") as mock_head:
        ip = SSRFProtector._validate_url_target("https://public.example/webhook")

    assert ip == PUBLIC_IP
    mock_head.assert_not_called()


def test_validate_url_target_rejects_internal_address(resolver):
    with patch("src.security.ssrf_protector.requests.head") as mock_head:
        with pytest.raises(SSRFSecurityException):
            SSRFProtector._validate_url_target(
                "https://metadata.evil.example/latest/meta-data/"
            )

    mock_head.assert_not_called()
