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
tests/core/test_webhook.py
--------------------------
Unit tests for webhook delivery, retry logic, HMAC signatures, and thread safety.
"""

import ast
import inspect
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
import requests

import src.core.webhook as webhook_module
from src.core.webhook import (
    _thread_local,
    compute_webhook_signature,
    dispatch_plagiarism_alert,
    send_plagiarism_alert,
    verify_webhook_signature,
)

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "core" / "webhook.py"
)

WEBHOOK_URL = "https://mock-webhook.url"


def make_response(status_code: int) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code

    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} response",
            response=response,
        )

    return response


@pytest.fixture(autouse=True)
def disable_retry_wait(monkeypatch):
    """Keep retry tests immediate while retaining production backoff."""
    from src.core import webhook

    monkeypatch.setattr(
        webhook._post_webhook.retry,
        "sleep",
        lambda _seconds: None,
    )


@pytest.fixture(autouse=True)
def reset_thread_local():
    """Ensure thread-local storage is clean before and after each test."""
    if hasattr(_thread_local, "attempt_counter"):
        del _thread_local.attempt_counter
    yield
    if hasattr(_thread_local, "attempt_counter"):
        del _thread_local.attempt_counter


@patch.dict(os.environ, {}, clear=True)
def test_send_plagiarism_alert_no_url():
    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.95)
    assert success is False
    assert attempts == 0


@patch.dict(
    os.environ,
    {
        "PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL,
        "APP_BASE_URL": "http://test-dashboard",
    },
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_send_plagiarism_alert_success(
    mock_post,
    mock_validate_url,
):
    mock_post.return_value = make_response(200)

    success, attempts = send_plagiarism_alert(
        "student_essay.pdf",
        "wikipedia_source.pdf",
        0.925,
    )

    assert success is True
    assert attempts == 1
    mock_validate_url.assert_called_once_with(WEBHOOK_URL)
    mock_post.assert_called_once()


@patch.dict(
    os.environ,
    {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL},
)
@patch("src.core.webhook.SSRFProtector.validate_webhook_url")
@patch("src.core.webhook.requests.post")
def test_connection_error_retries_three_times(
    mock_post,
    mock_validate_url,
):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection timed out")

    success, attempts = send_plagiarism_alert("DocA", "DocB", 0.99)

    assert success is False
    assert attempts == 3
    assert mock_post.call_count == 3


class TestWebhookThreadSafety:
    """Test suite for thread-safe attempt counting (Issue #1994)."""

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_concurrent_webhook_sends_do_not_share_counters(
        self, mock_post, mock_validate_url
    ):
        """Verify that concurrent webhook deliveries maintain isolated attempt counters.

        This test simulates multiple background tasks dispatching webhooks
        simultaneously. Each thread should track its own retry attempts
        without clobbering the counters of other threads.
        """

        # Configure mock to fail twice then succeed (3 attempts total per
        # delivery). The counter is keyed on the document name rather than on
        # the thread: ThreadPoolExecutor reuses its worker threads, so a
        # thread-local counter that is never reset makes the second delivery on
        # a reused thread succeed on attempt 1 and the assertion below fail
        # intermittently depending on how the pool schedules the work.
        call_counts: dict[str, int] = {}
        counts_lock = threading.Lock()

        def side_effect(*args, **kwargs):
            payload = kwargs.get("json") or {}
            key = payload.get("text", "")

            with counts_lock:
                call_counts[key] = call_counts.get(key, 0) + 1
                attempt = call_counts[key]

            if attempt < 3:
                raise requests.exceptions.ConnectionError("Simulated timeout")

            return make_response(200)

        mock_post.side_effect = side_effect

        results = []

        def worker(worker_id):
            success, attempts = send_plagiarism_alert(
                f"DocA_{worker_id}", f"DocB_{worker_id}", 0.90
            )
            results.append((worker_id, success, attempts))

        # Run 5 concurrent webhook deliveries
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            for f in futures:
                f.result()

        # Verify each thread saw exactly 3 attempts (2 failures + 1 success)
        assert len(results) == 5
        for worker_id, success, attempts in results:
            assert success is True, f"Worker {worker_id} failed"
            assert (
                attempts == 3
            ), f"Worker {worker_id} saw {attempts} attempts instead of 3"

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_sequential_sends_reset_counter(self, mock_post, mock_validate_url):
        """Verify that sequential sends in the same thread reset the counter."""
        mock_post.return_value = make_response(200)

        success1, attempts1 = send_plagiarism_alert("DocA", "DocB", 0.90)
        success2, attempts2 = send_plagiarism_alert("DocC", "DocD", 0.85)

        assert success1 is True
        assert attempts1 == 1

        assert success2 is True
        assert attempts2 == 1  # Should be 1, not 2 (counter was reset)


class TestHMACSignatures:
    """Test suite for HMAC signature generation and verification."""

    def test_compute_signature_deterministic(self):
        payload = b'{"test": "data"}'
        sig1 = compute_webhook_signature(payload, "secret", timestamp=1000)
        sig2 = compute_webhook_signature(payload, "secret", timestamp=1000)
        assert sig1 == sig2

    def test_verify_signature_valid(self):
        payload = b'{"alert": "test"}'
        secret = "my_secret"
        timestamp = int(time.time())

        signature = compute_webhook_signature(payload, secret, timestamp=timestamp)
        assert (
            verify_webhook_signature(payload, signature, secret, timestamp=timestamp)
            is True
        )

    def test_verify_signature_invalid(self):
        payload = b'{"alert": "test"}'
        assert (
            verify_webhook_signature(
                payload, "wrong_sig", "secret", timestamp=int(time.time())
            )
            is False
        )


class TestWebhookURLParameterOverride:
    """Test suite for webhook_url parameter override (Issue #1995)."""

    @patch.dict(
        os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://env-var-url.com/webhook"}
    )
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_explicit_webhook_url_overrides_env_var(self, mock_post, mock_validate):
        """Verify that an explicit webhook_url parameter overrides the environment variable."""
        mock_post.return_value = make_response(200)
        custom_url = "https://custom-override.com/webhook"

        success, attempts = send_plagiarism_alert(
            "DocA", "DocB", 0.95, webhook_url=custom_url
        )

        assert success is True
        # Verify the custom URL was used, not the env var
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == custom_url
        assert call_args[0][0] != "https://env-var-url.com/webhook"

    @patch.dict(
        os.environ, {"PLAGIARISM_WEBHOOK_URL": "https://env-var-url.com/webhook"}
    )
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_none_webhook_url_falls_back_to_env_var(self, mock_post, mock_validate):
        """Verify that passing None falls back to the environment variable."""
        mock_post.return_value = make_response(200)

        success, attempts = send_plagiarism_alert(
            "DocA", "DocB", 0.95, webhook_url=None
        )

        assert success is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://env-var-url.com/webhook"

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_explicit_url_works_without_env_var(self, mock_post, mock_validate):
        """Verify explicit URL works even when env var is completely missing."""
        mock_post.return_value = make_response(200)
        custom_url = "https://custom-override.com/webhook"

        success, attempts = send_plagiarism_alert(
            "DocA", "DocB", 0.95, webhook_url=custom_url
        )

        assert success is True
        mock_post.assert_called_once_with(
            custom_url,
            json=mock_post.call_args[1]["json"],
            headers=mock_post.call_args[1]["headers"],
            timeout=mock_post.call_args[1]["timeout"],
        )

    @patch("src.core.webhook.send_plagiarism_alert")
    def test_dispatch_passes_webhook_url_parameter(self, mock_send):
        """Verify dispatch_plagiarism_alert correctly passes the webhook_url parameter."""
        mock_send.return_value = (True, 1)
        custom_url = "https://dispatch-override.com/webhook"

        result = dispatch_plagiarism_alert("DocA", "DocB", 0.88, webhook_url=custom_url)

        assert result is True
        mock_send.assert_called_once_with(
            doc_a="DocA", doc_b="DocB", similarity=0.88, webhook_url=custom_url
        )


# ---------------------------------------------------------------------------
# Duplicate-definition regressions (Issue #2558)
# ---------------------------------------------------------------------------


class TestNoDuplicateDefinitions:
    """``webhook.py`` defined three functions twice at module level.

    Python keeps the last definition, and for ``dispatch_plagiarism_alert``
    the last one dropped the ``webhook_url`` argument, silently reverting the
    fix from #1995. The two signature helpers were also re-defined, with
    ``# noqa: F811`` suppressing the linter that would have flagged it.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "compute_webhook_signature",
            "verify_webhook_signature",
            "dispatch_plagiarism_alert",
            "send_plagiarism_alert",
            "_post_webhook",
        ],
    )
    def test_function_is_defined_exactly_once(self, name):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        lines = [
            node.lineno
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]

        assert len(lines) == 1, (
            f"{name} is defined {len(lines)} times, at lines {lines}; the last "
            "definition silently wins"
        )

    def test_no_redefinition_suppressions_remain(self):
        """``# noqa: F811`` is what stopped flake8 reporting the duplicates."""
        source = MODULE_PATH.read_text(encoding="utf-8")

        assert "noqa: F811" not in source, (
            "a redefinition suppression is back in webhook.py - remove the "
            "duplicate definition instead of silencing the warning"
        )

    def test_module_has_a_docstring(self):
        """``from __future__`` above the docstring demotes it to dead code."""
        assert webhook_module.__doc__ is not None
        assert "Webhook notification delivery" in webhook_module.__doc__

    def test_future_import_follows_the_docstring(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

        assert isinstance(tree.body[0], ast.Expr), "docstring must come first"
        assert isinstance(tree.body[1], ast.ImportFrom)
        assert tree.body[1].module == "__future__"

    def test_signature_helpers_kept_their_full_docstrings(self):
        """The surviving copies must be the documented originals.

        The duplicate definitions had stripped the argument documentation and
        the usage examples.
        """
        for func in (compute_webhook_signature, verify_webhook_signature):
            doc = inspect.getdoc(func) or ""
            assert "Args:" in doc, f"{func.__name__} lost its Args section"
            assert "Returns:" in doc, f"{func.__name__} lost its Returns section"
            assert "Examples:" in doc, f"{func.__name__} lost its examples"

    def test_expiry_warning_reports_the_limit(self, caplog):
        """The duplicate dropped ``max_age_seconds`` from the log message."""
        payload = b'{"text": "alert"}'
        stale = int(time.time()) - 10_000
        signature = compute_webhook_signature(payload, "secret", timestamp=stale)

        with caplog.at_level("WARNING"):
            result = verify_webhook_signature(
                payload, signature, "secret", timestamp=stale, max_age_seconds=300
            )

        assert result is False
        assert "Max allowed: 300 seconds" in caplog.text


class TestDispatchForwardsWebhookUrl:
    """``dispatch_plagiarism_alert`` must forward the URL override (#1995).

    The shadowing definition accepted ``webhook_url`` and then ignored it, so
    an explicit URL was silently replaced by ``PLAGIARISM_WEBHOOK_URL`` - and
    when that env var was unset, delivery was never attempted at all, with no
    error and no log line naming the ignored argument.
    """

    OVERRIDE_URL = "https://team-b.example/hook"

    @patch("src.core.webhook.send_plagiarism_alert")
    def test_override_reaches_send_plagiarism_alert(self, mock_send):
        mock_send.return_value = (True, 1)

        result = dispatch_plagiarism_alert(
            "DocA", "DocB", 0.88, webhook_url=self.OVERRIDE_URL
        )

        assert result is True
        mock_send.assert_called_once_with(
            doc_a="DocA",
            doc_b="DocB",
            similarity=0.88,
            webhook_url=self.OVERRIDE_URL,
        )

    @patch("src.core.webhook.send_plagiarism_alert")
    def test_omitted_override_is_passed_as_none(self, mock_send):
        """``None`` is what tells send_plagiarism_alert to read the env var."""
        mock_send.return_value = (True, 1)

        dispatch_plagiarism_alert("DocA", "DocB", 0.5)

        assert mock_send.call_args.kwargs["webhook_url"] is None

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_override_is_posted_when_env_var_is_unset(
        self, mock_post, mock_validate_url
    ):
        """The end-to-end case the shadowing definition broke completely.

        With PLAGIARISM_WEBHOOK_URL unset, the old code returned False without
        ever issuing a request.
        """
        mock_post.return_value = make_response(200)

        result = dispatch_plagiarism_alert(
            "DocA", "DocB", 0.99, webhook_url=self.OVERRIDE_URL
        )

        assert result is True
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == self.OVERRIDE_URL

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_override_wins_over_the_environment_variable(
        self, mock_post, mock_validate_url
    ):
        """Per-tenant and per-severity routing depends on this precedence."""
        mock_post.return_value = make_response(200)

        dispatch_plagiarism_alert("DocA", "DocB", 0.91, webhook_url=self.OVERRIDE_URL)

        assert mock_post.call_args.args[0] == self.OVERRIDE_URL

    @patch.dict(os.environ, {"PLAGIARISM_WEBHOOK_URL": WEBHOOK_URL})
    @patch("src.core.webhook.SSRFProtector.validate_webhook_url")
    @patch("src.core.webhook.requests.post")
    def test_environment_variable_is_used_when_no_override_is_given(
        self, mock_post, mock_validate_url
    ):
        mock_post.return_value = make_response(200)

        dispatch_plagiarism_alert("DocA", "DocB", 0.91)

        assert mock_post.call_args.args[0] == WEBHOOK_URL

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_false_when_no_url_is_available(self):
        assert dispatch_plagiarism_alert("DocA", "DocB", 0.91) is False

    @patch("src.core.webhook.send_plagiarism_alert")
    def test_returns_false_when_delivery_fails(self, mock_send):
        """dispatch collapses the (success, attempts) tuple to just success."""
        mock_send.return_value = (False, 3)

        assert dispatch_plagiarism_alert("DocA", "DocB", 0.91) is False

    def test_accepts_webhook_url_as_a_keyword_argument(self):
        """Guard the public signature the API layer calls with."""
        parameters = inspect.signature(dispatch_plagiarism_alert).parameters

        assert list(parameters) == ["doc_a", "doc_b", "similarity", "webhook_url"]
        assert parameters["webhook_url"].default is None
