"""
tests/utils/test_sso_state_payload_issue_3851.py
-------------------------------------------------
Regression tests for Issue #3851.

``src/utils/sso.py`` defined ``verify_sso_state`` twice with incompatible
signatures. The second definition (the DB-backed
``verify_sso_state(state) -> bool``) silently replaced the first, so the
stateless payload validator — ``(state, stored_state) -> (bool, error)`` — was
unreachable dead code, and with it the only use of
``STATE_EXPIRATION_SECONDS`` anywhere in the module.

The shadowed function is now ``verify_sso_state_payload``. These tests cover
it, since it has never had any, and pin the invariant that the two names stay
distinct.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import time
from unittest.mock import patch

import pytest

from src.utils import sso
from src.utils.sso import (
    STATE_EXPIRATION_SECONDS,
    verify_sso_state,
    verify_sso_state_payload,
)

MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "src" / "utils" / "sso.py"


def _payload(token: str = "google_abc", *, age_seconds: float = 0.0, **extra):
    """Build a ``state_data`` dict of the shape the auth URL builders return."""
    data = {
        "token": token,
        "created_at": time.time() - age_seconds,
        "provider": "google",
    }
    data.update(extra)
    return data


class TestBothValidatorsAreReachable:
    """The shadowing itself."""

    def test_module_exposes_both_names(self) -> None:
        assert callable(sso.verify_sso_state)
        assert callable(sso.verify_sso_state_payload)

    def test_they_are_different_functions(self) -> None:
        assert sso.verify_sso_state is not sso.verify_sso_state_payload

    def test_db_backed_validator_takes_one_argument(self) -> None:
        assert list(inspect.signature(verify_sso_state).parameters) == ["state"]

    def test_payload_validator_takes_two_arguments(self) -> None:
        assert list(inspect.signature(verify_sso_state_payload).parameters) == [
            "state",
            "stored_state",
        ]

    def test_no_top_level_function_is_defined_twice(self) -> None:
        """The redefinition ruff reported as F811 is gone.

        This also guards the rest of the module — ``sso.py`` is long enough
        that a second collision would be easy to miss.
        """
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert duplicates == []

    def test_expiration_constant_is_actually_used(self) -> None:
        """``STATE_EXPIRATION_SECONDS`` was referenced only from dead code."""
        source = inspect.getsource(verify_sso_state_payload)
        assert "STATE_EXPIRATION_SECONDS" in source

    def test_expiration_window_is_ten_minutes(self) -> None:
        assert STATE_EXPIRATION_SECONDS == 600


class TestAcceptsIssuedPayloads:
    """A payload straight out of the auth URL builders validates."""

    def test_fresh_payload_is_valid(self) -> None:
        is_valid, error = verify_sso_state_payload("google_abc", _payload())
        assert is_valid is True
        assert error is None

    def test_returns_a_two_tuple(self) -> None:
        result = verify_sso_state_payload("google_abc", _payload())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extra_keys_are_ignored(self) -> None:
        """The real payload also carries ``code_verifier`` (PKCE, #3453)."""
        payload = _payload(code_verifier="v" * 43)
        assert verify_sso_state_payload("google_abc", payload) == (True, None)

    def test_github_style_token_is_accepted(self) -> None:
        payload = _payload("github_xyz", provider="github")
        assert verify_sso_state_payload("github_xyz", payload) == (True, None)

    def test_a_payload_just_inside_the_window_is_valid(self) -> None:
        payload = _payload(age_seconds=STATE_EXPIRATION_SECONDS - 5)
        is_valid, error = verify_sso_state_payload("google_abc", payload)
        assert is_valid is True
        assert error is None

    def test_google_auth_url_payload_round_trips(self) -> None:
        """End-to-end: what ``get_google_auth_url`` hands back validates."""
        with patch.dict(
            "os.environ", {"GOOGLE_CLIENT_ID": "test-client-id"}, clear=False
        ), patch("src.db.auth.store_sso_state"):
            _url, state, state_data = sso.get_google_auth_url()

        assert verify_sso_state_payload(state, state_data) == (True, None)

    def test_google_auth_url_payload_rejects_a_different_token(self) -> None:
        with patch.dict(
            "os.environ", {"GOOGLE_CLIENT_ID": "test-client-id"}, clear=False
        ), patch("src.db.auth.store_sso_state"):
            _url, _state, state_data = sso.get_google_auth_url()

        is_valid, error = verify_sso_state_payload("google_attacker", state_data)
        assert is_valid is False
        assert error == "Invalid state token"


class TestRejectsMalformedPayloads:
    """The structural checks that were performed nowhere."""

    @pytest.mark.parametrize("empty", [None, {}, "", 0])
    def test_empty_payload_is_rejected(self, empty: object) -> None:
        is_valid, error = verify_sso_state_payload("google_abc", empty)  # type: ignore[arg-type]
        assert is_valid is False
        assert error == "Invalid state parameter"

    @pytest.mark.parametrize("wrong_type", ["a-string", ["token"], 42, 1.5])
    def test_non_dict_payload_is_rejected(self, wrong_type: object) -> None:
        is_valid, error = verify_sso_state_payload("google_abc", wrong_type)  # type: ignore[arg-type]
        assert is_valid is False
        assert error == "Invalid state data format"

    def test_missing_token_is_rejected(self) -> None:
        is_valid, error = verify_sso_state_payload(
            "google_abc", {"created_at": time.time()}
        )
        assert is_valid is False
        assert error == "Invalid state data: missing token"

    def test_blank_token_is_rejected(self) -> None:
        is_valid, error = verify_sso_state_payload(
            "google_abc", {"token": "", "created_at": time.time()}
        )
        assert is_valid is False
        assert error == "Invalid state data: missing token"

    def test_mismatched_token_is_rejected(self) -> None:
        is_valid, error = verify_sso_state_payload("google_forged", _payload())
        assert is_valid is False
        assert error == "Invalid state token"

    def test_token_comparison_is_exact(self) -> None:
        """No prefix, case or whitespace tolerance on the token."""
        for candidate in ("google_ab", "GOOGLE_ABC", " google_abc", "google_abc "):
            is_valid, _ = verify_sso_state_payload(candidate, _payload())
            assert is_valid is False

    def test_missing_timestamp_is_rejected(self) -> None:
        is_valid, error = verify_sso_state_payload(
            "google_abc", {"token": "google_abc"}
        )
        assert is_valid is False
        assert error == "Invalid state data: missing timestamp"

    def test_zero_timestamp_is_rejected(self) -> None:
        """A falsy ``created_at`` is treated as missing, not as the epoch."""
        is_valid, error = verify_sso_state_payload(
            "google_abc", {"token": "google_abc", "created_at": 0}
        )
        assert is_valid is False
        assert error == "Invalid state data: missing timestamp"

    def test_unparseable_string_timestamp_is_rejected(self) -> None:
        is_valid, error = verify_sso_state_payload(
            "google_abc", {"token": "google_abc", "created_at": "yesterday"}
        )
        assert is_valid is False
        assert error == "Invalid state timestamp format"

    @pytest.mark.parametrize("bad_type", [[123], {"t": 1}, (1, 2), object()])
    def test_non_numeric_timestamp_type_is_rejected(self, bad_type: object) -> None:
        is_valid, error = verify_sso_state_payload(
            "google_abc", {"token": "google_abc", "created_at": bad_type}
        )
        assert is_valid is False
        assert error == "Invalid state timestamp type"

    def test_numeric_string_timestamp_is_accepted(self) -> None:
        """Session storage may round-trip the float as a string."""
        payload = {"token": "google_abc", "created_at": str(time.time())}
        assert verify_sso_state_payload("google_abc", payload) == (True, None)

    def test_integer_timestamp_is_accepted(self) -> None:
        payload = {"token": "google_abc", "created_at": int(time.time())}
        assert verify_sso_state_payload("google_abc", payload) == (True, None)


class TestExpiry:
    """The 10-minute window (#3452) that the shadowing made inert."""

    def test_expired_payload_is_rejected(self) -> None:
        payload = _payload(age_seconds=STATE_EXPIRATION_SECONDS + 60)
        is_valid, error = verify_sso_state_payload("google_abc", payload)
        assert is_valid is False
        assert error is not None
        assert "expired" in error

    def test_expiry_message_reports_both_numbers(self) -> None:
        payload = _payload(age_seconds=STATE_EXPIRATION_SECONDS + 100)
        _is_valid, error = verify_sso_state_payload("google_abc", payload)
        assert error is not None
        assert str(STATE_EXPIRATION_SECONDS) in error
        assert "700s" in error

    def test_just_past_the_window_is_rejected(self) -> None:
        payload = _payload(age_seconds=STATE_EXPIRATION_SECONDS + 1)
        is_valid, _ = verify_sso_state_payload("google_abc", payload)
        assert is_valid is False

    def test_expiry_is_checked_after_the_token_matches(self) -> None:
        """A wrong token on an expired payload still reports the mismatch."""
        payload = _payload(age_seconds=STATE_EXPIRATION_SECONDS + 60)
        _is_valid, error = verify_sso_state_payload("google_forged", payload)
        assert error == "Invalid state token"

    def test_a_payload_from_the_future_is_not_expired(self) -> None:
        """Clock skew must not turn a fresh token into an expired one."""
        payload = _payload(age_seconds=-30)
        assert verify_sso_state_payload("google_abc", payload) == (True, None)

    def test_expiry_uses_the_module_constant(self) -> None:
        """Raising the window changes the verdict — no hard-coded 600."""
        payload = _payload(age_seconds=900)
        assert verify_sso_state_payload("google_abc", payload)[0] is False

        with patch.object(sso, "STATE_EXPIRATION_SECONDS", 3600):
            assert verify_sso_state_payload("google_abc", payload) == (True, None)


class TestDbBackedValidatorIsUnchanged:
    """The definition that survived keeps behaving exactly as before."""

    def test_empty_state_is_rejected(self) -> None:
        with patch("src.db.auth.log_security_event"):
            assert verify_sso_state("") is False

    def test_none_state_is_rejected(self) -> None:
        with patch("src.db.auth.log_security_event"):
            assert verify_sso_state(None) is False  # type: ignore[arg-type]

    def test_delegates_to_validate_sso_state(self) -> None:
        with patch("src.db.auth.validate_sso_state", return_value=True) as validate:
            assert verify_sso_state("google_abc") is True
        validate.assert_called_once_with("google_abc")

    def test_returns_false_when_the_row_is_not_live(self) -> None:
        with patch("src.db.auth.validate_sso_state", return_value=False), patch(
            "src.db.auth.log_security_event"
        ):
            assert verify_sso_state("google_abc") is False

    def test_a_rejected_state_is_audited(self) -> None:
        with patch("src.db.auth.validate_sso_state", return_value=False), patch(
            "src.db.auth.log_security_event"
        ) as log_event:
            verify_sso_state("google_abc")
        assert log_event.called

    def test_a_database_error_fails_closed(self) -> None:
        with patch(
            "src.db.auth.validate_sso_state", side_effect=RuntimeError("db down")
        ), patch("src.db.auth.log_security_event"):
            assert verify_sso_state("google_abc") is False

    def test_returns_a_bool_not_a_tuple(self) -> None:
        """Callers of the surviving name must not start seeing a 2-tuple."""
        with patch("src.db.auth.validate_sso_state", return_value=True):
            assert verify_sso_state("google_abc") is True
