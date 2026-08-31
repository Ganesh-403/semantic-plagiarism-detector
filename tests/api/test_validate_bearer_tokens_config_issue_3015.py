"""
tests/api/test_validate_bearer_tokens_config_issue_3015.py
----------------------------------------------------------
Unit tests for Issue #3015: Fail-fast on malformed Bearer Token JSON.

Verifies that ``validate_bearer_tokens_config()`` raises a fatal
``RuntimeError`` when ``API_BEARER_TOKENS_MAPPING`` contains:
  - Malformed JSON (syntax errors)
  - Valid JSON that is not a dict (e.g. array, string, number)
  - A dict with non-string keys
  - A dict with non-list scopes
  - A dict with non-string scope entries

Also verifies that valid configurations and empty/unset env vars
pass validation without raising.
"""

import json
import os
from unittest.mock import patch

import pytest

from src.api.middleware import validate_bearer_tokens_config, get_valid_tokens


class TestValidateBearerTokensConfig:
    """Test suite for validate_bearer_tokens_config() (Issue #3015)."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear LRU cache before and after each test."""
        get_valid_tokens.cache_clear()
        yield
        get_valid_tokens.cache_clear()

    def test_valid_json_passes(self):
        """Valid JSON mapping should not raise."""
        valid = json.dumps({"token1": ["read", "write"], "token2": ["admin"]})
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": valid}, clear=True):
            validate_bearer_tokens_config()  # should not raise

    def test_empty_env_passes(self):
        """Unset env var should not raise."""
        with patch.dict(os.environ, {}, clear=True):
            validate_bearer_tokens_config()  # should not raise

    def test_empty_string_passes(self):
        """Empty string env var should not raise."""
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": ""}, clear=True):
            validate_bearer_tokens_config()  # should not raise

    def test_malformed_json_raises(self):
        """Malformed JSON should raise RuntimeError."""
        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": "{invalid json}"}, clear=True
        ):
            with pytest.raises(RuntimeError, match="malformed JSON"):
                validate_bearer_tokens_config()

    def test_malformed_json_missing_brace_raises(self):
        """Missing closing brace should raise RuntimeError."""
        with patch.dict(
            os.environ,
            {"API_BEARER_TOKENS_MAPPING": '{"token": ["read"'},
            clear=True,
        ):
            with pytest.raises(RuntimeError, match="malformed JSON"):
                validate_bearer_tokens_config()

    def test_non_dict_json_raises(self):
        """JSON array (not object) should raise RuntimeError."""
        with patch.dict(
            os.environ,
            {"API_BEARER_TOKENS_MAPPING": json.dumps(["token1", "token2"])},
            clear=True,
        ):
            with pytest.raises(RuntimeError, match="must be a JSON object"):
                validate_bearer_tokens_config()

    def test_non_dict_json_string_raises(self):
        """JSON string (not object) should raise RuntimeError."""
        with patch.dict(
            os.environ,
            {"API_BEARER_TOKENS_MAPPING": json.dumps("just a string")},
            clear=True,
        ):
            with pytest.raises(RuntimeError, match="must be a JSON object"):
                validate_bearer_tokens_config()

    def test_non_dict_json_number_raises(self):
        """JSON number (not object) should raise RuntimeError."""
        with patch.dict(
            os.environ,
            {"API_BEARER_TOKENS_MAPPING": "42"},
            clear=True,
        ):
            with pytest.raises(RuntimeError, match="must be a JSON object"):
                validate_bearer_tokens_config()

    def test_empty_dict_passes(self):
        """Empty JSON object should pass (no tokens configured)."""
        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": "{}"}, clear=True
        ):
            validate_bearer_tokens_config()  # should not raise

    def test_non_string_key_raises(self):
        """Non-string token key should raise RuntimeError."""
        with patch(
            "json.loads",
            return_value={123: ["read"], "valid": ["write"]},
        ):
            with patch.dict(
                os.environ,
                {"API_BEARER_TOKENS_MAPPING": "{}"},  # placeholder, json.loads is mocked
                clear=True,
            ):
                with pytest.raises(RuntimeError, match="non-string token key"):
                    validate_bearer_tokens_config()

    def test_non_list_scopes_raises(self):
        """Non-list scopes should raise RuntimeError."""
        invalid = json.dumps({"token1": "read"})
        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": invalid}, clear=True
        ):
            with pytest.raises(RuntimeError, match="non-list scopes"):
                validate_bearer_tokens_config()

    def test_non_string_scope_raises(self):
        """Non-string scope entry should raise RuntimeError."""
        invalid = json.dumps({"token1": ["read", 123]})
        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": invalid}, clear=True
        ):
            with pytest.raises(RuntimeError, match="non-string scope"):
                validate_bearer_tokens_config()

    def test_error_message_mentions_server_refusal(self):
        """Error message should mention the server is refusing to start."""
        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": "{bad}"}, clear=True
        ):
            with pytest.raises(RuntimeError, match="refusing to start"):
                validate_bearer_tokens_config()

    def test_valid_config_populates_cache(self):
        """After successful validation, get_valid_tokens() should return the parsed mapping."""
        valid = json.dumps({"my_token": ["read", "write"]})
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": valid}, clear=True):
            validate_bearer_tokens_config()
            tokens = get_valid_tokens()
            assert "my_token" in tokens
            assert tokens["my_token"] == ["read", "write"]

    def test_multiple_tokens_pass(self):
        """Multiple tokens with different scopes should pass."""
        valid = json.dumps({
            "token_a": ["read"],
            "token_b": ["read", "write"],
            "token_c": ["admin"],
        })
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": valid}, clear=True):
            validate_bearer_tokens_config()  # should not raise

    def test_empty_scopes_list_passes(self):
        """A token with an empty scopes list should pass."""
        valid = json.dumps({"token_no_scopes": []})
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": valid}, clear=True):
            validate_bearer_tokens_config()  # should not raise
