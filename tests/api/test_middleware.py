"""
tests/api/test_middleware.py
----------------------------
Unit tests for API middleware components.

Includes tests for token validation, security headers, and JSON parsing.
"""

import json
import logging
import os
from unittest.mock import patch

import pytest

from src.api.middleware import _is_public_path, get_valid_tokens


class TestGetValidTokens:
    """Test suite for get_valid_tokens() JSON parsing."""

    @pytest.fixture(autouse=True)
    def clear_tokens_cache(self):
        """Clear LRU cache before and after each test to isolate env patches."""
        get_valid_tokens.cache_clear()
        yield
        get_valid_tokens.cache_clear()

    def test_returns_empty_dict_when_env_not_set(self):
        """Verify returns empty dict when API_BEARER_TOKENS_MAPPING not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_valid_tokens()
            assert result == {}

    def test_returns_empty_dict_when_env_empty_string(self):
        """Verify returns empty dict when env var is empty string."""
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": ""}, clear=True):
            result = get_valid_tokens()
            assert result == {}

    def test_parses_valid_json_correctly(self):
        """Verify parses valid JSON mapping correctly."""
        valid_json = json.dumps({"token123": ["read", "write"], "token456": ["admin"]})

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": valid_json}, clear=True
        ):
            result = get_valid_tokens()

        assert result == {"token123": ["read", "write"], "token456": ["admin"]}

    def test_logs_error_on_malformed_json(self, caplog):
        """Verify logs error when JSON is malformed."""
        malformed_json = '{"token123": ["read", "write"'  # Missing closing brace

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": malformed_json}, clear=True
        ):
            with caplog.at_level(logging.ERROR):
                result = get_valid_tokens()

        assert result == {}
        assert any(
            "Failed to parse API_BEARER_TOKENS_MAPPING as JSON" in record.message
            for record in caplog.records
        )

    def test_logs_error_on_non_dict_json(self, caplog):
        """Verify logs error when JSON is not a dict."""
        array_json = json.dumps(["token1", "token2"])

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": array_json}, clear=True
        ):
            with caplog.at_level(logging.ERROR):
                result = get_valid_tokens()

        assert result == {}
        assert any(
            "must be a JSON object" in record.message for record in caplog.records
        )

    def test_filters_non_string_token_keys(self, caplog):
        """Verify filters out non-string token keys with warning."""
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": "{}"}, clear=True):
            with patch(
                "json.loads", return_value={"valid_token": ["read"], 123: ["write"]}
            ):
                with caplog.at_level(logging.WARNING):
                    result = get_valid_tokens()

        assert "valid_token" in result
        assert 123 not in result
        assert any(
            "Skipping non-string token key" in record.message
            for record in caplog.records
        )

    def test_filters_non_list_scopes(self, caplog):
        """Verify filters out tokens with non-list scopes."""
        invalid_scopes = json.dumps(
            {"token1": "read", "token2": ["write"]}  # Invalid: string instead of list
        )

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": invalid_scopes}, clear=True
        ):
            with caplog.at_level(logging.WARNING):
                result = get_valid_tokens()

        assert "token1" not in result
        assert "token2" in result
        assert any("has non-list scopes" in record.message for record in caplog.records)

    def test_filters_non_string_scopes(self, caplog):
        """Verify filters out non-string scope values."""
        mixed_scopes = json.dumps({"token1": ["read", 123, "write"]})  # 123 is invalid

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": mixed_scopes}, clear=True
        ):
            with caplog.at_level(logging.WARNING):
                result = get_valid_tokens()

        assert result["token1"] == ["read", "write"]  # 123 filtered out
        assert any(
            "has non-string scopes" in record.message for record in caplog.records
        )

    def test_handles_unexpected_exception(self, caplog):
        """Verify handles unexpected exceptions gracefully."""
        # Mock json.loads to raise unexpected exception
        with patch.dict(os.environ, {"API_BEARER_TOKENS_MAPPING": "{}"}, clear=True):
            with patch("json.loads", side_effect=RuntimeError("Unexpected error")):
                with caplog.at_level(logging.ERROR):
                    result = get_valid_tokens()

        assert result == {}
        assert any(
            "Unexpected error parsing API_BEARER_TOKENS_MAPPING" in record.message
            for record in caplog.records
        )

    def test_empty_dict_is_valid(self):
        """Verify empty JSON object is valid."""
        empty_json = json.dumps({})

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": empty_json}, clear=True
        ):
            result = get_valid_tokens()

        assert result == {}

    def test_complex_valid_mapping(self):
        """Verify handles complex valid mapping correctly."""
        complex_json = json.dumps(
            {
                "admin_token_xyz": ["admin", "read", "write", "delete"],
                "readonly_token_abc": ["read"],
                "limited_token_123": ["read", "write"],
            }
        )

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": complex_json}, clear=True
        ):
            result = get_valid_tokens()

        assert len(result) == 3
        assert result["admin_token_xyz"] == ["admin", "read", "write", "delete"]
        assert result["readonly_token_abc"] == ["read"]
        assert result["limited_token_123"] == ["read", "write"]

    def test_lru_cache_behavior(self):
        """Verify get_valid_tokens caches result with lru_cache."""
        get_valid_tokens.cache_clear()
        valid_json = json.dumps({"token_cached": ["read"]})

        with patch.dict(
            os.environ, {"API_BEARER_TOKENS_MAPPING": valid_json}, clear=True
        ):
            res1 = get_valid_tokens()
            res2 = get_valid_tokens()

        info = get_valid_tokens.cache_info()
        assert res1 == res2 == {"token_cached": ["read"]}
        assert info.hits >= 1
        assert info.maxsize == 1


class TestVerifyBearerToken:
    """Test suite for verify_bearer_token() exception handling."""

    def test_valid_token_verification(self):
        """Verify valid token passes verification."""
        import asyncio
        from unittest.mock import MagicMock

        from fastapi.security import HTTPAuthorizationCredentials

        from src.api.middleware import verify_bearer_token

        async def _test():
            request = MagicMock()
            request.method = "GET"
            request.url.path = "/api/v1/protected"
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid_token_123"
            )

            with patch("src.db.auth.is_token_revoked", return_value=False):
                with patch(
                    "src.security.jwt_utils.verify_access_token",
                    return_value={"sub": "user"},
                ):
                    token = await verify_bearer_token(request, creds)
                    assert token == "valid_token_123"

        asyncio.run(_test())

    def test_jwt_verification_failure_returns_401(self):
        """Verify ValueError during verification raises 401 without logging unexpected error."""
        import asyncio
        from unittest.mock import MagicMock

        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from src.api.middleware import verify_bearer_token

        async def _test():
            request = MagicMock()
            request.method = "GET"
            request.url.path = "/api/v1/protected"
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="invalid_token"
            )

            with patch(
                "src.security.jwt_utils.verify_access_token",
                side_effect=ValueError("Invalid signature"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await verify_bearer_token(request, creds)
                assert exc_info.value.status_code == 401

        asyncio.run(_test())

    def test_unexpected_exception_logs_error_and_returns_401(self, caplog):
        """Verify unexpected Exception during verification logs error with exc_info and raises 401."""
        import asyncio
        from unittest.mock import MagicMock

        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from src.api.middleware import verify_bearer_token

        async def _test():
            request = MagicMock()
            request.method = "GET"
            request.url.path = "/api/v1/protected"
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="some_token"
            )

            with patch(
                "src.security.jwt_utils.verify_access_token",
                side_effect=RuntimeError("Corrupted secret key configuration"),
            ):
                with caplog.at_level(logging.ERROR):
                    with pytest.raises(HTTPException) as exc_info:
                        await verify_bearer_token(request, creds)

                assert exc_info.value.status_code == 401
                assert any(
                    "Unexpected error while verifying bearer token" in record.message
                    for record in caplog.records
                )

        asyncio.run(_test())


class TestIsPublicPath:
    """Test public API path matching."""

    def test_exact_public_paths(self):
        """Verify configured public paths are accessible."""
        assert _is_public_path("/health")
        assert _is_public_path("/metrics")
        assert _is_public_path("/metrics/json")
        assert _is_public_path("/api/v1/auth/login")
        assert _is_public_path("/api/v1/auth/refresh")
        assert _is_public_path("/api/v1/auth/revoke")
        assert _is_public_path("/api/v1/healthz")
        assert _is_public_path("/api/v1/status")
        assert _is_public_path("/api/v1/usage")
        assert _is_public_path("/docs")
        assert _is_public_path("/redoc")
        assert _is_public_path("/openapi.json")

    def test_public_paths_allow_trailing_slashes(self):
        """Verify public paths remain accessible with trailing slashes."""
        assert _is_public_path("/health/")
        assert _is_public_path("/metrics/")
        assert _is_public_path("/metrics/json/")
        assert _is_public_path("/api/v1/auth/login/")
        assert _is_public_path("/api/v1/healthz/")
        assert _is_public_path("/api/v1/status/")
        assert _is_public_path("/docs/")

    def test_unprotected_paths_are_not_public(self):
        """Verify protected API endpoints are not treated as public."""
        assert not _is_public_path("/api/v1/scan")
        assert not _is_public_path("/api/v1/incidents")

    def test_similar_prefixes_do_not_match(self):
        """Verify partial prefix matches do not bypass authentication."""
        assert not _is_public_path("/healthcheck")
        assert not _is_public_path("/api/v1/authentication")
        assert not _is_public_path("/api/v1/status-private")


class TestVerifyBearerToken:
    """Test suite for verify_bearer_token() exception handling."""

    def test_valid_token_verification(self):
        """Verify valid token passes verification."""
        import asyncio
        from unittest.mock import MagicMock

        from fastapi.security import HTTPAuthorizationCredentials

        from src.api.middleware import verify_bearer_token

        async def _test():
            request = MagicMock()
            request.method = "GET"
            request.url.path = "/api/v1/protected"
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid_token_123"
            )

            with patch("src.db.auth.is_token_revoked", return_value=False):
                with patch(
                    "src.security.jwt_utils.verify_access_token",
                    return_value={"sub": "user"},
                ):
                    token = await verify_bearer_token(request, creds)
                    assert token == "valid_token_123"

        asyncio.run(_test())

    def test_jwt_verification_failure_returns_401(self):
        """Verify ValueError during verification raises 401 without logging unexpected error."""
        import asyncio
        from unittest.mock import MagicMock

        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from src.api.middleware import verify_bearer_token

        async def _test():
            request = MagicMock()
            request.method = "GET"
            request.url.path = "/api/v1/protected"
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="invalid_token"
            )

            with patch(
                "src.security.jwt_utils.verify_access_token",
                side_effect=ValueError("Invalid signature"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await verify_bearer_token(request, creds)
                assert exc_info.value.status_code == 401

        asyncio.run(_test())

    def test_unexpected_exception_logs_error_and_returns_401(self, caplog):
        """Verify unexpected Exception during verification logs error with exc_info and raises 401."""
        import asyncio
        from unittest.mock import MagicMock

        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from src.api.middleware import verify_bearer_token

        async def _test():
            request = MagicMock()
            request.method = "GET"
            request.url.path = "/api/v1/protected"
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="some_token"
            )

            with patch(
                "src.security.jwt_utils.verify_access_token",
                side_effect=RuntimeError("Corrupted secret key configuration"),
            ):
                with caplog.at_level(logging.ERROR):
                    with pytest.raises(HTTPException) as exc_info:
                        await verify_bearer_token(request, creds)

                assert exc_info.value.status_code == 401
                assert any(
                    "Unexpected error while verifying bearer token" in record.message
                    for record in caplog.records
                )

        asyncio.run(_test())


def test_get_current_user_jwt_without_scopes_defaults_to_empty_list():
    """Verify JWT token without explicit scopes claim defaults to [] instead of read/write."""
    import asyncio

    from fastapi.security import SecurityScopes

    from src.api.middleware import get_current_user

    async def _test():
        security_scopes = SecurityScopes(scopes=[])
        token = "jwt_without_scopes"

        payload = {"sub": "user123"}  # No "scopes" claim
        with patch("src.security.jwt_utils.verify_access_token", return_value=payload):
            with patch("src.api.middleware.get_valid_tokens", return_value={}):
                res = await get_current_user(security_scopes, token=token)
                assert res["scopes"] == []

    asyncio.run(_test())


def test_no_inline_imports_in_middleware_functions():
    """Verify src/api/middleware.py has no inline imports inside function bodies (Issue #3016)."""
    import ast
    from pathlib import Path

    middleware_path = (
        Path(__file__).resolve().parent.parent.parent / "src" / "api" / "middleware.py"
    )
    tree = ast.parse(middleware_path.read_text(encoding="utf-8"))

    # Traverse AST to ensure no Import or ImportFrom exists inside FunctionDef/AsyncFunctionDef
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is not node and isinstance(
                    child, (ast.Import, ast.ImportFrom)
                ):
                    imported_names = [alias.name for alias in child.names]
                    module_name = getattr(child, "module", "")
                    pytest.fail(
                        f"Found inline import '{module_name} -> {imported_names}' inside function "
                        f"'{node.name}' at line {child.lineno} in {middleware_path.name}"
                    )
