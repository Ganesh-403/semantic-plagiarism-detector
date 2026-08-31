import json
import logging
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

from src.db import auth as db_auth
from src.security import jwt_utils
from src.security.rate_limiter import get_token_bucket_limiter


# Expected JWT verification exceptions
_JWT_EXCEPTIONS = [ValueError]
try:
    import jwt

    _JWT_EXCEPTIONS.append(jwt.PyJWTError)
except ImportError:
    pass

try:
    from jose import JWTError

    _JWT_EXCEPTIONS.append(JWTError)
except ImportError:
    pass

JWT_EXCEPTIONS = tuple(_JWT_EXCEPTIONS)

logger = logging.getLogger(__name__)

# auto_error=False prevents FastAPI from automatically returning 403 when the header is missing,
# allowing us to manually return 401 with the correct message.
security = HTTPBearer(auto_error=False)

PUBLIC_PATH_PREFIXES = (
    "/health",
    "/healthz",
    "/metrics",
    "/auth",
    "/api/v1/auth",
    "/api/v1/version",
    "/api/v1/health",
    "/api/v1/healthz",
    "/api/v1/status",
    "/api/v1/usage",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _is_public_path(path: str) -> bool:
    """Return whether the request path is publicly accessible."""
    normalized_path = path.rstrip("/") or "/"

    return any(
        normalized_path == prefix or normalized_path.startswith(f"{prefix}/")
        for prefix in PUBLIC_PATH_PREFIXES
    )


def get_expected_bearer_token() -> str:
    """Retrieve the API Bearer Token from environment variable.

    Raises:
        HTTPException: If API_BEARER_TOKEN is not set and not in test environment.
    """
    token = os.getenv("API_BEARER_TOKEN")
    if not token:
        is_test = os.getenv("APP_ENV") == "test"
        if is_test:
            return "dev-bearer-token"
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: API_BEARER_TOKEN not set.",
        )
    return token


@lru_cache(maxsize=1)
def get_valid_tokens() -> dict[str, list[str]]:
    """Parse and return the API bearer tokens mapping from environment.

    Returns:
        Dictionary mapping token strings to lists of scope strings.
        Returns empty dict if parsing fails or env var not set.

    Note:
        Logs an error if API_BEARER_TOKENS_MAPPING contains malformed JSON
        to help administrators debug configuration issues.
    """
    tokens_json = os.getenv("API_BEARER_TOKENS_MAPPING", "")

    if not tokens_json:
        logger.info("No static API tokens configured. Relying solely on JWT auth.")
        return {}

    try:
        tokens_mapping = json.loads(tokens_json)

        # Validate structure: should be dict of str -> list of str
        if not isinstance(tokens_mapping, dict):
            logger.error(
                "API_BEARER_TOKENS_MAPPING must be a JSON object, got %s",
                type(tokens_mapping).__name__,
            )
            return {}

        # Validate each entry
        validated = {}
        for token, scopes in tokens_mapping.items():
            if not isinstance(token, str):
                logger.warning(
                    "Skipping non-string token key in API_BEARER_TOKENS_MAPPING: %s",
                    token,
                )
                continue

            if not isinstance(scopes, list):
                logger.warning("Token '%s' has non-list scopes, skipping", token)
                continue

            # Ensure all scopes are strings
            valid_scopes = [s for s in scopes if isinstance(s, str)]
            if len(valid_scopes) != len(scopes):
                logger.warning(
                    "Token '%s' has non-string scopes, filtering them out", token
                )

            validated[token] = valid_scopes

        return validated

    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse API_BEARER_TOKENS_MAPPING as JSON: %s. "
            "Scoped token authentication is disabled. "
            "Please check the environment variable for syntax errors.",
            exc,
        )
        return {}
    except Exception as exc:
        logger.error(
            "Unexpected error parsing API_BEARER_TOKENS_MAPPING: %s. "
            "Scoped token authentication is disabled.",
            exc,
            exc_info=True,
        )
        return {}

def validate_bearer_tokens_config() -> None:
    """Fail-fast validation of API_BEARER_TOKENS_MAPPING at startup (Issue #3015).

    Parses the ``API_BEARER_TOKENS_MAPPING`` environment variable and raises
    a fatal :class:`RuntimeError` if the JSON is malformed or has an invalid
    structure. This prevents the server from starting with a broken security
    configuration that would silently disable scoped token authentication.

    This function should be called from the FastAPI lifespan / startup hook.
    It is safe to call multiple times — the result is also cached in the
    :func:`get_valid_tokens` LRU cache for runtime lookups.

    Raises:
        RuntimeError: If the JSON is malformed, not a JSON object, or
                      contains unexpected types that would silently disable
                      authentication.
    """
    tokens_json = os.getenv("API_BEARER_TOKENS_MAPPING", "")

    if not tokens_json:
        # Empty / unset is a valid state — means "JWT auth only".
        logger.info("No static API tokens configured. Relying solely on JWT auth.")
        return

    try:
        tokens_mapping = json.loads(tokens_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"API_BEARER_TOKENS_MAPPING contains malformed JSON: {exc}. "
            f"Fix the environment variable or remove it to disable static tokens. "
            f"Server refusing to start with broken security configuration."
        ) from exc

    if not isinstance(tokens_mapping, dict):
        raise RuntimeError(
            f"API_BEARER_TOKENS_MAPPING must be a JSON object (dict), "
            f"got {type(tokens_mapping).__name__}. "
            f"Server refusing to start with broken security configuration."
        )

    # Validate each entry — raise on structural issues that would silently
    # disable authentication for configured tokens.
    for token, scopes in tokens_mapping.items():
        if not isinstance(token, str):
            raise RuntimeError(
                f"API_BEARER_TOKENS_MAPPING contains a non-string token key: "
                f"{token!r} (type {type(token).__name__}). "
                f"All keys must be strings. "
                f"Server refusing to start with broken security configuration."
            )

        if not isinstance(scopes, list):
            raise RuntimeError(
                f"API_BEARER_TOKENS_MAPPING token '{token}' has non-list scopes: "
                f"{scopes!r} (type {type(scopes).__name__}). "
                f"Scopes must be a list of strings. "
                f"Server refusing to start with broken security configuration."
            )

        for scope in scopes:
            if not isinstance(scope, str):
                raise RuntimeError(
                    f"API_BEARER_TOKENS_MAPPING token '{token}' has a non-string "
                    f"scope: {scope!r} (type {type(scope).__name__}). "
                    f"All scopes must be strings. "
                    f"Server refusing to start with broken security configuration."
                )

    # Pre-populate the LRU cache so runtime lookups are instant.
    get_valid_tokens.cache_clear()
    logger.info(
        "API_BEARER_TOKENS_MAPPING validated successfully: %d static token(s) configured.",
        len(tokens_mapping),
    )

async def verify_bearer_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    Validate incoming Bearer token against configured secret.
    Excludes OPTIONS requests and public endpoints.
    """
    if request.method == "OPTIONS":
        return None

    if _is_public_path(request.url.path) and not credentials:
        return None

    valid_tokens = get_valid_tokens()
    is_valid = False

    if credentials and credentials.credentials in valid_tokens:
        is_valid = True
    elif credentials and credentials.credentials:
        try:
            jwt_utils.verify_access_token(credentials.credentials)
            is_valid = True
        except JWT_EXCEPTIONS:
            is_valid = False
        except Exception:
            logger.error("Unexpected error while verifying bearer token", exc_info=True)
            is_valid = False

    if not credentials or not is_valid:
        if _is_public_path(request.url.path):
            return None
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if db_auth.is_token_revoked(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Perform token-bucket rate limiting per credentials.credentials (Issue #2921)
    if credentials and credentials.credentials:
        token_str = credentials.credentials
        limiter = get_token_bucket_limiter()

        if not limiter.consume(token_str):
            logger.warning(
                "Rate limit exceeded for API Bearer token: %s...",
                token_str[:8] if len(token_str) >= 8 else token_str,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for API Bearer token. Please slow down requests.",
                headers={"Retry-After": "1"},
            )

    return credentials.credentials


def extract_token_scopes(token: Optional[str]) -> list[str]:
    """Extract granted scopes from a bearer token (static mapping or JWT)."""
    if not token:
        return []

    valid_tokens = get_valid_tokens()
    if token in valid_tokens:
        return list(valid_tokens[token])

    try:
        payload = jwt_utils.verify_access_token(token)
        return list(payload.get("scopes", []))
    except Exception:
        return []


def _validate_scopes(
    required_scopes: Optional[Iterable[str] | list[str] | Sequence[str]],
    token_scopes: Sequence[str],
    mode: str = "all",
) -> None:
    """Validate that token_scopes satisfy required_scopes under the specified mode.

    Args:
        required_scopes: List/iterable of required scope strings.
        token_scopes: Scopes possessed by the authenticated token.
        mode: Logic to apply ('all' / 'and' or 'any' / 'or'). Defaults to 'all'.

    Raises:
        HTTPException: 403 Forbidden if the token does not have required permissions.
    """
    if not required_scopes:
        return

    normalized_mode = mode.lower().strip()
    if normalized_mode not in ("all", "and", "any", "or"):
        raise ValueError(
            f"Invalid scope evaluation mode / Invalid scope mode '{mode}'. Supported modes: 'all', 'any'."
        )

    token_scopes_set = set(token_scopes)

    # 1. ANY / OR mode: token must possess at least one of the required scopes
    if normalized_mode in ("any", "or"):
        candidate_scopes = []
        for s in required_scopes:
            if s.startswith("any:"):
                s = s[4:]
            parts = [
                p.strip()
                for p in s.replace("||", "|").replace(",", "|").split("|")
                if p.strip()
            ]
            candidate_scopes.extend(parts)

        if not any(scope in token_scopes_set for scope in candidate_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Insufficient privileges/missing required scope.",
            )
        return

    # 2. ALL / AND mode: token must satisfy every required scope entry
    for scope_expr in required_scopes:
        if scope_expr.startswith("any:"):
            inner = scope_expr[4:]
            any_parts = [
                p.strip()
                for p in inner.replace("||", "|").replace(",", "|").split("|")
                if p.strip()
            ]
            if not any(p in token_scopes_set for p in any_parts):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient privileges/missing required scope.",
                )
        elif "|" in scope_expr:
            or_parts = [
                p.strip()
                for p in scope_expr.replace("||", "|").split("|")
                if p.strip()
            ]
            if not any(p in token_scopes_set for p in or_parts):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient privileges/missing required scope.",
                )
        else:
            if scope_expr not in token_scopes_set:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient privileges/missing required scope.",
                )


def validate_scopes(
    required_scopes: Sequence[str],
    token_scopes: Sequence[str],
    mode: str = "all",
) -> bool:
    """Check if token_scopes satisfy required_scopes under 'all' (AND) or 'any' (OR) logic.

    Args:
        required_scopes: Scopes required by the route.
        token_scopes: Scopes held by the token.
        mode: Either 'all' (all required scopes must be present) or 'any' (at least one required scope must be present).
            Defaults to 'all'.

    Returns:
        bool: True if authorized, False otherwise.
    """
    try:
        _validate_scopes(required_scopes, token_scopes, mode=mode)
        return True
    except HTTPException:
        return False


async def get_current_user(
    security_scopes: SecurityScopes,
    token: Optional[str] = Depends(verify_bearer_token),
) -> dict:
    """
    Dependency to authorize the token against required scopes.
    Supports ALL (AND) logic by default, and OR (ANY) logic via pipe-syntax ('admin|manager')
    or prefix-syntax ('any:admin,manager').
    """
    if token is None:
        return {"token": None, "scopes": []}

    token_scopes = extract_token_scopes(token)
    _validate_scopes(security_scopes.scopes, token_scopes, mode="all")
    return {"token": token, "scopes": token_scopes}


async def get_current_user_any(
    security_scopes: SecurityScopes,
    token: Optional[str] = Depends(verify_bearer_token),
) -> dict:
    """
    Dependency to authorize the token against required scopes using ANY (OR) logic.
    Requires the token to possess at least one of the required scopes.
    """
    if token is None:
        return {"token": None, "scopes": []}

    token_scopes = extract_token_scopes(token)
    _validate_scopes(security_scopes.scopes, token_scopes, mode="any")
    return {"token": token, "scopes": token_scopes}


class RequireScopes:
    """Dependency injection class allowing multiple valid scopes where possessing
    either all (mode='all') or at least one (mode='any') is sufficient (Issue #3017).

    Args:
        scopes: List or iterable of required scope strings.
        mode: Logic to apply ('all' / 'and' vs 'any' / 'or'). Defaults to 'all'.
    """

    def __init__(
        self,
        scopes: Optional[list[str] | set[str] | tuple[str, ...] | Sequence[str] | str] = None,
        mode: str = "all",
    ):
        if isinstance(scopes, str):
            self.scopes = [scopes]
        else:
            self.scopes = list(scopes) if scopes else []
        self.mode = mode.lower()
        if self.mode not in ("all", "and", "any", "or"):
            raise ValueError(
                f"Invalid scope evaluation mode / Invalid scope mode '{mode}'. Supported modes: 'all', 'any'."
            )

    async def __call__(
        self,
        security_scopes: Optional[SecurityScopes] = None,
        token: Optional[str] = Depends(verify_bearer_token),
    ) -> dict:
        if token is None:
            return {"token": None, "scopes": []}

        token_scopes = extract_token_scopes(token)
        effective_scopes = list(self.scopes)
        if security_scopes and security_scopes.scopes:
            effective_scopes.extend(security_scopes.scopes)

        _validate_scopes(effective_scopes, token_scopes, mode=self.mode)
        return {"token": token, "scopes": token_scopes}


def require_scopes(
    scopes: list[str] | set[str] | tuple[str, ...] | Sequence[str] | str,
    mode: str = "all",
) -> RequireScopes:
    """Factory helper to construct a RequireScopes dependency."""
    return RequireScopes(scopes=scopes, mode=mode)


def require_any_scope(*scopes: str) -> RequireScopes:
    """Convenience dependency requiring at least one of the specified scopes (OR logic)."""
    return RequireScopes(scopes=list(scopes), mode="any")


def require_any_scopes(*scopes: str) -> RequireScopes:
    """Convenience dependency requiring at least one of the specified scopes (OR logic)."""
    return RequireScopes(scopes=list(scopes), mode="any")


def require_all_scopes(*scopes: str) -> RequireScopes:
    """Convenience dependency requiring all of the specified scopes (AND logic)."""
    return RequireScopes(scopes=list(scopes), mode="all")

