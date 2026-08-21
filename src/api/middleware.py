import json
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

from src.db import auth as db_auth
from src.security import jwt_utils

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
    "/api/v1/auth",
    "/api/v1/version",
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
def get_valid_tokens() -> Dict[str, List[str]]:
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

    return credentials.credentials


async def get_current_user(
    security_scopes: SecurityScopes,
    token: Optional[str] = Depends(verify_bearer_token),
) -> dict:
    """
    Dependency to authorize the token against required scopes.
    """
    if token is None:
        return {"token": None, "scopes": []}

    valid_tokens = get_valid_tokens()
    if token in valid_tokens:
        token_scopes = valid_tokens[token]
    else:
        try:
            payload = jwt_utils.verify_access_token(token)
            token_scopes = payload.get("scopes", [])
        except Exception:
            token_scopes = []

    if security_scopes.scopes:
        for scope in security_scopes.scopes:
            if scope not in token_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient privileges/missing required scope.",
                )
    return {"token": token, "scopes": token_scopes}
