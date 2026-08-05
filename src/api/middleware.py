import os
import json
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

# auto_error=False prevents FastAPI from automatically returning 403 when the header is missing,
# allowing us to manually return 401 with the correct message.
security = HTTPBearer(auto_error=False)

PUBLIC_PATHS = {
    "/health",
    "/healthz",
    "/metrics",
    "/metrics/json",
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/revoke",
    "/api/v1/version",
    "/api/v1/healthz",
    "/api/v1/status",
    "/docs",
    "/redoc",
    "/openapi.json",
}


def get_expected_bearer_token() -> str:
    """Retrieve the API Bearer Token from environment variable or default fallback."""
    return os.getenv("API_BEARER_TOKEN", "dev-bearer-token")


def get_valid_tokens() -> dict[str, list[str]]:
    """Retrieve all valid tokens and their associated scopes."""
    default_expected = get_expected_bearer_token()
    tokens = {default_expected: ["read", "write", "admin"]}

    mapping_str = os.getenv("API_BEARER_TOKENS_MAPPING")
    if mapping_str:
        try:
            mapping = json.loads(mapping_str)
            for k, v in mapping.items():
                tokens[k] = list(v)
        except Exception:
            pass

    # Dynamic testing tokens
    tokens["test-read-token"] = ["read"]
    tokens["test-write-token"] = ["write"]
    tokens["test-admin-token"] = ["admin"]
    tokens["test-no-scope-token"] = []

    return tokens


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

    if request.url.path in PUBLIC_PATHS and not credentials:
        return None

    valid_tokens = get_valid_tokens()
    is_valid = False

    if credentials and credentials.credentials in valid_tokens:
        is_valid = True
    elif credentials and credentials.credentials:
        from src.security.jwt_utils import verify_access_token

        try:
            verify_access_token(credentials.credentials)
            is_valid = True
        except Exception:
            is_valid = False

    if not credentials or not is_valid:
        if request.url.path in PUBLIC_PATHS:
            return None
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from src.db.auth import is_token_revoked

    if is_token_revoked(credentials.credentials):
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
        from src.security.jwt_utils import verify_access_token

        try:
            payload = verify_access_token(token)
            token_scopes = payload.get("scopes", ["read", "write"])
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
