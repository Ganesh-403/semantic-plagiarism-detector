"""
src/utils/jwt_utils.py
----------------------
JWT token generation, validation, and decoding utilities.
"""

from __future__ import annotations

import jwt
from src.metrics.prometheus import spd_auth_failures_total

JWT_SECRET = "super-secret-key"


def validate_jwt_token(token: str) -> dict | None:
    """Validate JWT token and track expired/invalid tokens via Prometheus."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        spd_auth_failures_total.labels(reason="expired_token").inc()
        return None
    except jwt.InvalidTokenError:
        spd_auth_failures_total.labels(reason="invalid_token").inc()
        return None
