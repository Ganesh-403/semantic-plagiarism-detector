"""
src/core/auth.py
----------------
Authentication logic handling user logins and password validations.
"""

from __future__ import annotations

from src.metrics.prometheus import spd_auth_failures_total


def authenticate_user(username: str, password_attempt: str, stored_hash: str) -> bool:
    """Validate user credentials and track authentication failures via Prometheus."""
    # Example verification logic
    is_valid = verify_password(password_attempt, stored_hash)
    if not is_valid:
        spd_auth_failures_total.labels(reason="invalid_password").inc()
        return False
    return True


def verify_password(attempt: str, stored: str) -> bool:
    return attempt == stored
