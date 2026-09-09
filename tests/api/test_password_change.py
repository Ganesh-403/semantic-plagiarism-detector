from unittest.mock import patch
import time

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.db import auth
from src.security.jwt_utils import create_jwt_token


@pytest.fixture
def account(mock_db):
    auth.add_user("password_user", "OldPassword123!", "teacher")
    return "password_user"


def access_token(username):
    return create_jwt_token({"sub": username, "type": "access", "scopes": ["read", "write"]})


def test_change_password_success_and_revokes_cached_tokens(account):
    with patch("src.security.jwt_utils.time.time", return_value=time.time() - 60):
        old_token = access_token(account)
    assert auth.is_token_revoked(old_token) is False  # Warm the revocation cache.
    token = access_token(account)
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/change-password",
            json={"old_password": "OldPassword123!", "new_password": "NewPassword456!"},  # pragma: allowlist secret
            headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    assert auth.verify_user(account, "NewPassword456!")
    assert not auth.verify_user(account, "OldPassword123!")
    assert auth.is_token_revoked(old_token)


@pytest.mark.parametrize("old,new", [("wrong", "NewPassword456!"), ("OldPassword123!", "weak")])
def test_change_password_rejects_invalid_credentials_and_policy(account, old, new):
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/change-password",
            json={"old_password": old, "new_password": new},
            headers={"Authorization": f"Bearer {access_token(account)}"})
    assert response.status_code == 400
    assert auth.verify_user(account, "OldPassword123!")


def test_change_password_requires_authentication(account):
    with TestClient(app) as client:
        response = client.post("/api/v1/auth/change-password",
            json={"old_password": "OldPassword123!", "new_password": "NewPassword456!"})  # pragma: allowlist secret
    assert response.status_code == 401
    assert auth.verify_user(account, "OldPassword123!")
