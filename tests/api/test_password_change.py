import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.app import app
from src.db.auth import is_token_revoked, revoke_all_user_refresh_tokens

client = TestClient(app)

def test_change_password_success():
    """
    Assert that changing the password updates it and revokes active refresh tokens.
    """
    mock_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJzY29wZXMiOlsid3JpdGUiXSwiaWF0IjoxNzAwMDAwMDAwLCJleHAiOjE4MDAwMDAwMDB9.sig"
    
    with patch("src.security.jwt_utils.verify_access_token", return_value={"sub": "test_user"}):
        with patch("src.db.auth.authenticate_user", return_value=True) as mock_auth:
            with patch("src.db.auth.update_password") as mock_update:
                with patch("src.db.auth.revoke_all_user_refresh_tokens") as mock_revoke:
                    response = client.post(
                        "/api/v1/auth/change-password",
                        json={"old_password": "OldPassword1!", "new_password": "NewPassword2@"},
                        headers={"Authorization": f"Bearer {mock_token}"}
                    )
                    
                    assert response.status_code == 200
                    assert "Password changed successfully" in response.json()["message"]
                    mock_auth.assert_called_once_with("test_user", "OldPassword1!")
                    mock_update.assert_called_once_with("test_user", "NewPassword2@")
                    mock_revoke.assert_called_once_with("test_user")


def test_change_password_incorrect_old():
    """
    Assert that changing password fails if the old password is incorrect.
    """
    mock_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJzY29wZXMiOlsid3JpdGUiXSwiaWF0IjoxNzAwMDAwMDAwLCJleHAiOjE4MDAwMDAwMDB9.sig"
    
    with patch("src.security.jwt_utils.verify_access_token", return_value={"sub": "test_user"}):
        with patch("src.db.auth.authenticate_user", return_value=False):
            response = client.post(
                "/api/v1/auth/change-password",
                json={"old_password": "WrongOldPassword1!", "new_password": "NewPassword2@"},
                headers={"Authorization": f"Bearer {mock_token}"}
            )
            
            assert response.status_code == 400
            assert "Incorrect old password" in response.json()["detail"]


def test_token_invalidation_after_password_change():
    """
    Verify that refresh tokens issued before password_changed_at are considered revoked,
    while tokens issued after are considered active (not revoked).
    """
    # 1. Create a dummy user
    from src.db.auth import add_user, update_password, _connect
    username = "pass_change_test_user"
    
    # Clean up user if they already exist
    with _connect() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        
    add_user(username, "Password123!")
    
    # 2. Update password to set password_changed_at
    update_password(username, "NewPassword123!")
    
    # Get password_changed_at timestamp
    with _connect() as conn:
        row = conn.execute("SELECT password_changed_at FROM users WHERE username = ?", (username,)).fetchone()
        password_changed_at = row[0]
        
    password_changed_dt = datetime.fromisoformat(password_changed_at.replace("Z", "+00:00"))
    password_changed_ts = int(password_changed_dt.timestamp())
    
    # 3. Create a token issued BEFORE password_changed_at
    old_iat = password_changed_ts - 3600
    old_token = f"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwYXNzX2NoYW5nZV90ZXN0X3VzZXIiLCJpYXQiOiB7b2xkX2lhdH19.sig".replace("{old_iat}", str(old_iat))
    
    # 4. Create a token issued AFTER password_changed_at
    new_iat = password_changed_ts + 3600
    new_token = f"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwYXNzX2NoYW5nZV90ZXN0X3VzZXIiLCJpYXQiOiB7bmV3X2lhdH19.sig".replace("{new_iat}", str(new_iat))
    
    # 5. Assert that old_token is revoked, and new_token is not revoked
    assert is_token_revoked(old_token) is True
    assert is_token_revoked(new_token) is False
    
    # Clean up test user
    with _connect() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
