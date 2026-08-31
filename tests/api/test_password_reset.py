import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.app import app
from src.api.routers.auth import create_reset_token, verify_reset_token

client = TestClient(app)

def test_forgot_password_user_exists():
    """
    Assert that calling forgot-password with an existing user email/username
    returns HTTP 200 and prints the security notification.
    """
    with patch("src.db.auth._connect") as mock_connect:
        # Mock SQLite to return that user exists
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connect.return_value.__enter__.return_value.execute.return_value = mock_cursor
        
        response = client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "existing_user@openprep.ai"}
        )
        
        assert response.status_code == 200
        assert "dispatched to your email" in response.json()["message"]


def test_forgot_password_user_not_found():
    """
    Assert that calling forgot-password with a non-existent user email
    still returns HTTP 200 (anti-enumeration security).
    """
    with patch("src.db.auth._connect") as mock_connect:
        # Mock SQLite to return None (user doesn't exist)
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_connect.return_value.__enter__.return_value.execute.return_value = mock_cursor
        
        response = client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "non_existent_user@openprep.ai"}
        )
        
        assert response.status_code == 200
        assert "dispatched to your email" in response.json()["message"]


def test_reset_password_success():
    """
    Assert that calling reset-password with a valid token updates the password.
    """
    valid_token = create_reset_token("user@openprep.ai")
    
    with patch("src.db.auth._connect") as mock_connect:
        # Mock SQLite user existence check to return 1 (exists)
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connect.return_value.__enter__.return_value.execute.return_value = mock_cursor
        
        with patch("src.db.auth.update_password") as mock_update:
            response = client.post(
                "/api/v1/auth/reset-password",
                json={"token": valid_token, "new_password": "NewSecurePassword2@"}
            )
            
            assert response.status_code == 200
            assert "Password updated successfully" in response.json()["message"]
            mock_update.assert_called_once_with("user@openprep.ai", "NewSecurePassword2@")


def test_reset_password_invalid_token():
    """
    Assert that reset-password returns HTTP 400 when an invalid/expired token is passed.
    """
    response = client.post(
        "/api/v1/auth/reset-password",
        json={"token": "invalid_jwt_token_string", "new_password": "NewSecurePassword2@"}
    )
    
    assert response.status_code == 400
    assert "expired or is cryptographically invalid" in response.json()["detail"]


def test_reset_password_user_not_found():
    """
    Assert that reset-password returns HTTP 404 if the token is valid but the user
    does not exist in the database.
    """
    valid_token = create_reset_token("deleted_user@openprep.ai")
    
    with patch("src.db.auth._connect") as mock_connect:
        # Mock SQLite user check to return None (user not found)
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_connect.return_value.__enter__.return_value.execute.return_value = mock_cursor
        
        response = client.post(
            "/api/v1/auth/reset-password",
            json={"token": valid_token, "new_password": "NewSecurePassword2@"}
        )
        
        assert response.status_code == 404
        assert "User account context not found" in response.json()["detail"]
