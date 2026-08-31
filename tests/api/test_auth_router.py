import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.app import app

client = TestClient(app)

def test_refresh_token_rotation_success():
    """
    Assert that hitting the /refresh endpoint with a valid refresh token 
    successfully yields a fresh access token.
    """
    valid_refresh_token = "valid_refresh_token_string"
    mock_access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.new_access_token"

    # Mock dependencies inside refresh_token_endpoint
    with patch("src.db.auth.is_token_revoked", return_value=False) as mock_revoked:
        with patch("src.security.jwt_utils.verify_refresh_token", return_value={"sub": "test_user", "scopes": ["read", "write"]}) as mock_verify:
            with patch("src.security.jwt_utils.create_access_token", return_value=mock_access_token) as mock_create:
                response = client.post(
                    "/api/v1/auth/refresh",
                    json={"refresh_token": valid_refresh_token}
                )
                
                # Acceptance Criteria Assertion
                assert response.status_code == 200
                assert "access_token" in response.json()
                assert response.json()["access_token"] == mock_access_token
                
                mock_revoked.assert_called_once_with(valid_refresh_token)
                mock_verify.assert_called_once_with(valid_refresh_token)


def test_refresh_token_rotation_fails_if_revoked():
    """
    Assert that passing a revoked or invalid refresh token 
    returns an HTTP 401 Unauthorized error.
    """
    revoked_refresh_token = "revoked_refresh_token_string"

    # Mock the database check to return True (token is revoked)
    with patch("src.db.auth.is_token_revoked", return_value=True) as mock_revoked:
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": revoked_refresh_token}
        )
        
        # Validation Checks
        assert response.status_code == 401
        mock_revoked.assert_called_once_with(revoked_refresh_token)


def test_refresh_token_rotation_fails_if_invalid():
    """
    Assert that passing an invalid refresh token (which raises ValueError during verification)
    returns an HTTP 401 Unauthorized error.
    """
    invalid_refresh_token = "invalid_refresh_token_string"

    # Mock to verify that ValueError raises 401
    with patch("src.db.auth.is_token_revoked", return_value=False):
        with patch("src.security.jwt_utils.verify_refresh_token", side_effect=ValueError("Token signature is invalid")):
            response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": invalid_refresh_token}
            )
            
            # Validation Checks
            assert response.status_code == 401
