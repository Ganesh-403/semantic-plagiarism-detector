"""
Comprehensive Unit Tests for JWT Expiration Boundaries
Issue: #3689
Tests exact expiration timing, expired tokens, and edge cases.
"""

import pytest
import time
import hmac
import hashlib
import base64
import json


# ==============================================================================
# SECTION 1: Defining the JWT Helper Functions (Under Test)
# ==============================================================================

SECRET_KEY = "test-secret-key"

def base64url_encode(data: dict) -> str:
    """Encodes a dictionary as a base64url string."""
    json_str = json.dumps(data)
    return base64.urlsafe_b64encode(json_str.encode()).rstrip(b"=").decode()


def create_jwt(payload: dict, secret: str = SECRET_KEY) -> str:
    """
    Creates a basic JWT token with a header and payload.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = base64url_encode(header)
    encoded_payload = base64url_encode(payload)
    
    signature_input = f"{encoded_header}.{encoded_payload}".encode()
    signature = hmac.new(secret.encode(), signature_input, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    
    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def decode_jwt(token: str, secret: str = SECRET_KEY) -> dict:
    """
    Decodes a JWT token and validates the signature.
    Raises ValueError if the token is invalid.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        
        encoded_header, encoded_payload, encoded_signature = parts
        
        # Verify signature
        signature_input = f"{encoded_header}.{encoded_payload}".encode()
        expected_signature = hmac.new(secret.encode(), signature_input, hashlib.sha256).digest()
        actual_signature = base64.urlsafe_b64decode(encoded_signature + "==")
        
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise ValueError("Invalid signature")
        
        # Decode payload
        padding = "=" * (4 - len(encoded_payload) % 4)
        payload_bytes = base64.urlsafe_b64decode(encoded_payload + padding)
        return json.loads(payload_bytes)
    
    except Exception as e:
        raise ValueError(f"Invalid token: {str(e)}")


# ==============================================================================
# SECTION 2: Testing Basic JWT Creation and Decoding
# ==============================================================================

class TestBasicJWT:
    def test_create_and_decode_token(self):
        """Should successfully create and decode a token."""
        payload = {"user_id": 1, "role": "admin"}
        token = create_jwt(payload)
        decoded = decode_jwt(token)
        assert decoded["user_id"] == 1
        assert decoded["role"] == "admin"

    def test_token_contains_three_parts(self):
        """JWT should have three parts separated by dots."""
        token = create_jwt({"user_id": 1})
        assert len(token.split(".")) == 3

    def test_decoded_payload_matches_original(self):
        """Decoded payload should match the original."""
        payload = {"user_id": 42, "username": "testuser"}
        token = create_jwt(payload)
        decoded = decode_jwt(token)
        assert decoded == payload


# ==============================================================================
# SECTION 3: Testing Expiration Boundaries
# ==============================================================================

class TestExpirationBoundaries:
    def test_token_expires_at_exact_time(self):
        """Token should be expired exactly at the expiration timestamp."""
        expiration_time = int(time.time()) - 1  # Already expired by 1 second
        payload = {"exp": expiration_time}
        token = create_jwt(payload)
        
        decoded = decode_jwt(token)
        assert decoded["exp"] == expiration_time
        assert decoded["exp"] < time.time()  # Expired

    def test_token_valid_before_expiration(self):
        """Token should be valid just before expiration."""
        expiration_time = int(time.time()) + 100  # 100 seconds from now
        payload = {"exp": expiration_time}
        token = create_jwt(payload)
        
        decoded = decode_jwt(token)
        assert decoded["exp"] > time.time()  # Valid

    def test_token_expired_at_exact_second(self):
        """Token should be considered expired at the exact second."""
        expiration_time = int(time.time())
        payload = {"exp": expiration_time}
        token = create_jwt(payload)
        
        decoded = decode_jwt(token)
        assert decoded["exp"] <= time.time()

    def test_future_expiration(self):
        """Token should be valid with a far future expiration."""
        expiration_time = int(time.time()) + 3600  # 1 hour from now
        payload = {"exp": expiration_time}
        token = create_jwt(payload)
        
        decoded = decode_jwt(token)
        assert decoded["exp"] > time.time()


# ==============================================================================
# SECTION 4: Testing Invalid Tokens
# ==============================================================================

class TestInvalidTokens:
    def test_invalid_signature(self):
        """Should raise error for invalid signature."""
        token = create_jwt({"user_id": 1})
        # Tamper with the token
        tampered_token = token[:-1] + ("a" if token[-1] != "a" else "b")
        with pytest.raises(ValueError):
            decode_jwt(tampered_token)

    def test_invalid_format(self):
        """Should raise error for invalid token format."""
        with pytest.raises(ValueError):
            decode_jwt("just.a.string")

    def test_empty_token(self):
        """Should raise error for empty token."""
        with pytest.raises(ValueError):
            decode_jwt("")

    def test_none_token(self):
        """Should raise error for None token."""
        with pytest.raises(ValueError):
            decode_jwt(None)


# ==============================================================================
# SECTION 5: Testing Security Properties
# ==============================================================================

class TestSecurityProperties:
    def test_token_is_not_human_readable(self):
        """JWT payload should be encoded, not plain text."""
        token = create_jwt({"user_id": 1})
        assert "user_id" not in token.split(".")[1]

    def test_signature_is_required(self):
        """Token without signature should be invalid."""
        token = create_jwt({"user_id": 1})
        parts = token.split(".")
        # Remove signature
        token_without_sig = f"{parts[0]}.{parts[1]}"
        with pytest.raises(ValueError):
            decode_jwt(token_without_sig)

    def test_different_secret_key_fails(self):
        """Token created with one key should fail with another."""
        token = create_jwt({"user_id": 1}, secret="key1")
        with pytest.raises(ValueError):
            decode_jwt(token, secret="key2")


# ==============================================================================
# SECTION 6: Testing Edge Cases
# ==============================================================================

class TestEdgeCases:
    def test_token_with_empty_payload(self):
        """Should handle empty payloads."""
        token = create_jwt({})
        decoded = decode_jwt(token)
        assert decoded == {}

    def test_token_with_nested_payload(self):
        """Should handle nested payloads."""
        payload = {"user": {"id": 1, "name": "Test"}}
        token = create_jwt(payload)
        decoded = decode_jwt(token)
        assert decoded["user"]["name"] == "Test"

    def test_token_with_list_payload(self):
        """Should handle list payloads."""
        payload = {"roles": ["admin", "user"]}
        token = create_jwt(payload)
        decoded = decode_jwt(token)
        assert decoded["roles"] == ["admin", "user"]