"""
Unit tests for SSRF protector module.
"""

import pytest
import requests
from unittest.mock import Mock, patch
from src.security.ssrf_protector import (
    SSRFProtector,
    SSRFSecurityException,
    validate_url
)


class TestSSRFProtector:
    """Test cases for SSRFProtector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.protector = SSRFProtector()
    
    def test_validate_valid_url(self):
        """Test validation of valid URLs."""
        assert self.protector.validate_url("https://example.com") is True
        assert self.protector.validate_url("http://google.com") is True
    
    def test_validate_invalid_url(self):
        """Test validation of invalid URLs."""
        with pytest.raises(SSRFSecurityException):
            self.protector.validate_url("")
        
        with pytest.raises(SSRFSecurityException):
            self.protector.validate_url("not-a-url")
        
        with pytest.raises(SSRFSecurityException):
            self.protector.validate_url("ftp://example.com")
    
    def test_validate_blocked_ip(self):
        """Test validation of blocked IP addresses."""
        with pytest.raises(SSRFSecurityException, match="blocked IP"):
            self.protector.validate_url("http://127.0.0.1")
        
        with pytest.raises(SSRFSecurityException, match="blocked IP"):
            self.protector.validate_url("http://localhost")
    
    def test_handle_redirect_response_missing_location(self):
        """Test handling redirect response with missing Location header."""
        # Create a mock response with redirect status but no Location header
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {}  # No Location header
        
        with pytest.raises(SSRFSecurityException, match="missing or empty Location header"):
            self.protector.handle_redirect_response(mock_response)
    
    def test_handle_redirect_response_empty_location(self):
        """Test handling redirect response with empty Location header."""
        mock_response = Mock()
        mock_response.status_code = 301
        mock_response.headers = {"Location": ""}  # Empty Location header
        
        with pytest.raises(SSRFSecurityException, match="missing or empty Location header"):
            self.protector.handle_redirect_response(mock_response)
    
    def test_handle_redirect_response_valid_location(self):
        """Test handling redirect response with valid Location header."""
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {"Location": "https://example.com/redirect"}
        
        with patch.object(self.protector, 'validate_url', return_value=True):
            result = self.protector.handle_redirect_response(mock_response)
            assert result == "https://example.com/redirect"
    
    def test_handle_redirect_response_non_redirect_status(self):
        """Test handling response with non-redirect status code."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with pytest.raises(SSRFSecurityException, match="not a redirect"):
            self.protector.handle_redirect_response(mock_response)
    
    def test_safe_request_with_redirect(self):
        """Test safe request handling redirects."""
        # Mock the request and response
        mock_response = Mock()
        mock_response.status_code = 302
        mock_response.headers = {"Location": "https://example.com/redirect"}
        
        with patch('requests.request', return_value=mock_response):
            with patch.object(self.protector, 'validate_url', return_value=True):
                result = self.protector.safe_request("https://example.com", follow_redirects=False)
                assert result == mock_response
    
    def test_allowed_domains_filter(self):
        """Test allowed domains filtering."""
        protector = SSRFProtector(allowed_domains=["example.com", "api.example.org"])
        
        assert protector.validate_url("https://example.com/path") is True
        assert protector.validate_url("https://api.example.org") is True
        
        with pytest.raises(SSRFSecurityException, match="not in allowed list"):
            protector.validate_url("https://evil.com")
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('src.security.ssrf_protector.get_protector') as mock_get:
            mock_protector = Mock()
            mock_get.return_value = mock_protector
            mock_protector.validate_url.return_value = True
            
            assert validate_url("https://example.com") is True
            mock_protector.validate_url.assert_called_with("https://example.com")
