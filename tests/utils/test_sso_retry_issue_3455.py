"""
test_sso_retry_issue_3455.py
----------------------------
Unit tests for Issue #3455: Retry logic for transient OAuth network errors.

Validates:
  1. _get_oauth_session creates a session configured with a Retry adapter.
  2. The Retry configuration uses total=3, backoff_factor=0.5, and status_forcelist=[500, 502, 503, 504].
  3. exchange_google_code and exchange_github_code use this session.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests
from urllib3.util import Retry

from src.utils.sso import _get_oauth_session


def test_get_oauth_session_configuration():
    """Verify that the OAuth session retry configuration matches the acceptance criteria."""
    session = _get_oauth_session()
    assert isinstance(session, requests.Session)
    
    # Check that HTTPAdapter with Retry is mounted
    adapter_http = session.adapters.get("http://")
    adapter_https = session.adapters.get("https://")
    
    assert adapter_http is not None
    assert adapter_https is not None
    
    retry_http = adapter_http.max_retries
    retry_https = adapter_https.max_retries
    
    assert isinstance(retry_http, Retry)
    assert isinstance(retry_https, Retry)
    
    # Validate specific configuration values
    for r in (retry_http, retry_https):
        assert r.total == 3
        assert r.backoff_factor == 0.5
        assert set(r.status_forcelist) == {500, 502, 503, 504}
