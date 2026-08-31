"""
test_absent_api_bearer_tokens_mapping_issue_2608.py
---------------------------------------------------
Unit test suite for Issue #2608:
Validates that when API_BEARER_TOKENS_MAPPING is absent or empty,
get_valid_tokens() returns an empty dict and logs an INFO message:
"No static API tokens configured. Relying solely on JWT auth."
"""

import os
import logging
from unittest.mock import patch

from src.api.middleware import get_valid_tokens


def test_absent_api_bearer_tokens_mapping_logs_info(caplog):
    """Verify info message is logged when API_BEARER_TOKENS_MAPPING is empty or unconfigured."""
    get_valid_tokens.cache_clear()

    with patch.dict(os.environ, {}, clear=True), caplog.at_level(logging.INFO):
        tokens = get_valid_tokens()
        assert tokens == {}
        assert "No static API tokens configured. Relying solely on JWT auth." in caplog.text

    get_valid_tokens.cache_clear()
