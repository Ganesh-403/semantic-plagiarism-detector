# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
test_absent_api_bearer_tokens_mapping_issue_2608.py
---------------------------------------------------
Unit test suite for Issue #2608:
Validates that when API_BEARER_TOKENS_MAPPING is absent or empty,
get_valid_tokens() returns an empty dict and logs an INFO message:
"No static API tokens configured. Relying solely on JWT auth."
"""

import logging
import os
from unittest.mock import patch

from src.api.middleware import get_valid_tokens


def test_absent_api_bearer_tokens_mapping_logs_info(caplog):
    """Verify info message is logged when API_BEARER_TOKENS_MAPPING is empty or unconfigured."""
    get_valid_tokens.cache_clear()

    with patch.dict(os.environ, {}, clear=True), caplog.at_level(logging.INFO):
        tokens = get_valid_tokens()
        assert tokens == {}
        assert (
            "No static API tokens configured. Relying solely on JWT auth."
            in caplog.text
        )

    get_valid_tokens.cache_clear()
