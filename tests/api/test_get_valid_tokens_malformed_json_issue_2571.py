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
tests/api/test_get_valid_tokens_malformed_json_issue_2571.py
------------------------------------------------------------
Unit test verifying get_valid_tokens error handling when
API_BEARER_TOKENS_MAPPING contains malformed JSON string.

Fixes / Closes Issue #2571:
Description: src/api/middleware.py handles json.JSONDecodeError, but this error path is untested.
Acceptance Criteria:
Add a unit test setting API_BEARER_TOKENS_MAPPING="[invalid json}".
Assert that get_valid_tokens() returns {} and logs an error without crashing.
"""

import logging
import os
from unittest.mock import patch

from src.api.middleware import get_valid_tokens


def test_get_valid_tokens_malformed_json_returns_empty_dict_and_logs_error(caplog):
    """Setting API_BEARER_TOKENS_MAPPING='[invalid json}' should return {} and log an error without crashing."""
    get_valid_tokens.cache_clear()

    with patch.dict(
        os.environ, {"API_BEARER_TOKENS_MAPPING": "[invalid json}"}, clear=True
    ):
        with caplog.at_level(logging.ERROR):
            result = get_valid_tokens()

    assert result == {}
    assert any(
        "Failed to parse API_BEARER_TOKENS_MAPPING as JSON" in record.message
        for record in caplog.records
    )

    get_valid_tokens.cache_clear()
