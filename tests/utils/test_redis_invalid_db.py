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
tests/utils/test_redis_invalid_db.py
------------------------------------
Unit tests for handling invalid REDIS_DB configurations gracefully (Issue #2818).
"""

import importlib
import logging


def test_invalid_redis_db_falls_back_to_zero_and_logs_warning(caplog, monkeypatch):
    """
    Verify that if REDIS_DB is set to a non-integer string like 'db1',
    ValueError is caught, a warning is logged, and REDIS_DB defaults to 0.
    """
    monkeypatch.setenv("REDIS_DB", "db1")

    with caplog.at_level(logging.WARNING):
        import src.utils.redis_cache

        importlib.reload(src.utils.redis_cache)

    assert src.utils.redis_cache.REDIS_DB == 0
    assert any(
        "Invalid REDIS_DB configuration 'db1'. Defaulting to 0." in record.message
        for record in caplog.records
    )


def test_valid_redis_db_is_parsed_correctly(monkeypatch):
    """Verify that valid integer strings for REDIS_DB are parsed without issue."""
    monkeypatch.setenv("REDIS_DB", "3")
    import src.utils.redis_cache

    importlib.reload(src.utils.redis_cache)

    assert src.utils.redis_cache.REDIS_DB == 3
