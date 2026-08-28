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

"""Tests for document parser MAX_BATCH_SIZE configurable via PARSER_MAX_BATCH_SIZE (Issue #2708)."""

import importlib

import pytest


def test_max_batch_size_default(monkeypatch):
    """Test MAX_BATCH_SIZE defaults to 50 when PARSER_MAX_BATCH_SIZE is not set."""
    monkeypatch.delenv("PARSER_MAX_BATCH_SIZE", raising=False)
    import src.core.document_parser as dp

    importlib.reload(dp)
    assert dp.MAX_BATCH_SIZE == 50

    # 50 files should not raise
    dp.check_batch_rate_limit(50)

    # 51 files should raise ValueError
    with pytest.raises(ValueError, match="50"):
        dp.check_batch_rate_limit(51)


def test_max_batch_size_env_override(monkeypatch):
    """Test MAX_BATCH_SIZE is configurable via PARSER_MAX_BATCH_SIZE env var."""
    monkeypatch.setenv("PARSER_MAX_BATCH_SIZE", "150")
    import src.core.document_parser as dp

    importlib.reload(dp)
    assert dp.MAX_BATCH_SIZE == 150

    # 150 files should pass
    dp.check_batch_rate_limit(150)

    # 151 files should raise ValueError
    with pytest.raises(ValueError, match="150"):
        dp.check_batch_rate_limit(151)


def test_max_batch_size_invalid_env_fallback(monkeypatch):
    """Test MAX_BATCH_SIZE gracefully falls back to 50 on invalid or non-positive env values."""
    for invalid_val in ["invalid_str", "-10", "0", "   "]:
        monkeypatch.setenv("PARSER_MAX_BATCH_SIZE", invalid_val)
        import src.core.document_parser as dp

        importlib.reload(dp)
        assert dp.MAX_BATCH_SIZE == 50
