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

# tests/core/test_core_init.py


def test_with_sqlite_retry_accessible_via_getattr():
    """Verify with_sqlite_retry is accessible via __getattr__ at runtime."""
    from src import core

    # Should not raise AttributeError
    assert hasattr(core, "with_sqlite_retry")

    # Should be callable
    retry_decorator = core.with_sqlite_retry
    assert callable(retry_decorator)


def test_with_sqlite_retry_in_all_list():
    """Verify with_sqlite_retry is listed in __all__ for proper re-export."""
    from src import core

    assert "with_sqlite_retry" in core.__all__


def test_invalid_attribute_raises_attribute_error():
    """Verify accessing non-existent attributes raises AttributeError."""
    import pytest

    from src import core

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = core.nonexistent_function_xyz
