"""
Unit tests verifying package initialization and symbol exports of src.core (Issue #1711).
"""

import sys
import unittest
from typing import Callable


class TestCoreImports(unittest.TestCase):
    """Test suite for core package exports and initialization safety."""

    def test_import_src_core_succeeds(self):
        """Verify that src.core can be imported without raising ImportError."""
        import src.core

        self.assertIsNotNone(src.core)

    def test_with_sqlite_retry_exported(self):
        """Verify with_sqlite_retry is properly exported from src.core."""
        from src.core import with_sqlite_retry

        self.assertTrue(callable(with_sqlite_retry))

    def test_all_exports_present(self):
        """Verify __all__ in src.core contains expected symbols."""
        import src.core

        for symbol in src.core.__all__:
            self.assertTrue(
                hasattr(src.core, symbol),
                f"Exported symbol '{symbol}' not found in src.core",
            )


if __name__ == "__main__":
    unittest.main()
