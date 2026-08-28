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
Unit tests verifying package initialization and symbol exports of src.core (Issue #1711).
"""

import unittest


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

    def test_chunk_record_exports_distinct(self):
        """Verify FaissChunkRecord and PipelineChunkRecord are distinctly exported from src.core."""
        from src.core import FaissChunkRecord, PipelineChunkRecord
        from src.core.faiss_index import ChunkRecord as FaissOriginal
        from src.core.pipeline import ChunkRecord as PipelineOriginal

        self.assertIs(FaissChunkRecord, FaissOriginal)
        self.assertIs(PipelineChunkRecord, PipelineOriginal)
        self.assertIsNot(FaissChunkRecord, PipelineChunkRecord)


if __name__ == "__main__":
    unittest.main()
