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

from pathlib import Path

SOURCE = Path("src/core/document_parser.py")
TESTS = Path("tests/core/test_document_parser_process_pool_issue_1583.py")


def test_bulk_helper_accepts_worker_limit():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def extract_texts_parallel(" in source
    assert "max_workers: int | None = None" in source


def test_executor_uses_resolved_worker_bound():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def _resolve_process_pool_workers(" in source
    assert "available_cpus = os.cpu_count() or 1" in source
    assert "max_workers=worker_count" in source


def test_unit_test_verifies_process_pool_bounds():
    source = TESTS.read_text(encoding="utf-8")

    assert "test_process_pool_is_capped_by_cpu_count" in source
    assert "RecordingExecutor.recorded_max_workers == 4" in source
