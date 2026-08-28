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

EMBEDDINGS_PATH = Path("src/core/embeddings.py")
TEST_PATH = Path("tests/core/test_embeddings_multithreaded_issue_1261.py")


def test_generate_embeddings_has_required_parameter():
    source = EMBEDDINGS_PATH.read_text(encoding="utf-8")

    assert "def generate_embeddings(" in source
    assert "num_threads: int = 4" in source


def test_thread_pool_and_batch_processing_are_used():
    source = EMBEDDINGS_PATH.read_text(encoding="utf-8")

    assert "ThreadPoolExecutor(" in source
    assert "executor.map(_encode_batch, batches)" in source
    assert "np.concatenate(encoded_batches" in source


def test_dimension_equivalence_test_exists():
    source = TEST_PATH.read_text(encoding="utf-8")

    assert "test_multithreaded_dimensions_match_single_threaded" in source
    assert "single.shape == multi.shape" in source
