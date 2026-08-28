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

SIMILARITY_PATH = Path("src/core/similarity.py")
TEST_PATH = Path("tests/core/test_similarity.py")


def test_required_function_exists_with_annotation():
    source = SIMILARITY_PATH.read_text(encoding="utf-8")

    assert "def manhattan_similarity(" in source
    assert "vec_a: np.ndarray" in source
    assert "vec_b: np.ndarray" in source
    assert ") -> float:" in source


def test_similarity_uses_normalized_manhattan_formula():
    source = SIMILARITY_PATH.read_text(encoding="utf-8")

    assert "np.sum(np.abs(array_a - array_b)" in source
    assert "1.0 / (1.0 + distance)" in source
    assert "np.clip(similarity, 0.0, 1.0)" in source


def test_unit_tests_are_added_to_requested_file():
    source = TEST_PATH.read_text(encoding="utf-8")

    assert "test_manhattan_similarity_identical_vectors" in source
    assert "test_manhattan_similarity_rejects_shape_mismatch" in source
    assert "test_manhattan_similarity_remains_bounded" in source
