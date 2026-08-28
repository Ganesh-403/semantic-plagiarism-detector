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

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from src.core.embeddings import generate_embeddings


class DeterministicModel:
    """Small thread-safe model stub with deterministic rows."""

    def __init__(self):
        self.calls = []
        self.thread_names = set()
        self._lock = threading.Lock()

    def encode(self, texts, convert_to_numpy=True):
        assert convert_to_numpy is True
        with self._lock:
            self.calls.append(list(texts))
            self.thread_names.add(threading.current_thread().name)

        # Encourage overlap so the test exercises executor workers.
        time.sleep(0.01)
        return np.asarray(
            [
                [
                    float(int(text.split("-")[-1])),
                    float(len(text)),
                    1.0,
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_multithreaded_dimensions_match_single_threaded():
    texts = [f"chunk-{index}" for index in range(12)]
    model = DeterministicModel()

    with patch("src.core.embeddings._model", model):
        single = generate_embeddings(
            texts,
            num_threads=1,
        )
        multi = generate_embeddings(
            texts,
            num_threads=4,
        )

    assert single.shape == multi.shape == (12, 3)
    np.testing.assert_array_equal(single, multi)


def test_multithreaded_output_preserves_input_order():
    texts = [f"chunk-{index}" for index in range(9)]
    model = DeterministicModel()

    with patch("src.core.embeddings._model", model):
        result = generate_embeddings(
            texts,
            num_threads=3,
        )

    assert result[:, 0].tolist() == list(range(9))


def test_worker_count_is_capped_by_input_count():
    texts = ["chunk-0", "chunk-1"]
    model = DeterministicModel()

    with patch("src.core.embeddings._model", model):
        result = generate_embeddings(
            texts,
            num_threads=20,
        )

    assert result.shape == (2, 3)
    assert len(model.calls) == 2


def test_single_thread_path_avoids_executor():
    model = DeterministicModel()

    with patch("src.core.embeddings._model", model):
        with patch("src.core.embeddings.ThreadPoolExecutor") as executor:
            result = generate_embeddings(
                ["chunk-0", "chunk-1"],
                num_threads=1,
            )

    executor.assert_not_called()
    assert result.shape == (2, 3)
    assert len(model.calls) == 1


def test_default_uses_multiple_worker_batches():
    texts = [f"chunk-{index}" for index in range(8)]
    model = DeterministicModel()

    with patch("src.core.embeddings._model", model):
        result = generate_embeddings(texts)

    assert result.shape == (8, 3)
    assert len(model.calls) == 4
    assert all(len(batch) == 2 for batch in model.calls)


def test_empty_input_returns_stable_empty_matrix():
    result = generate_embeddings([], num_threads=4)

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 0)
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    "num_threads",
    [0, -1],
)
def test_non_positive_thread_count_is_rejected(num_threads):
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        generate_embeddings(
            ["chunk-0"],
            num_threads=num_threads,
        )


@pytest.mark.parametrize(
    "num_threads",
    [1.5, "4", None, True],
)
def test_non_integer_thread_count_is_rejected(num_threads):
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        generate_embeddings(
            ["chunk-0"],
            num_threads=num_threads,
        )


def test_non_string_chunk_is_rejected():
    with pytest.raises(
        TypeError,
        match="must be strings",
    ):
        generate_embeddings(
            ["valid", 123],
            num_threads=2,
        )


def test_worker_exception_is_propagated():
    class FailingModel:
        def encode(self, texts, convert_to_numpy=True):
            if "explode" in texts:
                raise RuntimeError("model failure")
            return np.ones(
                (len(texts), 3),
                dtype=np.float32,
            )

    with patch(
        "src.core.embeddings._model",
        FailingModel(),
    ):
        with pytest.raises(
            RuntimeError,
            match="model failure",
        ):
            generate_embeddings(
                [
                    "chunk-0",
                    "chunk-1",
                    "explode",
                    "chunk-3",
                ],
                num_threads=2,
            )


def test_incompatible_worker_dimensions_are_rejected():
    class InconsistentModel:
        def encode(self, texts, convert_to_numpy=True):
            dimension = 2 if texts[0] == "first" else 3
            return np.ones(
                (len(texts), dimension),
                dtype=np.float32,
            )

    with patch(
        "src.core.embeddings._model",
        InconsistentModel(),
    ):
        with pytest.raises(
            RuntimeError,
            match="incompatible output shapes",
        ):
            generate_embeddings(
                ["first", "second"],
                num_threads=2,
            )


def test_incorrect_worker_row_count_is_rejected():
    class MissingRowModel:
        def encode(self, texts, convert_to_numpy=True):
            rows = max(0, len(texts) - 1)
            return np.ones(
                (rows, 3),
                dtype=np.float32,
            )

    with patch(
        "src.core.embeddings._model",
        MissingRowModel(),
    ):
        with pytest.raises(
            RuntimeError,
            match="unexpected row count",
        ):
            generate_embeddings(
                [
                    "chunk-0",
                    "chunk-1",
                    "chunk-2",
                    "chunk-3",
                ],
                num_threads=2,
            )
