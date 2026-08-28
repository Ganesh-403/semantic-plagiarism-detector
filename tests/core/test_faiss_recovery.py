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

import faiss
import numpy as np
import pytest

import src.core.faiss_index as module


def _matrix(count: int, dim: int = 384) -> np.ndarray:
    if count == 0:
        return np.empty((0, dim), dtype=np.float32)
    matrix = np.arange(count * dim, dtype=np.float32).reshape(count, dim)
    faiss.normalize_L2(matrix)
    return matrix


def _registry(count: int):
    return [module.ChunkRecord(f"{i}.pdf", 0, f"text-{i}") for i in range(count)]


def test_validate_index_accepts_matching_index():
    index = module.build_index_from_matrix(_matrix(2))
    assert module.validate_index(index, 2, expected_dimension=384)


def test_validate_index_rejects_count_mismatch():
    index = module.build_index_from_matrix(_matrix(2))
    assert not module.validate_index(index, 3)


def test_atomic_save_round_trip(tmp_path):
    path = tmp_path / "corpus.index"
    module.save_index(module.build_index_from_matrix(_matrix(2)), str(path))
    assert module.load_index(str(path)).ntotal == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_index_is_rebuilt(monkeypatch, tmp_path):
    path = tmp_path / "missing.index"
    monkeypatch.setattr("src.db.corpus_db.get_all_embeddings", lambda: _matrix(2))
    monkeypatch.setattr("src.db.corpus_db.get_chunk_registry", lambda: _registry(2))
    index, registry, recovered = module.load_or_rebuild_index(str(path))
    assert recovered is True
    assert index.ntotal == 2
    assert len(registry) == 2
    assert path.exists()


def test_corrupted_index_is_rebuilt(monkeypatch, tmp_path):
    path = tmp_path / "corrupt.index"
    path.write_bytes(b"not-a-faiss-index")
    monkeypatch.setattr("src.db.corpus_db.get_all_embeddings", lambda: _matrix(1))
    monkeypatch.setattr("src.db.corpus_db.get_chunk_registry", lambda: _registry(1))
    index, _, recovered = module.load_or_rebuild_index(str(path))
    assert recovered is True
    assert index.ntotal == 1


def test_mismatched_index_is_rebuilt(monkeypatch, tmp_path):
    path = tmp_path / "mismatch.index"
    module.save_index(module.build_index_from_matrix(_matrix(1)), str(path))
    monkeypatch.setattr("src.db.corpus_db.get_all_embeddings", lambda: _matrix(2))
    monkeypatch.setattr("src.db.corpus_db.get_chunk_registry", lambda: _registry(2))
    index, _, recovered = module.load_or_rebuild_index(str(path))
    assert recovered is True
    assert index.ntotal == 2


def test_valid_index_loads_without_recovery(monkeypatch, tmp_path):
    path = tmp_path / "valid.index"
    matrix = _matrix(2)
    module.save_index(module.build_index_from_matrix(matrix), str(path))
    monkeypatch.setattr("src.db.corpus_db.get_all_embeddings", lambda: matrix)
    monkeypatch.setattr("src.db.corpus_db.get_chunk_registry", lambda: _registry(2))
    index, _, recovered = module.load_or_rebuild_index(str(path))
    assert recovered is False
    assert index.ntotal == 2


def test_empty_corpus_returns_empty_index(monkeypatch, tmp_path):
    path = tmp_path / "empty.index"
    monkeypatch.setattr("src.db.corpus_db.get_all_embeddings", lambda: _matrix(0))
    monkeypatch.setattr("src.db.corpus_db.get_chunk_registry", lambda: [])
    index, registry, recovered = module.load_or_rebuild_index(str(path))
    assert recovered is True
    assert index.ntotal == 0
    assert registry == []


def test_registry_mismatch_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("src.db.corpus_db.get_all_embeddings", lambda: _matrix(2))
    monkeypatch.setattr("src.db.corpus_db.get_chunk_registry", lambda: _registry(1))
    with pytest.raises(ValueError, match="does not match"):
        module.load_or_rebuild_index(str(tmp_path / "index"))
