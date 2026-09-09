"""Concurrent encoding preserves every input row without depending on a model download."""
import threading
from types import SimpleNamespace
import numpy as np
from src.core import embeddings


def test_embedding_generation_performance(monkeypatch):
    barrier = threading.Barrier(4)
    workers = set()
    def encode(texts, **kwargs):
        workers.add(threading.get_ident())
        barrier.wait(timeout=10)
        return np.array([[int(text), int(text) ** 2] for text in texts], dtype=np.float32)
    monkeypatch.setattr(embeddings, "_get_model", lambda: SimpleNamespace(encode=encode))
    result = embeddings.generate_embeddings([str(i) for i in range(12)], num_threads=4)
    np.testing.assert_array_equal(result, [[i, i * i] for i in range(12)])
    assert len(workers) == 4
