"""
tests/core/test_cross_encoder_benchmark.py
--------------------------------------------
Manual runtime/memory benchmark for the cross-encoder re-ranking stage
(Issue #3911). Skipped by default because it downloads and runs a real
sentence-transformers CrossEncoder model.

Run explicitly with:
    RUN_CROSS_ENCODER_BENCHMARK=1 pytest tests/core/test_cross_encoder_benchmark.py -s
"""

from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from src.core.similarity import clear_cross_encoder_cache, flag_plagiarism

RUN_BENCHMARK = os.getenv("RUN_CROSS_ENCODER_BENCHMARK", "").lower() in ("1", "true")


class _Chunk:
    def __init__(self, text: str):
        self.text = text


def _synthetic_corpus(n_docs: int = 60, dim: int = 384, seed: int = 42):
    rng = np.random.default_rng(seed)
    doc_names = [f"doc_{i}" for i in range(n_docs)]

    raw = rng.normal(size=(n_docs, dim)).astype("float32")
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    embeddings = {name: raw[i : i + 1] for i, name in enumerate(doc_names)}
    chunked_docs = {
        name: [_Chunk(f"Synthetic paragraph text for {name}.")] for name in doc_names
    }
    sim = np.clip(raw @ raw.T, 0.0, 1.0)
    sim_df = pd.DataFrame(sim, index=doc_names, columns=doc_names)
    return sim_df, chunked_docs, embeddings


@pytest.mark.skipif(
    not RUN_BENCHMARK,
    reason=(
        "Manual benchmark - downloads a real CrossEncoder model. Run with "
        "RUN_CROSS_ENCODER_BENCHMARK=1 to measure runtime/memory impact."
    ),
)
def test_cross_encoder_reranking_runtime_and_memory_vs_retrieval_only():
    """Compares retrieval-only flagging against cross-encoder re-ranked
    flagging on a synthetic corpus, reporting wall-clock time and peak
    memory for both stages."""
    clear_cross_encoder_cache()
    sim_df, chunked_docs, embeddings = _synthetic_corpus()

    tracemalloc.start()
    t0 = time.perf_counter()
    flag_plagiarism(
        sim_df,
        threshold=0.0,
        chunked_docs=chunked_docs,
        embeddings=embeddings,
        use_cross_encoder=False,
    )
    retrieval_only_time = time.perf_counter() - t0
    _, retrieval_only_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    flag_plagiarism(
        sim_df,
        threshold=0.0,
        chunked_docs=chunked_docs,
        embeddings=embeddings,
        use_cross_encoder=True,
        cross_encoder_top_k=50,
    )
    reranked_time = time.perf_counter() - t0
    _, reranked_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nRetrieval-only: {retrieval_only_time:.3f}s, peak {retrieval_only_peak / 1e6:.2f} MB")
    print(f"Cross-encoder re-ranked: {reranked_time:.3f}s, peak {reranked_peak / 1e6:.2f} MB")

    assert retrieval_only_time >= 0
    assert reranked_time >= 0