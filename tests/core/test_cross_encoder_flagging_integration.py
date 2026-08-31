"""
tests/core/test_cross_encoder_flagging_integration.py
-------------------------------------------------------
Evaluation tests comparing retrieval-only (FAISS/bi-encoder) results against
cross-encoder re-ranked results produced by flag_plagiarism() (Issue #3911).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.core.similarity import clear_cross_encoder_cache, flag_plagiarism


class _Chunk:
    """Minimal chunk stand-in exposing a `.text` attribute."""

    def __init__(self, text: str):
        self.text = text


class _ReversingDummyModel:
    """Fake CrossEncoder that scores pairs in a fixed, given order."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, sentence_pairs, batch_size=32):
        return np.array(self.scores[: len(sentence_pairs)])


def _build_fixture():
    sim_df = pd.DataFrame(
        [
            [1.00, 0.95, 0.65],
            [0.95, 1.00, 0.30],
            [0.65, 0.30, 1.00],
        ],
        index=["doc_a", "doc_b", "doc_c"],
        columns=["doc_a", "doc_b", "doc_c"],
    )
    chunked_docs = {
        "doc_a": [_Chunk("The mitochondria is the powerhouse of the cell.")],
        "doc_b": [_Chunk("The mitochondria is the powerhouse of the cell.")],
        "doc_c": [_Chunk("Cells rely on mitochondria to produce usable energy.")],
    }
    embeddings = {
        "doc_a": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "doc_b": np.array([[0.9, 0.1, 0.0, 0.0]]),
        "doc_c": np.array([[0.5, 0.5, 0.0, 0.0]]),
    }
    return sim_df, chunked_docs, embeddings


def setup_function():
    clear_cross_encoder_cache()


def teardown_function():
    clear_cross_encoder_cache()


def test_retrieval_only_results_are_unaffected_by_default_flag():
    """use_cross_encoder=False keeps the original FAISS/bi-encoder-only behaviour."""
    sim_df, chunked_docs, embeddings = _build_fixture()

    flags = flag_plagiarism(
        sim_df,
        chunked_docs=chunked_docs,
        embeddings=embeddings,
        use_cross_encoder=False,
    )

    assert [f["doc_a"] for f in flags] == ["doc_a", "doc_a"]
    assert [f["doc_b"] for f in flags] == ["doc_b", "doc_c"]
    assert [f["similarity"] for f in flags] == [0.95, 0.65]
    for f in flags:
        assert "cross_encoder_score" not in f
        assert "semantic_score" not in f


def test_reranked_results_can_change_order_and_preserve_original_score():
    """Cross-encoder re-ranking can promote a lower bi-encoder-scored pair while
    the original score is preserved as `semantic_score` for diagnostics."""
    sim_df, chunked_docs, embeddings = _build_fixture()

    # (doc_a, doc_b) is evaluated first (higher bi-encoder score, 0.95), then
    # (doc_a, doc_c) (0.65). Give the cross-encoder the opposite preference.
    mock_model = _ReversingDummyModel(scores=[-2.0, 2.0])

    with patch("src.core.similarity._get_cross_encoder", return_value=mock_model):
        flags = flag_plagiarism(
            sim_df,
            chunked_docs=chunked_docs,
            embeddings=embeddings,
            use_cross_encoder=True,
            cross_encoder_top_k=10,
        )

    # Cross-encoder re-ranked (doc_a, doc_c) above (doc_a, doc_b).
    assert [(f["doc_a"], f["doc_b"]) for f in flags] == [
        ("doc_a", "doc_c"),
        ("doc_a", "doc_b"),
    ]

    reranked_top = flags[0]
    assert reranked_top["semantic_score"] == 0.65  # original FAISS/bi-encoder score
    assert reranked_top["cross_encoder_score"] > flags[1]["cross_encoder_score"]


def test_cross_encoder_top_k_bounds_reranking_scope():
    """Only the highest-scoring `cross_encoder_top_k` candidates get re-scored;
    the rest keep their original similarity untouched (evidence preserved)."""
    sim_df, chunked_docs, embeddings = _build_fixture()
    mock_model = _ReversingDummyModel(scores=[0.5])

    with patch("src.core.similarity._get_cross_encoder", return_value=mock_model):
        flags = flag_plagiarism(
            sim_df,
            chunked_docs=chunked_docs,
            embeddings=embeddings,
            use_cross_encoder=True,
            cross_encoder_top_k=1,
        )

    reranked = [f for f in flags if "cross_encoder_score" in f]
    untouched = [f for f in flags if "cross_encoder_score" not in f]

    assert len(reranked) == 1
    assert reranked[0]["doc_a"] == "doc_a" and reranked[0]["doc_b"] == "doc_b"
    assert len(untouched) == 1
    assert untouched[0]["similarity"] == 0.65


def test_cross_encoder_unavailable_falls_back_to_bi_encoder_ranking():
    """If the cross-encoder model can't be loaded, flagging still works and
    the retrieval-only ranking/scores are preserved."""
    sim_df, chunked_docs, embeddings = _build_fixture()

    with patch("src.core.similarity._get_cross_encoder", return_value=None):
        flags = flag_plagiarism(
            sim_df,
            chunked_docs=chunked_docs,
            embeddings=embeddings,
            use_cross_encoder=True,
        )

    assert [f["similarity"] for f in flags] == [0.95, 0.65]
    for f in flags:
        assert f.get("cross_encoder_score", f["similarity"]) == f["similarity"]