import numpy as np
import pandas as pd
import pytest

from src.core.processing import run_full_pipeline


@pytest.fixture
def sample_documents():
    return {
        "doc1.txt": b"""Artificial intelligence is transforming education.
Students use AI for personalized learning.
Machine learning improves teaching.""",

        "doc2.txt": b"""Artificial intelligence is transforming education.
Students use AI for personalized learning.
Deep learning also improves education.""",

        "doc3.txt": b"""Python is a popular programming language.
It is widely used for web development and data science.""",

        "doc4.txt": b"""Cloud computing provides scalable infrastructure.
Organizations use cloud services to deploy applications.""",

        "doc5.txt": b"""Cybersecurity protects systems from attacks.
Encryption improves information security.""",
    }


def fake_embed_documents(chunked_docs, batch_size=None):
    """
    Return deterministic fake embeddings.

    Every chunk receives a 384-dimensional vector so downstream
    similarity computation behaves normally.
    """
    embeddings = {}

    rng = np.random.default_rng(seed=42)

    for doc, chunks in chunked_docs.items():
        vectors = []

        for chunk in chunks:
            vector = rng.random(384)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)

        embeddings[doc] = np.vstack(vectors) if vectors else np.empty((0, 384))

    return embeddings


def fake_build_index(embeddings, chunked_docs):
    class DummyIndex:
        ntotal = sum(len(v) for v in embeddings.values())

    registry = []

    return DummyIndex(), registry


def fake_ai_detector(chunked_docs):
    return {
        name: {
            "probability": 0.10,
            "label": "Human",
        }
        for name in chunked_docs
    }


def test_run_full_pipeline(
    sample_documents,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.core.processing.embed_documents",
        fake_embed_documents,
    )

    monkeypatch.setattr(
        "src.core.processing.build_index",
        fake_build_index,
    )

    monkeypatch.setattr(
        "src.core.processing.detect_documents_ai_probability",
        fake_ai_detector,
    )

    (
        raw_texts,
        chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
        flags,
    ) = run_full_pipeline(sample_documents)

    # ---------- Raw extraction ----------

    assert len(raw_texts) == 5
    assert set(raw_texts.keys()) == set(sample_documents.keys())

    # ---------- Chunking ----------

    assert len(chunked_docs) == 5

    for chunks in chunked_docs.values():
        assert len(chunks) >= 1

    # ---------- Embeddings ----------

    assert len(embeddings) == 5

    for emb in embeddings.values():
        assert emb.shape[1] == 384

    # ---------- Similarity matrix ----------

    assert isinstance(sim_df, pd.DataFrame)
    assert sim_df.shape == (5, 5)

    np.testing.assert_allclose(
        np.diag(sim_df.values),
        np.ones(5),
        atol=1e-6,
    )

    # ---------- Chunk similarity ----------

    assert isinstance(chunk_sim_df, pd.DataFrame)
    assert chunk_sim_df.shape == (5, 5)

    # ---------- FAISS ----------

    assert hasattr(faiss_index, "ntotal")
    assert isinstance(registry, list)

    # ---------- AI detector ----------

    assert len(ai_probabilities) == 5

    # ---------- Flags ----------

    assert isinstance(flags, list)
