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
Integration test for multi-format batch scans (Issue #1401).

A single batch containing one PDF, one DOCX, one TXT, and one RTF document
is pushed through the full processing pipeline to verify that mixed file
formats can be parsed and analyzed together. The similarity matrix must
come back as a (4, 4) DataFrame whose diagonal is exactly 1.0 for every
document.
"""

import io

import docx
import numpy as np
import pandas as pd
import pytest
from reportlab.pdfgen import canvas

from src.core.processing import run_full_pipeline

BATCH_FILENAMES = ("report.pdf", "essay.docx", "notes.txt", "summary.rtf")


def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal in-memory PDF containing the given text."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    words = (text + " word" * 10).split()
    c.drawString(50, 150, " ".join(words))
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_docx_bytes(text: str) -> bytes:
    """Create a minimal in-memory DOCX containing the given text."""
    document = docx.Document()
    document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _make_rtf_bytes(text: str) -> bytes:
    """Create a minimal in-memory RTF containing the given text."""
    rtf_content = r"{\rtf1\ansi" r"{\fonttbl{\f0 Arial;}}" rf"\f0\fs24 {text}" r"}"
    return rtf_content.encode("utf-8")


@pytest.fixture
def multi_format_batch():
    """Return one PDF, DOCX, TXT, and RTF document as in-memory bytes."""
    report_text = (
        "Artificial intelligence is transforming modern education systems. "
        "Students use machine learning tools for personalized learning. "
        "Educators evaluate semantic similarity between academic submissions."
    )
    essay_text = (
        "This essay explores artificial intelligence in education systems. "
        "Machine learning personalizes the learning experience for students. "
        "Educators rely on semantic similarity to review academic submissions."
    )
    notes_text = (
        "Notes on artificial intelligence and modern education systems. "
        "Students leverage machine learning for personalized learning paths. "
        "Semantic similarity helps educators review academic submissions."
    )
    summary_text = (
        "Summary of artificial intelligence in education systems today. "
        "Machine learning tools personalize the learning journey for students. "
        "Semantic similarity assists educators in reviewing submissions."
    )

    return {
        "report.pdf": _make_pdf_bytes(report_text),
        "essay.docx": _make_docx_bytes(essay_text),
        "notes.txt": notes_text.encode("utf-8"),
        "summary.rtf": _make_rtf_bytes(summary_text),
    }


def fake_embed_documents(chunked_docs, batch_size=None):
    """
    Return deterministic fake embeddings.

    Every chunk receives a 384-dimensional vector so downstream similarity
    computation behaves normally without loading a real model.
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

    return DummyIndex(), []


def fake_ai_detector(chunked_docs):
    return {
        name: {
            "probability": 0.10,
            "label": "Human",
        }
        for name in chunked_docs
    }


def test_multi_format_batch_scan_pipeline(
    multi_format_batch,
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
    ) = run_full_pipeline(multi_format_batch)

    # ---------- Raw extraction: all four formats must be parsed ----------

    assert set(raw_texts.keys()) == set(BATCH_FILENAMES)
    assert len(raw_texts) == 4

    for filename in BATCH_FILENAMES:
        assert raw_texts[filename].strip(), f"{filename} parsed to empty text"

    # ---------- Chunking ----------

    assert set(chunked_docs.keys()) == set(BATCH_FILENAMES)

    for filename, chunks in chunked_docs.items():
        assert len(chunks) >= 1, f"{filename} produced no chunks"

    # ---------- Embeddings ----------

    assert set(embeddings.keys()) == set(BATCH_FILENAMES)

    for emb in embeddings.values():
        assert emb.shape[1] == 384

    # ---------- Similarity matrix ----------

    assert isinstance(sim_df, pd.DataFrame)
    assert sim_df.shape == (4, 4)
    assert list(sim_df.index) == list(BATCH_FILENAMES)
    assert list(sim_df.columns) == list(BATCH_FILENAMES)

    np.testing.assert_allclose(
        np.diag(sim_df.values),
        np.ones(4),
        atol=1e-6,
    )

    # ---------- Chunk similarity ----------

    assert isinstance(chunk_sim_df, pd.DataFrame)
    assert chunk_sim_df.shape == (4, 4)

    np.testing.assert_allclose(
        np.diag(chunk_sim_df.values),
        np.ones(4),
        atol=1e-6,
    )

    # ---------- Downstream outputs ----------

    assert hasattr(faiss_index, "ntotal")
    assert isinstance(registry, list)
    assert len(ai_probabilities) == 4
    assert isinstance(flags, list)
