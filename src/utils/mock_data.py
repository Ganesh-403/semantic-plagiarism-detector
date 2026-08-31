"""
mock_data.py
------------
Admin utility for generating fake student essays using the Faker library.

Resolves Issue #255: "Generate Dummy Data" Admin Utility.

When developers clone the repo they previously had to manually upload PDFs to
see anything happen.  This module creates 5 realistic fake essays, stores
them in corpus.db, and builds a temporary FAISS index so the app is
immediately usable with demo data.

Usage (from the Admin sidebar inside streamlit_app.py):
    from src.utils.mock_data import generate_mock_data
    result = generate_mock_data()
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Essay templates – each template is varied by Faker so every generated
# essay reads differently even across multiple calls.
# ---------------------------------------------------------------------------
_ESSAY_TOPICS = [
    "climate_change",
    "artificial_intelligence",
    "social_media",
    "space_exploration",
    "genetic_engineering",
]

_TOPIC_TITLES = {
    "climate_change": "The Impact of Climate Change on Global Ecosystems",
    "artificial_intelligence": "Artificial Intelligence: Transforming the Modern World",
    "social_media": "Social Media and Its Effects on Mental Health",
    "space_exploration": "The Future of Human Space Exploration",
    "genetic_engineering": "Ethical Implications of Genetic Engineering",
}


def _build_essay(fake, topic: str, student_name: str) -> str:
    """Generate a realistic multi-paragraph essay using Faker sentence helpers."""

    title = _TOPIC_TITLES[topic]

    # Introduction
    intro_sentences = [fake.sentence(nb_words=12) for _ in range(4)]
    intro = " ".join(intro_sentences)

    # Body paragraphs (3)
    paragraphs = []
    for _ in range(3):
        body_sentences = [fake.sentence(nb_words=14) for _ in range(5)]
        paragraphs.append(" ".join(body_sentences))

    # Conclusion
    conc_sentences = [fake.sentence(nb_words=11) for _ in range(3)]
    conclusion = " ".join(conc_sentences)

    # Compose the essay
    essay = (
        f"{title}\n"
        f"By {student_name}\n\n"
        f"Introduction\n{intro}\n\n"
        + "\n\n".join(paragraphs)
        + f"\n\nConclusion\n{conclusion}\n"
    )
    return essay


def generate_mock_essays(num_essays: int = 5) -> list[tuple[str, str, str]]:
    """Return a list of (filename, student_name, essay_text) tuples.

    Args:
        num_essays: Number of fake essays to generate (default 5).

    Returns:
        List of ``(filename, student_name, essay_text)`` tuples.

    Raises:
        ImportError: If the ``faker`` package is not installed.
    """
    try:
        from faker import Faker
    except ImportError as exc:
        raise ImportError(
            "The 'faker' package is required for mock data generation. "
            "Install it with: pip install faker"
        ) from exc

    fake = Faker()
    Faker.seed(42)  # Reproducible names/sentences across runs

    essays: list[tuple[str, str, str]] = []
    for i in range(num_essays):
        topic = _ESSAY_TOPICS[i % len(_ESSAY_TOPICS)]
        student_name = fake.name()
        safe_name = student_name.replace(" ", "_").replace(".", "")
        filename = f"mock_{safe_name}_{topic}.txt"
        text = _build_essay(fake, topic, student_name)
        essays.append((filename, student_name, text))

    return essays


def generate_mock_data(
    num_essays: int = 5,
    class_section: str = "Demo Class",
    assignment_title: str = "Demo Assignment",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    """Generate fake essays, persist them to corpus.db, and build a FAISS index.

    This is the main entry-point called by the Admin sidebar button.

    Args:
        num_essays:        Number of fake essays to create (default 5).
        class_section:     Class/section label stored with each document.
        assignment_title:  Assignment title stored with each document.
        chunk_size:        Character target length for text chunks.
        chunk_overlap:     Overlap characters between consecutive chunks.

    Returns:
        A dict with keys:
            ``essays``      – list of (filename, student_name) pairs added,
            ``skipped``     – list of filenames that already existed in the DB,
            ``faiss_ntotal``– total vectors in the rebuilt FAISS index,
            ``index_path``  – absolute path of the saved FAISS index file.

    Raises:
        ImportError: If ``faker`` is not installed.
    """

    import numpy as np

    from src.core.embedding_model import embed_documents
    from src.core.faiss_index import build_index, save_index
    from src.core.text_chunking import chunk_documents
    from src.db.corpus_db import (
        add_chunks,
        add_document,
        get_all_embeddings,
        get_chunk_registry,
        init_corpus_db,
    )

    # Ensure the DB schema is initialised
    init_corpus_db()

    essays = generate_mock_essays(num_essays)

    added: list[tuple[str, str]] = []
    skipped: list[str] = []

    raw_texts: dict[str, str] = {}

    for filename, student_name, text in essays:
        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        from src.core.document_parser import detect_text_language

        detected_lang = detect_text_language(text)

        inserted = add_document(
            filename=filename,
            file_hash=file_hash,
            class_section=class_section,
            student_name=student_name,
            assignment_title=assignment_title,
            detected_language=detected_lang,
        )

        if not inserted:
            skipped.append(filename)
            continue

        raw_texts[filename] = text
        added.append((filename, student_name))

    # ── Chunk + Embed new documents ──────────────────────────────────────────
    if raw_texts:
        chunked_docs = chunk_documents(
            raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        embeddings = embed_documents(chunked_docs)

        # Persist chunks + embeddings to the DB
        for doc_name, emb_array in embeddings.items():
            if emb_array.size == 0:
                continue
            chunks = chunked_docs.get(doc_name, [])

            # Determine the starting vector_id for this document
            # (use current max + 1 to avoid collisions)
            current_count = _get_current_vector_count()

            chunk_rows = []
            for chunk_idx, (chunk_text, vec) in enumerate(zip(chunks, emb_array)):
                vector_id = current_count + len(chunk_rows)
                chunk_rows.append((vector_id, doc_name, chunk_idx, chunk_text.text, vec))

            if chunk_rows:
                add_chunks(chunk_rows)

    # ── Rebuild FAISS index from ALL embeddings in DB ────────────────────────
    index_path = _get_index_path()
    all_embeddings = get_all_embeddings()

    if all_embeddings.size > 0:
        # Rebuild using full corpus embeddings
        registry = get_chunk_registry()

        # Group embeddings back into per-doc dicts for build_index
        chunked_for_index: dict[str, list[str]] = {}
        embeddings_for_index: dict[str, np.ndarray] = {}

        for record in registry:
            chunked_for_index.setdefault(record.doc_name, []).append(record.chunk_text)

        start = 0
        for doc_name, chunks in chunked_for_index.items():
            count = len(chunks)
            embeddings_for_index[doc_name] = all_embeddings[start : start + count]
            start += count

        faiss_index, _ = build_index(embeddings_for_index, chunked_for_index)
        save_index(faiss_index, index_path)
        ntotal = faiss_index.ntotal
    else:
        ntotal = 0

    return {
        "essays": added,
        "skipped": skipped,
        "faiss_ntotal": ntotal,
        "index_path": index_path,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _get_current_vector_count() -> int:
    """Return the current number of chunk vectors in corpus.db."""
    from src.db.corpus_db import get_embedding_count

    return get_embedding_count()


def _get_index_path() -> str:
    """Return the absolute path to corpus.index."""
    # Use the centralized FAISS index path from app_config so this module
    # agrees with app/streamlit_app.py and src/api/app.py on the index
    # location.
    from src.core.app_config import FAISS_INDEX_PATH

    return str(FAISS_INDEX_PATH)
