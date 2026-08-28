#!/usr/bin/env python3
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
scripts/seed_dev_data.py
------------------------
Populates a local development environment with sample users, documents,
embeddings, and a usable FAISS index. Idempotent and safe to run multiple times.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Temporarily allow the 'user' role for development seeding so student
# accounts can be created, as the default configuration often limits
# valid roles to 'admin,teacher'. The repository explicitly defines
# 'user' in UserRole for students.
os.environ["ALLOWED_USER_ROLES"] = "admin,teacher,user"

# Ensure src module is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.app_config import FAISS_INDEX_PATH
from src.core.document_parser import detect_text_language
from src.core.embedding_model import embed_documents
from src.core.faiss_index import build_index_from_matrix, save_index
from src.core.text_chunking import chunk_documents
from src.db.auth import add_user
from src.db.auth import init_db as init_auth_db
from src.db.corpus_db import (
    add_chunks,
    add_document,
    get_all_embeddings,
    get_document_by_hash,
    get_document_chunks_count,
    get_embedding_count,
    init_corpus_db,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DIR = _REPO_ROOT / "data" / "sample_documents"

USERS = [
    ("teacher1", "devpassword", "teacher"),
    ("teacher2", "devpassword", "teacher"),
    ("student1", "devpassword", "user"),
    ("student2", "devpassword", "user"),
    ("student3", "devpassword", "user"),
]

DOCUMENTS = [
    # Topic 1: Space Exploration (3 related documents)
    (
        "space_exploration_1.txt",
        "student1",
        "The future of human space exploration relies heavily on the continued development of reusable rocket technology and the establishment of sustainable habitats on the Moon and Mars. Recent advancements by both governmental agencies and private spaceflight companies have significantly lowered the cost per kilogram to low Earth orbit. This reduction in cost is a critical first step toward making interplanetary travel economically viable. Furthermore, in-situ resource utilization—the practice of harvesting local materials for life support and propellants—will be essential for long-duration missions.",
    ),
    (
        "space_exploration_2.txt",
        "student2",
        "Humanity's future in space depends on developing reusable rockets and building self-sustaining bases on the lunar surface and Mars. New achievements by space agencies and private aerospace firms have drastically reduced the price to launch payloads into low Earth orbit. Decreasing these launch costs represents the primary hurdle to economically feasible travel between planets. Additionally, utilizing local resources on other planets to produce water, oxygen, and fuel will be absolutely necessary for extended deep-space exploration.",
    ),
    (
        "space_exploration_3.txt",
        "student3",
        "Space exploration has always captured the human imagination. While robotic probes have mapped the outer solar system, human missions remain focused on the immediate neighborhood. The Apollo missions proved that humans could survive on the Moon, but establishing a permanent presence requires overcoming immense logistical and physiological challenges. Radiation exposure, microgravity-induced bone loss, and psychological isolation are among the primary hazards facing future astronauts on long-term missions.",
    ),
    # Topic 2: Machine Learning (3 related documents)
    (
        "machine_learning_1.txt",
        "teacher1",
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms capable of learning from and making predictions based on data. Supervised learning, where models are trained on labeled datasets, is currently the most widely deployed approach in industry. Applications range from computer vision and natural language processing to predictive maintenance in manufacturing. However, the performance of these models is heavily dependent on the quality and volume of the training data.",
    ),
    (
        "machine_learning_2.txt",
        "student1",
        "As a branch of artificial intelligence, machine learning centers around creating algorithms that can learn patterns from data to make accurate predictions. The most commonly used method in the tech industry today is supervised learning, which involves training models using datasets that already have labels. This technology powers applications like image recognition, text analysis, and automated diagnostics. Despite its success, model accuracy relies entirely on having access to high-quality, large-scale training data.",
    ),
    (
        "machine_learning_3.txt",
        "student2",
        "Deep learning, a specialized field within machine learning, utilizes artificial neural networks with multiple layers to model complex patterns in data. These deep networks are particularly adept at handling unstructured data such as images, audio, and raw text. The backpropagation algorithm, combined with stochastic gradient descent, is used to iteratively adjust the network's weights to minimize prediction error. The recent explosion in deep learning capabilities is largely attributed to the availability of massive datasets and accelerated computing hardware.",
    ),
    # Topic 3: History (2 distinct documents)
    (
        "history_rome.txt",
        "student3",
        "The fall of the Western Roman Empire was a complex process involving a multitude of factors over several centuries. Internal political instability, economic decline, and the reliance on mercenary armies severely weakened the empire's structural integrity. Simultaneously, external pressures from various migrating Germanic tribes—including the Visigoths, Vandals, and Ostrogoths—overwhelmed the Roman borders. By 476 AD, when Odoacer deposed the last Western Roman Emperor, the political authority of Rome in the West had effectively collapsed.",
    ),
    (
        "history_egypt.txt",
        "student1",
        "Ancient Egyptian civilization thrived along the Nile River for over three millennia, leaving behind a legacy of monumental architecture, intricate hieroglyphic writing, and advanced agricultural techniques. The annual flooding of the Nile deposited nutrient-rich silt, creating a fertile strip of land that sustained a booming population. The society was highly stratified, with the Pharaoh occupying the pinnacle of the social and religious hierarchy, viewed as a living deity responsible for maintaining Ma'at, the universal order and balance.",
    ),
    # Topic 4: Climate Change (2 related documents)
    (
        "climate_change_1.txt",
        "teacher2",
        "Global climate change is primarily driven by the anthropogenic emission of greenhouse gases, such as carbon dioxide and methane, which trap heat within the Earth's atmosphere. This enhanced greenhouse effect has led to rising average global temperatures, resulting in the accelerated melting of polar ice caps and a measurable increase in sea levels. Additionally, shifting weather patterns are causing more frequent and severe extreme weather events, including prolonged droughts, intense hurricanes, and devastating wildfires.",
    ),
    (
        "climate_change_2.txt",
        "student2",
        "The main driver of contemporary climate change is the human release of greenhouse gases like carbon dioxide and methane, which accumulate and trap solar radiation in the atmosphere. This intensified greenhouse effect causes global temperatures to rise, leading to the rapid thawing of glaciers and rising ocean levels. Furthermore, the changing climate disrupts traditional weather systems, triggering an increase in extreme events such as extended dry spells, powerful storms, and widespread forest fires.",
    ),
]


def seed_users():
    """Seed development users idempotently."""
    created = 0
    skipped = 0

    for username, password, role in USERS:
        try:
            add_user(username=username, password=password, role=role)
            created += 1
        except ValueError as e:
            if "already exists" in str(e):
                skipped += 1
            else:
                raise

    logger.info(f"✓ Users: {created} created / {skipped} already present")


def ensure_sample_documents():
    """Ensure the sample text files exist on disk."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, _, content in DOCUMENTS:
        filepath = SAMPLE_DIR / filename
        if not filepath.exists():
            filepath.write_text(content, encoding="utf-8")


def seed_documents():
    """Process documents through the ingestion pipeline and store in corpus.db."""
    created_docs = []
    skipped_docs = []
    recovered_docs = []
    raw_texts: dict[str, str] = {}

    for filename, student_name, _ in DOCUMENTS:
        filepath = SAMPLE_DIR / filename
        text = filepath.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Use repository API to cleanly check idempotency
        existing_filename = get_document_by_hash(file_hash)
        if existing_filename is not None:
            # Check if chunking successfully completed in a prior run
            if get_document_chunks_count(existing_filename) > 0:
                skipped_docs.append(filename)
                continue
            else:
                recovered_docs.append(existing_filename)
                filename = existing_filename

        detected_lang = detect_text_language(text)

        # If it already exists, this is a no-op that just returns the existing ID
        add_document(
            filename=filename,
            file_hash=file_hash,
            class_section="Development Setup",
            student_name=student_name,
            assignment_title="Sample Dataset",
            detected_language=detected_lang,
        )

        if filename not in recovered_docs:
            created_docs.append(filename)

        raw_texts[filename] = text

    logger.info(
        f"✓ Documents: {len(created_docs)} created / {len(recovered_docs)} recovered / {len(skipped_docs)} already present"
    )

    if raw_texts:
        try:
            # Chunking
            chunked_docs = chunk_documents(raw_texts, chunk_size=500, chunk_overlap=50)
            logger.info("✓ Chunks generated")

            # Embedding
            embeddings = embed_documents(chunked_docs)
            logger.info("✓ Embeddings generated")

            # Store chunks and vectors
            current_count = get_embedding_count()
            for doc_name, emb_array in embeddings.items():
                if getattr(emb_array, "size", 0) == 0:
                    continue
                chunks = chunked_docs.get(doc_name, [])
                chunk_rows = []
                for chunk_idx, (chunk_text, vec) in enumerate(zip(chunks, emb_array)):
                    vector_id = current_count + len(chunk_rows)
                    chunk_rows.append((vector_id, doc_name, chunk_idx, chunk_text, vec))

                if chunk_rows:
                    add_chunks(chunk_rows)
                    current_count += len(chunk_rows)
            logger.info("✓ Embeddings and chunks stored in database")
        except Exception as e:
            logger.error(f"Failed during embedding/chunking pipeline: {e}")
            sys.exit(1)
    else:
        logger.info("✓ Chunks generated (0 new)")
        logger.info("✓ Embeddings generated (0 new)")


def rebuild_faiss_index():
    """
    Rebuild the FAISS index using the repository's established pattern.
    Extracts the matrix of all embeddings directly from the database and uses
    build_index_from_matrix() to maintain exact indexing.
    """
    all_embeddings = get_all_embeddings()
    if getattr(all_embeddings, "size", 0) > 0:
        new_index = build_index_from_matrix(all_embeddings)
        save_index(new_index, str(FAISS_INDEX_PATH))
        logger.info(f"✓ FAISS index rebuilt (Total vectors: {new_index.ntotal})")
    else:
        logger.info("✓ FAISS index rebuilt (Total vectors: 0)")


def main():
    print("Development dataset seed")
    print("-" * 24)

    try:
        # Initialize Databases
        init_auth_db()
        init_corpus_db()
        logger.info("✓ Database initialized")

        # Seed Data
        seed_users()
        ensure_sample_documents()
        seed_documents()
        rebuild_faiss_index()

        print("✓ Development dataset ready")
    except Exception as e:
        logger.error(f"Failed to seed development dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
