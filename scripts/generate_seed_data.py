"""
Generate deterministic seed databases and a FAISS index.

The optional ``--target-similarity`` flag controls the cosine
similarity of the seeded Alice/Bob plagiarism pair.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

SEED_DIR = os.path.join(ROOT_DIR, "tests", "dummy_data")
os.makedirs(SEED_DIR, exist_ok=True)

DEFAULT_TARGET_SIMILARITY = 0.95
BACKGROUND_SIMILARITY = 0.15
EMBEDDING_DIMENSION = 384
RANDOM_SEED = 42

MOCK_NAMES = [
    f"Student_{i:04d}" for i in range(1, 53)
]


@dataclass
class SeedConfig:
    target_similarity: float
    verbose: bool
    seed_dir: str = SEED_DIR


def parse_target_similarity(value: str) -> float:
    raw_value = value.strip()
    is_percentage = raw_value.endswith("%")
    numeric_text = raw_value[:-1].strip() if is_percentage else raw_value
    try:
        parsed_value = float(numeric_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target similarity must be a number such as 0.85, 85, or 85%"
        ) from exc
    if is_percentage or parsed_value > 1.0:
        parsed_value /= 100.0
    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError(
            "target similarity must be between 0 and 1 (or between 0% and 100%)"
        )
    return parsed_value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic seed databases and a FAISS index for local testing."
    )
    parser.add_argument(
        "--target-similarity",
        type=parse_target_similarity,
        default=DEFAULT_TARGET_SIMILARITY,
        metavar="VALUE",
        help=(
            "Cosine similarity for the flagged Alice/Bob pair. "
            "Accepts 0.85, 85, or 85%% "
            f"(default: {DEFAULT_TARGET_SIMILARITY})."
        ),
    )
    parser.add_argument(
        "--seed-dir",
        default=SEED_DIR,
        metavar="PATH",
        help=(
            "Directory where users.db, corpus.db, and corpus.index are "
            "generated (default: tests/dummy_data)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed similarity validation output."
    )
    return parser


def parse_args(argv: list[str] | None = None) -> SeedConfig:
    namespace = build_argument_parser().parse_args(argv)
    return SeedConfig(
        target_similarity=namespace.target_similarity,
        verbose=namespace.verbose,
        seed_dir=os.path.abspath(os.path.expanduser(namespace.seed_dir)),
    )


def generate_similar_vector(
    base_vector: np.ndarray,
    target_similarity: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    noise = random_generator.standard_normal(base_vector.shape[0])
    noise -= np.dot(noise, base_vector) * base_vector
    noise_norm = np.linalg.norm(noise)
    if noise_norm < 1e-12:
        raise RuntimeError("Unable to generate an orthogonal noise vector.")
    noise /= noise_norm
    generated = target_similarity * base_vector + np.sqrt(1 - target_similarity**2) * noise
    generated /= np.linalg.norm(generated)
    return generated


def calculate_cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Cosine similarity requires non-zero vectors.")
    return float(np.dot(first, second) / denominator)


def validate_target_similarity(
    base_vector: np.ndarray,
    generated_vector: np.ndarray,
    target_similarity: float,
    *,
    tolerance: float = 1e-6,
) -> float:
    actual_similarity = calculate_cosine_similarity(base_vector, generated_vector)
    if not np.isclose(actual_similarity, target_similarity, atol=tolerance, rtol=0.0):
        raise ValueError(
            "Generated similarity does not match target: "
            f"expected {target_similarity:.6f}, got {actual_similarity:.6f}"
        )
    return actual_similarity


def _clean_seed_files(seed_dir: str = SEED_DIR) -> None:
    """Remove prior generated seed artifacts from ``seed_dir``."""
    os.makedirs(seed_dir, exist_ok=True)
    for filename in ("users.db", "corpus.db", "corpus.index"):
        path = os.path.join(seed_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            os.remove(path)
            print(f"Removed old seed {filename}")
        except OSError as error:
            print(f"Warning: Could not remove old seed {filename} ({error})")


def main(argv: list[str] | None = None) -> None:
    import importlib

    _init_py = os.path.join(ROOT_DIR, "src", "__init__.py")
    with open(_init_py, "r", encoding="utf-8") as _f:
        _backup = _f.read()

    try:
        with open(_init_py, "w", encoding="utf-8") as _f:
            _f.write("from __future__ import annotations\n")

        importlib.invalidate_caches()

        if "src" in sys.modules:
            del sys.modules["src"]
        for _key in list(sys.modules):
            if _key.startswith("src."):
                del sys.modules[_key]

        from src.core.faiss_index import build_index_from_matrix, save_index  # noqa: F811
        from src.db.auth import configure_db_path as configure_auth_db_path, add_user  # noqa: F811
        from src.db.auth import init_db as init_auth_db  # noqa: F811
        from src.db.corpus_db import (  # noqa: F811
            configure_db_path as configure_corpus_db_path,
            add_chunks,
            add_document,
            init_corpus_db,
        )
        from src.db.incidents import sync_flagged_incidents  # noqa: F811

        config = parse_args(argv)

        seed_dir = config.seed_dir
        os.makedirs(seed_dir, exist_ok=True)

        auth_db_path = os.path.join(seed_dir, "users.db")
        corpus_db_path = os.path.join(seed_dir, "corpus.db")
        configure_auth_db_path(auth_db_path)
        configure_corpus_db_path(corpus_db_path)
        _clean_seed_files(seed_dir)

        print("Initializing databases...")
        init_auth_db()
        add_user("teacher", "teacher123", "teacher")
        print("Auth DB initialized and seeded.")

        init_corpus_db()
        print("Corpus DB initialized.")

        text_alice = (
            "Artificial intelligence (AI) is intelligence "
            "demonstrated by machines, in contrast to the natural "
            "intelligence displayed by humans and other animals. "
            "Study of intelligent agents: any device that perceives "
            "its environment and takes actions that maximize its "
            "chance of successfully achieving its goals."
        )
        text_bob = (
            "Machine intelligence, also known as artificial "
            "intelligence (AI), refers to intelligent behavior "
            "exhibited by software and machines. The field studies "
            "intelligent agents, which perceive surroundings and "
            "take actions to achieve specified goals effectively."
        )
        text_charlie = (
            "A blockchain is a decentralized, distributed, and "
            "public digital ledger used to record transactions "
            "across many computers."
        )

        documents = [
            ("Introduction_to_AI.pdf", text_alice, "Student_0001"),
            ("AI_Concepts_Homework.pdf", text_bob, "Student_0002"),
            ("Introduction_to_Blockchain.pdf", text_charlie, "Student_0003"),
        ]

        print("Adding dummy documents...")
        for filename, text, student_name in documents:
            add_document(
                filename=filename,
                file_hash=hashlib.sha256(text.encode()).hexdigest(),
                class_section="CS-101",
                student_name=student_name,
                assignment_title="Final Essay",
            )

        print("Generating mock embeddings with mathematical similarities...")
        random_generator = np.random.default_rng(RANDOM_SEED)

        alice_vector = random_generator.standard_normal(EMBEDDING_DIMENSION)
        alice_vector /= np.linalg.norm(alice_vector)

        bob_vector = generate_similar_vector(alice_vector, config.target_similarity, random_generator)
        actual_target_similarity = validate_target_similarity(
            alice_vector, bob_vector, config.target_similarity
        )

        charlie_vector = generate_similar_vector(alice_vector, BACKGROUND_SIMILARITY, random_generator)
        validate_target_similarity(alice_vector, charlie_vector, BACKGROUND_SIMILARITY)

        if config.verbose:
            print(
                "Validated target pair similarity: "
                f"requested={config.target_similarity:.6f}, "
                f"actual={actual_target_similarity:.6f}"
            )

        chunks = [
            (0, "Introduction_to_AI.pdf", 0, text_alice, alice_vector),
            (1, "AI_Concepts_Homework.pdf", 0, text_bob, bob_vector),
            (2, "Introduction_to_Blockchain.pdf", 0, text_charlie, charlie_vector),
        ]

        print("Inserting chunks...")
        add_chunks(chunks)

        print("Syncing plagiarism incidents...")
        sync_flagged_incidents(
            [
                {
                    "doc_a": "AI_Concepts_Homework.pdf",
                    "doc_b": "Introduction_to_AI.pdf",
                    "similarity": actual_target_similarity,
                    "severity": "High" if actual_target_similarity >= 0.80 else "Medium",
                }
            ],
            db_path=corpus_db_path,
        )

        print("Building and saving FAISS index...")
        matrix = np.vstack([alice_vector, bob_vector, charlie_vector])
        index = build_index_from_matrix(matrix)
        save_index(index, os.path.join(seed_dir, "corpus.index"))

        print(
            "Seed data successfully generated with target "
            f"similarity {actual_target_similarity:.1%} and stored "
            f"in {seed_dir}!"
        )
    finally:
        with open(_init_py, "w", encoding="utf-8") as _f:
            _f.write(_backup)
        importlib.invalidate_caches()


if __name__ == "__main__":
    main()
