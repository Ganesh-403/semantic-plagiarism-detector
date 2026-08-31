import time

from src.core.embeddings import generate_embeddings


def test_embedding_generation_performance():
    """Ensure embedding generation remains within acceptable latency."""

    texts = [
        "Artificial intelligence is transforming software development.",
        "Python is widely used for backend services.",
        "Machine learning enables predictive analytics.",
        "Vector databases improve semantic search.",
        "Cloud computing powers scalable applications.",
        "Cybersecurity protects sensitive information.",
        "Open source software accelerates innovation.",
        "Unit testing improves software reliability.",
        "Continuous integration catches regressions early.",
        "Performance optimization enhances user experience.",
    ]

    start_time = time.perf_counter()

    embeddings = generate_embeddings(texts)

    elapsed = time.perf_counter() - start_time

    assert len(embeddings) == len(texts)
    assert elapsed < 5.0, (
        f"Embedding generation took {elapsed:.2f}s which exceeds the 5.0s threshold."
    )
