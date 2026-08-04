from pathlib import Path


EMBEDDINGS_PATH = Path("src/core/embeddings.py")
TEST_PATH = Path(
    "tests/core/test_embeddings_multithreaded_issue_1261.py"
)


def test_generate_embeddings_has_required_parameter():
    source = EMBEDDINGS_PATH.read_text(encoding="utf-8")

    assert "def generate_embeddings(" in source
    assert "num_threads: int = 4" in source


def test_thread_pool_and_batch_processing_are_used():
    source = EMBEDDINGS_PATH.read_text(encoding="utf-8")

    assert "ThreadPoolExecutor(" in source
    assert "executor.map(_encode_batch, batches)" in source
    assert "np.concatenate(encoded_batches" in source


def test_dimension_equivalence_test_exists():
    source = TEST_PATH.read_text(encoding="utf-8")

    assert (
        "test_multithreaded_dimensions_match_single_threaded"
        in source
    )
    assert "single.shape == multi.shape" in source
