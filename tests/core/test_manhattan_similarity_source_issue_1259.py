from pathlib import Path


SIMILARITY_PATH = Path("src/core/similarity.py")
TEST_PATH = Path("tests/core/test_similarity.py")


def test_required_function_exists_with_annotation():
    source = SIMILARITY_PATH.read_text(encoding="utf-8")

    assert "def manhattan_similarity(" in source
    assert "vec_a: np.ndarray" in source
    assert "vec_b: np.ndarray" in source
    assert ") -> float:" in source


def test_similarity_uses_normalized_manhattan_formula():
    source = SIMILARITY_PATH.read_text(encoding="utf-8")

    assert "np.sum(np.abs(array_a - array_b)" in source
    assert "1.0 / (1.0 + distance)" in source
    assert "np.clip(similarity, 0.0, 1.0)" in source


def test_unit_tests_are_added_to_requested_file():
    source = TEST_PATH.read_text(encoding="utf-8")

    assert (
        "test_manhattan_similarity_identical_vectors"
        in source
    )
    assert (
        "test_manhattan_similarity_rejects_shape_mismatch"
        in source
    )
    assert (
        "test_manhattan_similarity_remains_bounded"
        in source
    )
