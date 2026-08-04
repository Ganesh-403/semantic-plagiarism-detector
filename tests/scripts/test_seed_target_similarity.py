import argparse

import numpy as np
import pytest

from scripts.generate_seed_data import (
    DEFAULT_TARGET_SIMILARITY,
    calculate_cosine_similarity,
    generate_similar_vector,
    parse_args,
    parse_target_similarity,
    validate_target_similarity,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("0.85", 0.85),
        ("85", 0.85),
        ("85%", 0.85),
        (" 90% ", 0.90),
        ("1", 1.0),
        ("100", 1.0),
        ("0", 0.0),
    ],
)
def test_parse_target_similarity_accepts_supported_formats(
    raw_value,
    expected,
):
    assert parse_target_similarity(raw_value) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "raw_value",
    [
        "-0.01",
        "101",
        "101%",
        "1.01",
        "invalid",
        "",
    ],
)
def test_parse_target_similarity_rejects_invalid_values(
    raw_value,
):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_similarity(raw_value)


def test_cli_uses_existing_95_percent_default():
    config = parse_args([])

    assert config.target_similarity == (
        DEFAULT_TARGET_SIMILARITY
    )
    assert config.verbose is False


def test_cli_accepts_decimal_target():
    config = parse_args(
        ["--target-similarity", "0.85"]
    )

    assert config.target_similarity == pytest.approx(0.85)


def test_cli_accepts_percentage_target():
    config = parse_args(
        ["--target-similarity", "85%"]
    )

    assert config.target_similarity == pytest.approx(0.85)


@pytest.mark.parametrize(
    "target",
    [0.0, 0.15, 0.50, 0.85, 0.95, 1.0],
)
def test_generated_vector_hits_requested_similarity(target):
    random_generator = np.random.default_rng(42)
    base_vector = random_generator.standard_normal(64)
    base_vector /= np.linalg.norm(base_vector)

    generated = generate_similar_vector(
        base_vector,
        target,
        random_generator,
    )
    actual = validate_target_similarity(
        base_vector,
        generated,
        target,
    )

    assert actual == pytest.approx(target, abs=1e-6)


def test_validate_target_similarity_rejects_wrong_vector():
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])

    with pytest.raises(
        ValueError,
        match="does not match target",
    ):
        validate_target_similarity(
            first,
            second,
            0.85,
        )


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(
        ValueError,
        match="non-zero vectors",
    ):
        calculate_cosine_similarity(
            np.zeros(2),
            np.ones(2),
        )
