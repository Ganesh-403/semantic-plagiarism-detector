"""Tests for corpus-level exact and near-duplicate detection."""

from __future__ import annotations

from src.core.corpus_duplicate_filter import (
    build_fingerprint,
    classify_relationship,
    estimate_similarity,
)


def test_exact_duplicate_is_detected():
    first = build_fingerprint(
        "first.pdf",
        "This is the original assignment text.",
    )
    second = build_fingerprint(
        "second.pdf",
        "This is the original assignment text.",
    )

    relationship, similarity = classify_relationship(first, second)

    assert relationship == "exact_duplicate"
    assert similarity == 1.0


def test_formatting_only_changes_are_exact_duplicates():
    first = build_fingerprint(
        "first.pdf",
        "This is the original assignment.\n\nIt contains several words.",
    )
    second = build_fingerprint(
        "second.pdf",
        "  THIS is the original assignment. "
        "It contains several words.  ",
    )

    relationship, similarity = classify_relationship(first, second)

    assert relationship == "exact_duplicate"
    assert similarity == 1.0


def test_minor_edits_can_be_detected_as_near_duplicate():
    first = build_fingerprint(
        "first.pdf",
        (
            "Machine learning is a method for building predictive systems. "
            "It uses data to learn useful patterns from examples. "
            "These systems can improve when additional training data is provided."
        ),
    )

    second = build_fingerprint(
        "second.pdf",
        (
            "Machine learning is a method for building predictive systems. "
            "It uses data to learn useful patterns from examples. "
            "These systems can improve when additional training data is available."
        ),
    )

    relationship, similarity = classify_relationship(
        first,
        second,
        threshold=0.80,
    )

    assert relationship == "near_duplicate"
    assert similarity >= 0.80


def test_genuinely_different_documents_are_not_duplicates():
    first = build_fingerprint(
        "first.pdf",
        (
            "Machine learning models classify documents using statistical "
            "patterns extracted from training data."
        ),
    )

    second = build_fingerprint(
        "second.pdf",
        (
            "The solar system contains planets orbiting a central star. "
            "Astronomical observations help scientists understand their motion."
        ),
    )

    relationship, similarity = classify_relationship(
        first,
        second,
        threshold=0.92,
    )

    assert relationship is None
    assert similarity < 0.92


def test_similarity_is_bounded():
    first = build_fingerprint(
        "first.pdf",
        "A document containing several useful words for testing.",
    )
    second = build_fingerprint(
        "second.pdf",
        "A completely different collection of words.",
    )

    similarity = estimate_similarity(first, second)

    assert 0.0 <= similarity <= 1.0

    """Tests for corpus-level exact and near-duplicate detection."""

from __future__ import annotations

from src.core.corpus_duplicate_filter import (
    build_fingerprint,
    classify_relationship,
    estimate_similarity,
)


def test_exact_duplicate_is_detected():
    first = build_fingerprint(
        "first.pdf",
        "This is the original assignment text.",
    )
    second = build_fingerprint(
        "second.pdf",
        "This is the original assignment text.",
    )

    relationship, similarity = classify_relationship(first, second)

    assert relationship == "exact_duplicate"
    assert similarity == 1.0


def test_formatting_only_changes_are_exact_duplicates():
    first = build_fingerprint(
        "first.pdf",
        "This is the original assignment.\n\nIt contains several words.",
    )
    second = build_fingerprint(
        "second.pdf",
        "  THIS is the original assignment. "
        "It contains several words.  ",
    )

    relationship, similarity = classify_relationship(first, second)

    assert relationship == "exact_duplicate"
    assert similarity == 1.0


def test_minor_edits_can_be_detected_as_near_duplicate():
    first = build_fingerprint(
        "first.pdf",
        (
            "Machine learning is a method for building predictive systems. "
            "It uses data to learn useful patterns from examples. "
            "These systems can improve when additional training data is provided."
        ),
    )

    second = build_fingerprint(
        "second.pdf",
        (
            "Machine learning is a method for building predictive systems. "
            "It uses data to learn useful patterns from examples. "
            "These systems can improve when additional training data is available."
        ),
    )

    relationship, similarity = classify_relationship(
        first,
        second,
        threshold=0.80,
    )

    assert relationship == "near_duplicate"
    assert similarity >= 0.80


def test_genuinely_different_documents_are_not_duplicates():
    first = build_fingerprint(
        "first.pdf",
        (
            "Machine learning models classify documents using statistical "
            "patterns extracted from training data."
        ),
    )

    second = build_fingerprint(
        "second.pdf",
        (
            "The solar system contains planets orbiting a central star. "
            "Astronomical observations help scientists understand their motion."
        ),
    )

    relationship, similarity = classify_relationship(
        first,
        second,
        threshold=0.92,
    )

    assert relationship is None
    assert similarity < 0.92


def test_similarity_is_bounded():
    first = build_fingerprint(
        "first.pdf",
        "A document containing several useful words for testing.",
    )
    second = build_fingerprint(
        "second.pdf",
        "A completely different collection of words.",
    )

    similarity = estimate_similarity(first, second)

    assert 0.0 <= similarity <= 1.0