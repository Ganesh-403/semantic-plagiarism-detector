"""Tests for language-aware plagiarism similarity configuration."""

from src.core.language_similarity_config import (
    build_language_metadata,
    detect_document_language,
    get_language_pair_policy,
)


def test_same_language_policy():
    policy = get_language_pair_policy(
        "en",
        "en",
        base_threshold=0.75,
    )

    assert policy.same_language is True
    assert policy.cross_lingual is False
    assert policy.threshold == 0.75
    assert policy.lexical_processing_available is True


def test_cross_language_policy():
    policy = get_language_pair_policy(
        "es",
        "en",
        base_threshold=0.75,
    )

    assert policy.same_language is False
    assert policy.cross_lingual is True
    assert policy.threshold == 0.75
    assert policy.embedding_compatible is True
    assert policy.lexical_processing_available is False


def test_cross_language_threshold_can_be_configured(monkeypatch):
    monkeypatch.setenv(
        "PLAGIARISM_CROSS_LANGUAGE_THRESHOLD",
        "0.65",
    )

    policy = get_language_pair_policy(
        "fr",
        "en",
        base_threshold=0.75,
    )

    assert policy.cross_lingual is True
    assert policy.threshold == 0.65


def test_low_confidence_language_is_safe():
    policy = get_language_pair_policy(
        "unknown",
        "en",
        detection_confident=False,
        base_threshold=0.75,
    )

    assert policy.same_language is False
    assert policy.cross_lingual is False
    assert policy.detection_confident is False
    assert policy.threshold == 0.75


def test_document_language_metadata():
    documents = {
        "english.txt": (
            "The student submitted the assignment to the professor "
            "for review and received detailed feedback."
        ),
        "spanish.txt": (
            "El estudiante presentó su trabajo al profesor "
            "para revisión y recibió comentarios detallados."
        ),
    }

    metadata = build_language_metadata(documents)

    assert metadata["english.txt"]["language"] == "en"
    assert metadata["english.txt"]["language_confident"] is True

    assert metadata["spanish.txt"]["language"] == "es"
    assert metadata["spanish.txt"]["language_confident"] is True


def test_unknown_language_is_safe():
    language, confident = detect_document_language(
        "12345 67890 111213 141516"
    )

    assert language == "unknown"
    assert confident is False