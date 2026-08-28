"""
Unit tests for Multimodal PDF OCR & Neural Paraphrase Alignment Engine.

`MultimodalPDFOCREngine.get_extraction_summary` was declared without `self`
while its body used `self.processed_pages_log` (issue #3784), so the telemetry
summary was unreachable by any call form: bound calls raised TypeError
("takes 0 positional arguments but 1 was given") and unbound class calls got
past the signature only to raise NameError inside the body.

The suite below pins the bound-call contract and covers the aggregation the
summary is supposed to perform, which had never actually run.
"""

import inspect

import pytest

from src.services.multimodal_ocr_engine import (
    MultimodalPDFOCREngine,
    ParaphraseNeuralAlignmentEngine,
)

SAMPLE_PAGE = b"PDF_PAGE_MOCK_IMAGE_DATA_12345"
NOMINAL_CONFIDENCE = 98.4


@pytest.fixture
def ocr_engine():
    return MultimodalPDFOCREngine(dpi_resolution=300, enable_table_extraction=True)


# ── the regression itself ──────────────────────────────────────────────────────


def test_get_extraction_summary_is_a_bound_method():
    """Without `self` the method is uncallable on an instance."""
    signature = inspect.signature(MultimodalPDFOCREngine.get_extraction_summary)

    assert "self" in signature.parameters


def test_get_extraction_summary_is_callable_on_an_instance(ocr_engine):
    """This used to raise TypeError before the receiver was restored."""
    summary = ocr_engine.get_extraction_summary()

    assert isinstance(summary, dict)


def test_get_extraction_summary_reads_instance_state(ocr_engine):
    """Two engines must not share a page log."""
    ocr_engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)
    other = MultimodalPDFOCREngine()

    assert ocr_engine.get_extraction_summary()["totalPagesProcessed"] == 1
    assert other.get_extraction_summary()["totalPagesProcessed"] == 0


# ── OCR extraction ─────────────────────────────────────────────────────────────


def test_multimodal_pdf_ocr_extraction(ocr_engine):
    page_data = ocr_engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)

    assert page_data["pageNumber"] == 1
    assert "OCR Extracted Content" in page_data["extractedText"]
    assert page_data["layoutMetadata"]["ocrConfidenceScorePct"] > 90.0

    summary = ocr_engine.get_extraction_summary()
    assert summary["totalPagesProcessed"] == 1
    assert summary["status"] == "OCR_PIPELINE_READY"


def test_extraction_is_deterministic_for_the_same_bytes(ocr_engine):
    first = ocr_engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)
    second = ocr_engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)

    assert first["layoutMetadata"]["imageHash"] == second["layoutMetadata"]["imageHash"]


def test_different_bytes_produce_different_hashes(ocr_engine):
    a = ocr_engine.extract_text_from_pdf_page(1, b"page-one")
    b = ocr_engine.extract_text_from_pdf_page(2, b"page-two")

    assert a["layoutMetadata"]["imageHash"] != b["layoutMetadata"]["imageHash"]


def test_dpi_is_carried_into_layout_metadata():
    engine = MultimodalPDFOCREngine(dpi_resolution=600)

    page = engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)

    assert page["layoutMetadata"]["dpi"] == 600


def test_table_extraction_flag_controls_detected_table_count():
    on = MultimodalPDFOCREngine(enable_table_extraction=True)
    off = MultimodalPDFOCREngine(enable_table_extraction=False)

    assert on.extract_text_from_pdf_page(1, SAMPLE_PAGE)["layoutMetadata"][
        "detectedTablesCount"
    ] == 2
    assert off.extract_text_from_pdf_page(1, SAMPLE_PAGE)["layoutMetadata"][
        "detectedTablesCount"
    ] == 0


def test_empty_page_bytes_are_accepted(ocr_engine):
    """An empty scan should still hash and log rather than raise."""
    page = ocr_engine.extract_text_from_pdf_page(1, b"")

    assert page["layoutMetadata"]["imageHash"]
    assert ocr_engine.get_extraction_summary()["totalPagesProcessed"] == 1


def test_each_extraction_appends_to_the_page_log(ocr_engine):
    for page_number in range(1, 4):
        ocr_engine.extract_text_from_pdf_page(page_number, SAMPLE_PAGE)

    assert len(ocr_engine.processed_pages_log) == 3
    assert [p["pageNumber"] for p in ocr_engine.processed_pages_log] == [1, 2, 3]


# ── summary aggregation ────────────────────────────────────────────────────────


def test_summary_on_a_fresh_engine_does_not_divide_by_zero(ocr_engine):
    """`total_pages or 1` guards the mean; assert the guard actually holds."""
    summary = ocr_engine.get_extraction_summary()

    assert summary["totalPagesProcessed"] == 0
    assert summary["avgOCRConfidencePct"] == 0.0


def test_summary_counts_every_processed_page(ocr_engine):
    for page_number in range(1, 6):
        ocr_engine.extract_text_from_pdf_page(page_number, SAMPLE_PAGE)

    assert ocr_engine.get_extraction_summary()["totalPagesProcessed"] == 5


def test_summary_averages_the_confidence_scores(ocr_engine):
    ocr_engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)
    ocr_engine.extract_text_from_pdf_page(2, SAMPLE_PAGE)

    assert ocr_engine.get_extraction_summary()["avgOCRConfidencePct"] == pytest.approx(
        NOMINAL_CONFIDENCE
    )


def test_summary_average_reflects_edited_page_records(ocr_engine):
    """The mean must be computed, not hardcoded to the nominal score."""
    ocr_engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)
    ocr_engine.extract_text_from_pdf_page(2, SAMPLE_PAGE)
    ocr_engine.processed_pages_log[0]["layoutMetadata"]["ocrConfidenceScorePct"] = 90.0
    ocr_engine.processed_pages_log[1]["layoutMetadata"]["ocrConfidenceScorePct"] = 80.0

    assert ocr_engine.get_extraction_summary()["avgOCRConfidencePct"] == pytest.approx(
        85.0
    )


def test_summary_rounds_to_two_decimals(ocr_engine):
    ocr_engine.extract_text_from_pdf_page(1, SAMPLE_PAGE)
    ocr_engine.processed_pages_log[0]["layoutMetadata"][
        "ocrConfidenceScorePct"
    ] = 91.23456

    assert ocr_engine.get_extraction_summary()["avgOCRConfidencePct"] == 91.23


def test_summary_shape_is_stable(ocr_engine):
    assert set(ocr_engine.get_extraction_summary()) == {
        "totalPagesProcessed",
        "avgOCRConfidencePct",
        "status",
    }


# ── paraphrase alignment ───────────────────────────────────────────────────────


def test_paraphrase_neural_alignment_engine():
    paraphrase_engine = ParaphraseNeuralAlignmentEngine(semantic_threshold=0.75)
    sent_a = (
        "Deep learning transformers utilize self-attention mechanisms for "
        "language models."
    )
    sent_b = (
        "Self-attention layers in transformer architectures enable robust "
        "neural text modeling."
    )

    alignment = paraphrase_engine.align_sentence_pair(sent_a, sent_b)

    assert alignment["paraphraseSimilarityScore"] > 0.0
    assert "confidenceGrade" in alignment


def test_identical_sentences_align_perfectly():
    engine = ParaphraseNeuralAlignmentEngine()
    text = "Semantic similarity detection across student assignments."

    alignment = engine.align_sentence_pair(text, text)

    assert alignment["paraphraseSimilarityScore"] == pytest.approx(1.0)
    assert alignment["isParaphraseDetected"] is True
    assert alignment["confidenceGrade"] == "HIGH_PARAPHRASE"


def test_empty_sentences_do_not_divide_by_zero():
    """Zero vectors would give a 0/0 magnitude without the `or 1.0` guard."""
    engine = ParaphraseNeuralAlignmentEngine()

    alignment = engine.align_sentence_pair("", "")

    assert alignment["paraphraseSimilarityScore"] == 0.0
    assert alignment["isParaphraseDetected"] is False


def test_threshold_decides_the_paraphrase_verdict():
    text = "Vector search over document embeddings."
    strict = ParaphraseNeuralAlignmentEngine(semantic_threshold=1.1)
    lenient = ParaphraseNeuralAlignmentEngine(semantic_threshold=0.1)

    assert strict.align_sentence_pair(text, text)["isParaphraseDetected"] is False
    assert lenient.align_sentence_pair(text, text)["isParaphraseDetected"] is True


def test_alignments_accumulate_on_the_instance():
    engine = ParaphraseNeuralAlignmentEngine()

    engine.align_sentence_pair("alpha beta", "beta alpha")
    engine.align_sentence_pair("gamma", "delta")

    assert len(engine.aligned_sentence_pairs) == 2


def test_similarity_is_symmetric():
    engine = ParaphraseNeuralAlignmentEngine()
    a = "Paraphrase detection using dense retrieval."
    b = "Dense retrieval applied to paraphrase detection."

    forward = engine.align_sentence_pair(a, b)["paraphraseSimilarityScore"]
    backward = engine.align_sentence_pair(b, a)["paraphraseSimilarityScore"]

    assert forward == pytest.approx(backward)


def test_vectorize_sentence_has_the_declared_width():
    engine = ParaphraseNeuralAlignmentEngine()

    assert len(engine._vectorize_sentence("some words here")) == 128
