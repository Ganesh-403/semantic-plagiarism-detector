"""
tests/core/test_pptx_scan.py
----------------------------
Unit tests for Presentation Slide (PPTX) Semantic and Layout Plagiarism Detection.
"""

import pytest
import zipfile
import io
from src.core.pptx_slide_extractor import extract_presentation_deck, Slide, SlideElement
from src.core.slide_sequence_aligner import (
    compute_slide_text_similarity,
    compute_deck_similarity,
    PresentationDeck,
)


class TestPPTXSlideExtractor:
    def test_extract_presentation_deck_empty(self):
        deck = extract_presentation_deck(b"")
        assert deck.slide_count == 0

    def test_extract_presentation_deck_invalid_zip(self):
        deck = extract_presentation_deck(b"not a zip file")
        assert deck.slide_count == 0

    def test_extract_presentation_deck_valid_mock(self):
        # Create a minimal mock PPTX (ZIP with one slide XML)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            slide_xml = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" 
                   xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                <p:cSld>
                    <p:spTree>
                        <p:sp>
                            <p:txBody>
                                <a:p><a:r><a:t>Hello World</a:t></a:r></a:p>
                            </p:txBody>
                        </p:sp>
                    </p:spTree>
                </p:cSld>
            </p:sld>"""
            zf.writestr("ppt/slides/slide1.xml", slide_xml)

        deck = extract_presentation_deck(buffer.getvalue())
        assert deck.slide_count == 1
        assert "Hello World" in deck.slides[0].get_full_text()


class TestSlideSequenceAligner:
    def test_compute_slide_text_similarity_identical(self):
        slide_a = Slide(slide_number=1, elements=[SlideElement(text="Hello World")])
        slide_b = Slide(slide_number=1, elements=[SlideElement(text="Hello World")])
        sim = compute_slide_text_similarity(slide_a, slide_b)
        assert sim == 1.0

    def test_compute_slide_text_similarity_different(self):
        slide_a = Slide(slide_number=1, elements=[SlideElement(text="Hello World")])
        slide_b = Slide(
            slide_number=1, elements=[SlideElement(text="Goodbye Universe")]
        )
        sim = compute_slide_text_similarity(slide_a, slide_b)
        assert sim < 0.5

    def test_compute_deck_similarity_cloned(self):
        deck_a = PresentationDeck(
            slide_count=1, slides=[Slide(1, [SlideElement(text="A")])]
        )
        deck_b = PresentationDeck(
            slide_count=1, slides=[Slide(1, [SlideElement(text="A")])]
        )
        result = compute_deck_similarity(deck_a, deck_b)
        assert result["overall_score"] == 1.0
        assert result["is_cloned_deck"] is True
