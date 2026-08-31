"""
src/core/pptx_slide_extractor.py
--------------------------------
Presentation Slide (PPTX) Extractor.

Parses PPTX files (which are ZIP archives containing XML) to extract
slide text, speaker notes, and visual element bounding boxes. This
allows the system to detect slide-by-slide semantic alignment and
layout cloning.
"""

import zipfile
import xml.etree.ElementTree as ET
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SlideElement:
    """Represents a text or visual element on a slide."""

    text: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Slide:
    """Represents a single presentation slide."""

    slide_number: int
    elements: List[SlideElement] = field(default_factory=list)
    notes: str = ""

    def get_full_text(self) -> str:
        return " ".join(e.text for e in self.elements if e.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_number": self.slide_number,
            "elements": [e.to_dict() for e in self.elements],
            "notes": self.notes,
        }


@dataclass
class PresentationDeck:
    """Represents a complete presentation deck."""

    slide_count: int
    slides: List[Slide] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_count": self.slide_count,
            "slides": [s.to_dict() for s in self.slides],
        }


# PPTX XML Namespaces
NSMAP = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _extract_text_from_xml(xml_content: bytes) -> List[SlideElement]:
    """Extract text and bounding boxes from a slide XML file."""
    elements = []
    try:
        root = ET.fromstring(xml_content)

        # Find all shape trees (sp)
        for sp in root.findall(".//p:sp", NSMAP):
            # Extract text from <a:t> tags
            text_parts = [t.text for t in sp.findall(".//a:t", NSMAP) if t.text]
            text = " ".join(text_parts).strip()

            # Extract bounding box from <a:off> and <a:ext>
            x, y, w, h = 0, 0, 0, 0
            off = sp.find(".//a:off", NSMAP)
            ext = sp.find(".//a:ext", NSMAP)

            if off is not None:
                x = int(off.get("x", 0))
                y = int(off.get("y", 0))
            if ext is not None:
                w = int(ext.get("cx", 0))
                h = int(ext.get("cy", 0))

            if text or (w > 0 and h > 0):
                elements.append(SlideElement(text=text, x=x, y=y, width=w, height=h))

    except ET.ParseError as e:
        logger.warning("Failed to parse slide XML: %s", e)

    return elements


def _extract_notes_from_xml(xml_content: bytes) -> str:
    """Extract speaker notes from a notesSlide XML file."""
    try:
        root = ET.fromstring(xml_content)
        # Notes are typically in <a:t> tags within the notes slide
        text_parts = [t.text for t in root.findall(".//a:t", NSMAP) if t.text]
        return " ".join(text_parts).strip()
    except ET.ParseError:
        return ""


def extract_presentation_deck(pptx_bytes: bytes) -> PresentationDeck:
    """Parse a PPTX file and extract all slides, text, and notes.

    Args:
        pptx_bytes: Raw bytes of the PPTX file.

    Returns:
        A PresentationDeck object containing all extracted slide data.
    """
    if not pptx_bytes:
        return PresentationDeck(slide_count=0)

    slides = []

    try:
        with zipfile.ZipFile(pptx_bytes) as zf:
            # Find all slide files (e.g., ppt/slides/slide1.xml)
            slide_files = sorted(
                [f for f in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml", f)],
                key=lambda x: int(re.search(r"\d+", x.split("/")[-1]).group()),
            )

            for slide_file in slide_files:
                slide_num = int(re.search(r"\d+", slide_file.split("/")[-1]).group())
                slide_xml = zf.read(slide_file)
                elements = _extract_text_from_xml(slide_xml)

                # Attempt to find corresponding notes slide
                notes_file = slide_file.replace(
                    "ppt/slides/slide", "ppt/notesSlides/notesSlide"
                )
                notes_text = ""
                if notes_file in zf.namelist():
                    notes_xml = zf.read(notes_file)
                    notes_text = _extract_notes_from_xml(notes_xml)

                slides.append(
                    Slide(slide_number=slide_num, elements=elements, notes=notes_text)
                )

    except zipfile.BadZipFile:
        logger.error("Invalid PPTX file: Not a valid ZIP archive.")
        return PresentationDeck(slide_count=0)

    logger.info("Extracted %d slides from presentation.", len(slides))

    return PresentationDeck(slide_count=len(slides), slides=slides)
