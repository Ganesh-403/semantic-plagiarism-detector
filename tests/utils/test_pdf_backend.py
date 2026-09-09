"""Contract tests for PDF parsing, rendering, encryption and metadata removal."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter
from src.utils import pdf_backend as pdf


def sample_pdf():
    doc = pdf.open()
    doc.new_page(width=300, height=400).insert_text((30, 60), "First matching paragraph.")
    doc.new_page(width=300, height=400).insert_text((30, 60), "Second page content.")
    return doc


def test_text_search_annotation_roundtrip(tmp_path):
    with sample_pdf() as doc:
        page = doc[0]
        rect = page.search_for("matching")[0]
        assert page.rect.contains(rect)
        annotation = page.add_highlight_annot(rect)
        annotation.set_info(content="Source B", title="Similarity: 90%")
        annotation.set_colors(stroke=(1., 0.8, 0.))
        annotation.update()
        doc.save(tmp_path / "report.pdf")
    with pdf.open(tmp_path / "report.pdf") as loaded:
        assert len(loaded) == 2
        assert "First matching paragraph" in loaded[0].get_text()
        annotation = list(loaded[0].annots())[0]
        assert annotation.type == (8, "Highlight")
        assert annotation.info == {"content": "Source B", "title": "Similarity: 90%"}
        assert annotation.colors["stroke"] == pytest.approx((1, .8, 0))
        assert tuple(annotation.rect) == pytest.approx(tuple(rect))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotation_coordinates_and_rendering(rotation):
    with sample_pdf() as doc:
        page = doc[0]
        page.set_rotation(rotation)
        rect = pdf.Rect(10, 20, 50, 80)
        assert tuple(rect * page.rotation_matrix * page.derotation_matrix) == pytest.approx(tuple(rect))
        image = page.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=True)
        expected = (800, 600) if rotation in (90, 270) else (600, 800)
        assert (image.width, image.height) == expected
        assert image.n == 4
        assert len(image.samples) == image.width * image.height * 4
        assert Image.open(BytesIO(image.tobytes())).size == expected
        assert page.search_for("matching")


def test_passwords_and_unencrypted_export():
    with sample_pdf() as doc:
        encrypted = doc.tobytes(encryption=pdf.PDF_ENCRYPT_AES_256, user_pw="UserPassword123!", owner_pw="OwnerPassword123!")
    with pdf.open(stream=encrypted) as doc:
        assert doc.needs_pass
        with pytest.raises(ValueError, match="authentication"):
            doc[0].get_text()
        assert doc.authenticate("wrong") == 0
        assert doc.authenticate("UserPassword123!")
        assert not doc.needs_pass
        assert "matching" in doc[0].get_text()
        output = doc.tobytes(encryption=pdf.PDF_ENCRYPT_NONE)
    assert not PdfReader(BytesIO(output)).is_encrypted
    with pdf.open(stream=encrypted) as doc:
        assert doc.authenticate("OwnerPassword123!")


def test_metadata_scrubbing_removes_orphaned_values():
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.add_metadata({"/Author": "Sensitive Author", "/Custom": "Sensitive Custom"})
    buffer = BytesIO()
    writer.write(buffer)
    with pdf.open(stream=buffer.getvalue()) as doc:
        assert doc.metadata["author"] == "Sensitive Author"
        doc.set_metadata({})
        doc.del_xml_metadata()
        output = doc.tobytes(garbage=4, clean=True)
    assert b"Sensitive" not in output
    assert not PdfReader(BytesIO(output)).metadata


def test_render_cache_reused_and_invalidated(monkeypatch):
    with sample_pdf() as doc:
        serializer = Mock(wraps=doc.tobytes)
        monkeypatch.setattr(doc, "tobytes", serializer)
        for page in doc:
            assert page.get_text()
            assert page.search_for("page") or page.search_for("matching")
        assert serializer.call_count == 1
        doc[0].insert_text((30, 90), "Added later")
        assert "Added later" in doc[0].get_text()
        assert serializer.call_count == 2
        annotation = doc[0].add_highlight_annot(doc[0].search_for("Added")[0])
        annotation.set_colors(stroke=(1, 0, 0))
        doc[0].get_pixmap()
        assert serializer.call_count == 3
        doc[0].set_rotation(90)
        assert doc[0].get_pixmap().width == 400
        assert serializer.call_count == 4


def test_insert_image_and_fonts():
    buffer = BytesIO()
    Image.new("RGB", (20, 30), "red").save(buffer, format="PNG")
    with sample_pdf() as doc:
        page = doc[0]
        page.insert_image(pdf.Rect(30, 100, 90, 190), stream=buffer.getvalue())
        assert len(page.get_images()) == 1
        assert page.get_fonts()
        assert doc.extract_font(page.get_fonts()[0][0])[-1] == b""  # Standard PDF font.
        assert page.get_pixmap().image.getpixel((50, 150)) == (255, 0, 0)
        assert page.get_text("blocks")[0][4]


def test_concurrent_independent_documents():
    with sample_pdf() as doc:
        data = doc.tobytes()
    def read(_):
        with pdf.open(stream=data) as doc:
            return doc[0].get_text(), doc[1].get_pixmap().width
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, range(12)))
    assert all("matching" in text and width == 300 for text, width in results)


def test_invalid_input_closed_document_and_page_bounds():
    with pytest.raises(pdf.FileDataError):
        pdf.open(stream=b"not a PDF")
    with sample_pdf() as doc:
        assert "Second" in doc[-1].get_text()
        with pytest.raises(IndexError):
            doc[2]
        with pytest.raises(ValueError):
            doc[0].get_text("unsupported")
    assert doc.is_closed
    with pytest.raises(ValueError, match="closed"):
        doc.tobytes()
