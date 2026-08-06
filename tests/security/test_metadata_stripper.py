import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from src.security.metadata_stripper import (
    strip_exif_metadata,
    strip_pdf_javascript,
    inspect_pdf_fonts,
)

def test_strip_pdf_javascript_removes_open_action():
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, TextStringObject

    pdf = io.BytesIO()

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)

    javascript_action = DictionaryObject()
    javascript_action.update(
        {
            NameObject("/S"): NameObject("/JavaScript"),
            NameObject("/JS"): TextStringObject("app.alert('test')"),
        }
    )

    writer._root_object.update({NameObject("/OpenAction"): javascript_action})

    writer.write(pdf)

    cleaned = strip_pdf_javascript(pdf.getvalue())

    reader = PdfReader(io.BytesIO(cleaned))
    root = reader.trailer["/Root"]

    assert "/OpenAction" not in root


def test_strip_exif_metadata_size_within_limit():
    # A small dummy file of 100 bytes
    dummy_data = b"Hello world, this is a small text file."
    result = strip_exif_metadata(dummy_data, "test.txt")
    assert result == dummy_data


def test_strip_exif_metadata_size_exceeds_default_limit():
    # Default limit is 25 MB (25,000,000 bytes)
    # We will pass a dummy byte array that is larger than 25,000,000 bytes
    large_data = b"0" * 25_000_001
    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(large_data, "test.txt")
    assert "File size exceeds EXIF stripping limit" in str(excinfo.value)


def test_strip_exif_metadata_size_exceeds_custom_limit():
    # Pass a custom max_bytes limit of 100 bytes and a file of 101 bytes
    small_data = b"0" * 101
    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(small_data, "test.txt", max_bytes=100)
    assert "File size exceeds EXIF stripping limit" in str(excinfo.value)


def test_strip_exif_metadata_exactly_at_custom_limit():
    # Pass a custom max_bytes limit of 100 bytes and a file of exactly 100 bytes
    data = b"0" * 100
    result = strip_exif_metadata(data, "test.txt", max_bytes=100)
    assert result == data


def test_strip_image_metadata_dimension_safety_limit_width():
    # Create a dummy image that exceeds the 10,000px width limit
    # We create a minimal valid PNG header + fake large dimensions
    # Using PIL to generate a 10001x100 image to trigger the check
    img = Image.new("RGB", (10001, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(img_bytes.getvalue(), "test.png")
    assert "Image dimensions exceed 10,000px safety limit" in str(excinfo.value)


def test_strip_image_metadata_dimension_safety_limit_height():
    # Create a dummy image that exceeds the 10,000px height limit
    img = Image.new("RGB", (100, 10001), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")

    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(img_bytes.getvalue(), "test.jpg")
    assert "Image dimensions exceed 10,000px safety limit" in str(excinfo.value)


def test_strip_image_metadata_dimension_exactly_at_limit():
    # Create a dummy image exactly at the 10,000px limit (should pass)
    img = Image.new("RGB", (10000, 10000), color="green")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    # Should not raise an error
    result = strip_exif_metadata(img_bytes.getvalue(), "test.png")
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_strip_palette_image_preserves_colors():
    # A palette-based (P mode) image: pixels store palette indices, not channels
    palette_image = Image.new("P", (10, 10))
    palette_colors = []
    for i in range(256):
        palette_colors.extend((i, 0, 255 - i))
    palette_image.putpalette(palette_colors)
    # Index 7 maps to RGB (7, 0, 248)
    palette_image.putdata([7] * 100)
    img_bytes = io.BytesIO()
    palette_image.save(img_bytes, format="PNG")

    result = strip_exif_metadata(img_bytes.getvalue(), "test.png")

    with Image.open(io.BytesIO(result)) as out_image:
        # P mode must be converted to RGBA before saving so channels survive
        assert out_image.mode == "RGBA"
        pixel = out_image.getpixel((5, 5))
    assert pixel == (7, 0, 248, 255)


def test_strip_image_metadata_decompression_bomb(monkeypatch):
    # Mock Image.open to raise DecompressionBombError
    def mock_open(*args, **kwargs):
        raise Image.DecompressionBombError("Image size exceeds limit")
    
    monkeypatch.setattr(Image, "open", mock_open)
    
    img_bytes = b"fake image bytes"
    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(img_bytes, "test.jpg")
    
    assert str(excinfo.value) == "Image dimensions exceed security safety limits."


def test_rejects_image_with_excessive_memory_footprint(monkeypatch):
    class MockImage:
        size = (5120, 5121)  # 5120 * 5121 * 4 = 104,878,080 bytes > 100 MB
        mode = "RGBA"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_open(*args, **kwargs):
        return MockImage()

    monkeypatch.setattr(Image, "open", mock_open)

    with pytest.raises(
        ValueError,
        match="Decompressed image memory footprint exceeds 100 MB safety limit",
    ):
        strip_exif_metadata(b"fake image bytes", "test.jpg")


def test_accepts_image_with_safe_memory_footprint():
    img = Image.new("RGB", (100, 100), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")

    result = strip_exif_metadata(img_bytes.getvalue(), "test.jpg")
    assert isinstance(result, bytes)
    assert len(result) > 0

@patch("src.security.metadata_stripper.fitz.open")
def test_inspect_pdf_fonts_exceeds_limit(mock_fitz_open):
    mock_page = MagicMock()
    mock_page.get_fonts.return_value = [(7, 0, "Type1", "F1", "Arial", "")]

    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = iter([mock_page])
    mock_doc.extract_font.return_value = ("Arial", "ttf", "Type1", b"0" * 10_000_001)
    mock_fitz_open.return_value = mock_doc

    with pytest.raises(ValueError) as excinfo:
        inspect_pdf_fonts(b"dummy pdf bytes", max_font_bytes=10_000_000)
    assert "Embedded PDF font stream exceeds safety limit" in str(excinfo.value)


@patch("src.security.metadata_stripper.fitz.open")
def test_inspect_pdf_fonts_within_limit(mock_fitz_open):
    mock_page = MagicMock()
    mock_page.get_fonts.return_value = [(7, 0, "Type1", "F1", "Arial", "")]

    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = iter([mock_page])
    mock_doc.extract_font.return_value = ("Arial", "ttf", "Type1", b"0" * 100)
    mock_fitz_open.return_value = mock_doc

    result = inspect_pdf_fonts(b"dummy pdf bytes")
    assert result is True
