import io
import pytest
from PIL import Image
from src.security.metadata_stripper import strip_exif_metadata, strip_pdf_javascript


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
    # Use "L" mode (1 byte/px) so 10000x10000 = 100MB, which is exactly at/under the 100MB memory limit
    img = Image.new("L", (10000, 10000), color=0)
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


def test_strip_image_metadata_memory_limit_exceeded(monkeypatch):
    # Mock Image.open to return an object with large dimensions but no actual memory allocation
    class MockImage:
        def __init__(self, mode, size):
            self.mode = mode
            self.size = size
            self.format = "PNG"
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # A 6000x6000 RGB image = 6000 * 6000 * 3 = 108,000,000 bytes > 100 MB
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: MockImage("RGB", (6000, 6000)))

    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(b"fake_data", "test.png")
    
    assert str(excinfo.value) == "Decompressed image memory footprint exceeds 100 MB safety limit"


def test_strip_image_metadata_memory_limit_accepted(monkeypatch):
    # Mock an image that is just under 100 MB
    class MockImage:
        def __init__(self, mode, size):
            self.mode = mode
            self.size = size
            self.format = "PNG"
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def getdata(self):
            return []
        def convert(self, mode):
            return self

    # A 5000x5000 RGB image = 5000 * 5000 * 3 = 75,000,000 bytes < 100 MB
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: MockImage("RGB", (5000, 5000)))
    
    # We also have to mock Image.new since the function creates a new image
    monkeypatch.setattr(Image, "new", lambda *args, **kwargs: MockImage("RGB", (5000, 5000)))
    
    # And mock putdata / save
    MockImage.putdata = lambda self, data: None
    MockImage.save = lambda self, io_obj, format: io_obj.write(b"safe_data")

    result = strip_exif_metadata(b"fake_data", "test.png")
    assert result == b"safe_data"
