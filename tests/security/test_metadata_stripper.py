import io
import pytest
from PIL import Image
from src.security.metadata_stripper import strip_exif_metadata

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
    img = Image.new('RGB', (10001, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    
    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(img_bytes.getvalue(), "test.png")
    assert "Image dimensions exceed 10,000px safety limit" in str(excinfo.value)

def test_strip_image_metadata_dimension_safety_limit_height():
    # Create a dummy image that exceeds the 10,000px height limit
    img = Image.new('RGB', (100, 10001), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    
    with pytest.raises(ValueError) as excinfo:
        strip_exif_metadata(img_bytes.getvalue(), "test.jpg")
    assert "Image dimensions exceed 10,000px safety limit" in str(excinfo.value)

def test_strip_image_metadata_dimension_exactly_at_limit():
    # Create a dummy image exactly at the 10,000px limit (should pass)
    img = Image.new('RGB', (10000, 10000), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    
    # Should not raise an error
    result = strip_exif_metadata(img_bytes.getvalue(), "test.png")
    assert isinstance(result, bytes)
    assert len(result) > 0


from PIL import Image
import pytest

def test_decompression_bomb_is_rejected(monkeypatch):
    def fake_open(*args, **kwargs):
        raise Image.DecompressionBombError("Bomb detected")

    monkeypatch.setattr(Image, "open", fake_open)

    with pytest.raises(ValueError, match="Image dimensions exceed security safety limits."):
        strip_exif_metadata(b"fake-image-data", "test.png")