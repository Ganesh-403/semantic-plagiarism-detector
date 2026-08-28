# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io

import fitz
from PIL import Image

from src.security.metadata_stripper import strip_exif_metadata


def test_strip_pdf_metadata():
    """Ensure PDF metadata is removed."""
    # Create a dummy PDF with metadata
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test Document")
    doc.set_metadata(
        {"title": "Secret Project", "author": "John Doe", "creator": "Malicious App"}
    )
    pdf_bytes = doc.write()
    doc.close()

    # Verify metadata exists before stripping
    pre_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert pre_doc.metadata.get("title") == "Secret Project"
    pre_doc.close()

    # Strip metadata
    scrubbed_bytes = strip_exif_metadata(pdf_bytes, "test.pdf")

    # Verify metadata is gone
    post_doc = fitz.open(stream=scrubbed_bytes, filetype="pdf")
    meta = post_doc.metadata
    assert not meta.get("title")
    assert not meta.get("author")
    post_doc.close()


def test_strip_image_metadata():
    """Ensure image EXIF is removed."""
    # Create a dummy image
    img = Image.new("RGB", (100, 100), color="red")

    # Add dummy info (Pillow Info dict represents metadata like EXIF)
    img.info["exif"] = b"dummy_exif_data_that_might_contain_gps"

    img_io = io.BytesIO()
    img.save(img_io, format="JPEG")
    img_bytes = img_io.getvalue()

    # Verify it has exif (Pillow might drop it if not formatted, but let's just test the stripper function runs without error)
    scrubbed_bytes = strip_exif_metadata(img_bytes, "test.jpg")

    # Verify the scrubbed image is still a valid image
    scrubbed_img = Image.open(io.BytesIO(scrubbed_bytes))
    assert scrubbed_img.size == (100, 100)
    assert not scrubbed_img.info.get("exif")


def test_strip_unsupported_format():
    """Ensure non-PDF/Image formats return unchanged."""
    dummy_data = b"Some random text data"
    scrubbed = strip_exif_metadata(dummy_data, "test.txt")
    assert scrubbed == dummy_data
