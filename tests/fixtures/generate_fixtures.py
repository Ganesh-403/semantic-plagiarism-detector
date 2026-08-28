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

import os

import fitz
from PIL import Image, ImageDraw


def create_clean_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 50),
        "Artificial Intelligence and Machine Learning are transforming modern higher education. This is a clean PDF with selectable text.",
        fontsize=11,
    )
    doc.save("tests/fixtures/clean.pdf")
    doc.close()


def create_encrypted_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 50),
        "Artificial Intelligence and Machine Learning are transforming modern higher education. This is an encrypted PDF.",
        fontsize=11,
    )
    # Using simple encryption
    doc.save(
        "tests/fixtures/encrypted.pdf",
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="secret",
    )
    doc.close()


def create_scanned_pdf():
    # Make image large enough for OCR
    img = Image.new("RGB", (800, 400), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    text = "Artificial Intelligence and Machine Learning are transforming modern higher education."
    # Draw text multiple times to ensure sufficient words for MIN_NATIVE_WORDS_PER_PAGE bypassing
    d.text((50, 50), text, fill=(0, 0, 0))
    d.text((50, 100), "This is scanned text on an image.", fill=(0, 0, 0))
    d.text((50, 150), text, fill=(0, 0, 0))
    img.save("tests/fixtures/scanned_temp.png")

    doc = fitz.open()
    page = doc.new_page()
    # Insert image at top-left
    page.insert_image(
        fitz.Rect(0, 0, 800, 400), filename="tests/fixtures/scanned_temp.png"
    )
    doc.save("tests/fixtures/scanned.pdf")
    doc.close()

    os.remove("tests/fixtures/scanned_temp.png")


if __name__ == "__main__":
    create_clean_pdf()
    create_encrypted_pdf()
    create_scanned_pdf()
    print("Fixtures generated.")
