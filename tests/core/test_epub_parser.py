"""Real EPUB archive regression tests, including ordered chapters and hostile input."""
import io
import zipfile

import pytest

from src.core.document_parser import extract_text, extract_text_from_epub
from src.utils.epub_reader import extract_epub_text


def epub_bytes(href="one.xhtml", content=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>')
        z.writestr("OPS/book.opf", f'<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="two" href="two.xhtml" media-type="application/xhtml+xml"/><item id="one" href="{href}" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="one"/><itemref idref="two"/></spine></package>')
        z.writestr("OPS/one.xhtml", content or '<html><body><h1>Chapter 1</h1><p>This is the first chapter text.</p><script>secret_script</script><style>font-size:12px</style></body></html>')
        z.writestr("OPS/two.xhtml", '<html><body><h2>Chapter 2</h2><p>Second chapter with <em>formatted</em> text.</p></body></html>')
    return buffer.getvalue()


def test_extract_text_from_epub_clean_chapters():
    text = extract_text_from_epub(epub_bytes())
    assert text == "Chapter 1 This is the first chapter text.\n\nChapter 2 Second chapter with formatted text."


def test_extract_text_from_epub_handles_invalid_or_corrupt_files():
    assert extract_text_from_epub(b"corrupt zip data") == ""


def test_extract_text_pipeline_epub_dispatch():
    assert "first chapter" in extract_text(epub_bytes(), "book.epub")


@pytest.mark.parametrize("href", ["../../secret.txt", "/secret", "https://example.com/data"])
def test_rejects_unsafe_content_paths(href):
    with pytest.raises(ValueError):
        extract_epub_text(epub_bytes(href=href))


def test_rejects_excessive_expansion(monkeypatch):
    monkeypatch.setattr("src.utils.epub_reader.MAX_EXPANDED_BYTES", 100)
    with pytest.raises(ValueError, match="expanded size"):
        extract_epub_text(epub_bytes())


def test_reads_filelike_without_extracting_files():
    assert "Chapter 2" in extract_epub_text(io.BytesIO(epub_bytes()))
