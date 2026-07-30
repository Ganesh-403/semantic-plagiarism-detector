import io
import docx
from src.core.document_parser import extract_text_from_docx
from src.core.text_chunking import chunk_text

def test_docx_headings_extraction_and_chunking():
    # Create an in-memory DOCX file
    doc = docx.Document()
    doc.add_paragraph("Heading 1 Title", style="Heading 1")
    doc.add_paragraph("This is the content of the first section. It has some text to chunk.")
    doc.add_paragraph("Heading 2 Title", style="Heading 2")
    doc.add_paragraph("This is the content of the second section. More text for chunking.")

    # Save to bytes
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_bytes = file_stream.getvalue()

    # Extract text
    parsed_text = extract_text_from_docx(file_bytes)
    assert hasattr(parsed_text, "word_headings")
    
    # Chunk text
    chunks = chunk_text(parsed_text, chunk_size=30, chunk_overlap=5)
    
    assert len(chunks) >= 2
    
    # First chunk inherits the correct section title (Heading 1 Title)
    assert chunks[0].metadata.get("section_title") == "Heading 1 Title"
    
    # Heading changes updating subsequent chunks (Heading 2 Title)
    found_heading_2 = False
    for chunk in chunks:
        if "second section" in chunk:
            assert chunk.metadata.get("section_title") == "Heading 2 Title"
            found_heading_2 = True
    assert found_heading_2

def test_docx_no_headings():
    doc = docx.Document()
    doc.add_paragraph("Just normal text here.")
    doc.add_paragraph("No headings at all.")

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_bytes = file_stream.getvalue()

    parsed_text = extract_text_from_docx(file_bytes)
    chunks = chunk_text(parsed_text, chunk_size=30, chunk_overlap=5)
    
    for chunk in chunks:
        assert not chunk.metadata or chunk.metadata.get("section_title") is None

def test_non_docx_no_headings():
    # Test simple plain text string
    normal_text = "This is a simple plain text string that has no docx structure."
    chunks = chunk_text(normal_text, chunk_size=30, chunk_overlap=5)
    for chunk in chunks:
        assert not hasattr(chunk, "metadata") or chunk.metadata.get("section_title") is None
