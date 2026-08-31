import pytest
from src.core.text_chunking import Chunk, ChunkString, chunk_text, chunk_text_dynamic


def test_chunk_dataclass_initialization_with_position_and_section():
    """Verify Chunk dataclass stores and exposes position and section attributes."""
    chunk = Chunk(
        text="Sample paragraph.",
        page_number=3,
        char_start=150,
        char_end=220,
        section_title="Introduction",
        metadata={"custom_tag": "test"},
    )

    assert chunk.text == "Sample paragraph."
    assert chunk.page_number == 3
    assert chunk.char_start == 150
    assert chunk.char_end == 220
    assert chunk.section_title == "Introduction"
    assert chunk.metadata["page_number"] == 3
    assert chunk.metadata["char_start"] == 150
    assert chunk.metadata["char_end"] == 220
    assert chunk.metadata["section_title"] == "Introduction"
    assert chunk.metadata["custom_tag"] == "test"


def test_chunk_dataclass_defaults():
    """Verify Chunk default fields and backward compatibility with ChunkString."""
    chunk = Chunk(text="Default text.")
    assert chunk.text == "Default text."
    assert chunk.page_number is None
    assert chunk.char_start == 0
    assert chunk.char_end == 0
    assert chunk.section_title is None
    assert isinstance(chunk.metadata, dict)
    assert ChunkString is Chunk


def test_chunk_text_populates_position_fields():
    """Verify chunk_text populates char_start and char_end for generated chunks."""
    text = (
        "First sentence for chunking test. "
        "Second sentence extending length of document to generate multiple chunks. "
        "Third sentence to complete paragraph."
    )
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10, min_words=2)
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.char_start >= 0
        assert c.char_end > c.char_start


def test_chunk_text_dynamic_populates_position_fields():
    """Verify chunk_text_dynamic populates char_start and char_end."""
    text = "Dynamic sentence one. Dynamic sentence two. Dynamic sentence three."
    chunks = chunk_text_dynamic(text, target_size=30, min_overlap=5)
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.char_start >= 0
        assert c.char_end > c.char_start
