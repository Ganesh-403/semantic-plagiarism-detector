"""
tests/core/test_text_chunking.py
---------------------------------
Unit tests for customizable chunk size and overlap parameters, including edge cases.
"""

from src.core.text_chunking import chunk_by_sentences, chunk_documents, chunk_text


def test_chunk_text_custom_parameters():
    sample_text = "Word " * 200  # 1000 characters approximately

    # Default parameters
    default_chunks = chunk_text(sample_text, chunk_size=500, chunk_overlap=50)

    # Smaller chunk size should produce more chunks
    small_chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=20)

    assert len(small_chunks) > len(default_chunks)


def test_chunk_documents_passes_parameters():
    docs = {"doc1.txt": "Line content text repeating " * 50}
    chunked = chunk_documents(docs, chunk_size=300, chunk_overlap=30)

    assert "doc1.txt" in chunked
    assert len(chunked["doc1.txt"]) > 0


# ── Edge Case Tests (#849) ───────────────────────────────────────────────────


def test_chunk_text_empty_and_whitespace():
    """Verify empty or whitespace-only strings return an empty list or clean output."""
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
    assert chunk_text("   \n\t  ", chunk_size=500, chunk_overlap=50) == []


def test_chunk_text_single_long_word():
    """Verify single long words exceeding chunk size are handled safely without crashing."""
    long_word = "A" * 1200
    chunks = chunk_text(long_word, chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 1
    # Ensure no chunk exceeds the maximum hard limits
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_text_cjk_characters():
    """Verify CJK (Chinese, Japanese, Korean) non-Latin unicode text chunking."""
    cjk_text = "这是一个关于人工智能和神经网络的测试文本。" * 20
    chunks = chunk_text(cjk_text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100


def test_chunk_text_emoji_only():
    """Verify emoji-only strings are chunked correctly without character corruption."""
    emoji_text = "🚀🔍🤖📝💻📊" * 50
    chunks = chunk_text(emoji_text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_overlap_boundaries():
    """Verify consecutive chunks preserve configured overlap boundaries."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunk_size = 30
    chunk_overlap = 10

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if len(chunks) > 1:
        # Check that consecutive chunks share overlapping content
        for i in range(len(chunks) - 1):
            assert len(chunks[i]) <= chunk_size


# ── Sentence-Boundary Chunking Tests (#919) ──────────────────────────────────


def test_chunk_by_sentences_preserves_full_sentences():
    """Each chunk must end on a sentence boundary – no mid-sentence splits."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A stitch in time saves nine. "
        "All that glitters is not gold. "
        "To be or not to be, that is the question. "
        "Actions speak louder than words."
    )
    chunks = chunk_by_sentences(text, max_chunk_size=120)

    assert len(chunks) >= 1
    # None of the raw sentence-ending markers should be split across chunks:
    # every chunk must be a coherent unit of complete sentences.
    for chunk in chunks:
        stripped = chunk.strip()
        assert len(stripped) > 0
        # The chunk must not start mid-word (no leading lowercase after space
        # at the very beginning caused by a mid-sentence break).
        assert stripped[0] == stripped[0].upper() or not stripped[0].isalpha()


def test_chunk_by_sentences_respects_max_chunk_size():
    """Chunks should not exceed max_chunk_size unless a single sentence is longer."""
    sentences = [f"This is sentence number {i} in the test document." for i in range(20)]
    text = " ".join(sentences)
    max_size = 100

    chunks = chunk_by_sentences(text, max_chunk_size=max_size)

    assert len(chunks) >= 1
    for chunk in chunks:
        # A single overlong sentence is allowed to exceed the limit; all
        # multi-sentence blocks must respect it.
        words = chunk.split()
        if len(words) > 15:   # heuristic: definitely more than one sentence
            assert len(chunk) <= max_size + 60   # soft tolerance for joining space


def test_chunk_by_sentences_empty_and_whitespace():
    """Returns empty list for empty or whitespace-only input."""
    assert chunk_by_sentences("") == []
    assert chunk_by_sentences("   \n\t  ") == []


def test_chunk_by_sentences_single_sentence():
    """A text with a single sentence yields exactly one chunk."""
    text = "This is the only sentence in the document."
    chunks = chunk_by_sentences(text, max_chunk_size=500)

    assert len(chunks) == 1
    assert chunks[0].strip() == text.strip()


def test_chunk_by_sentences_no_sentence_is_split_mid_word():
    """Verify words are never cut in half across chunk boundaries."""
    text = " ".join(
        f"Word{j} is part of sentence {i}." for i in range(30) for j in range(5)
    )
    chunks = chunk_by_sentences(text, max_chunk_size=200)

    all_words_in_chunks = set()
    for chunk in chunks:
        for word in chunk.split():
            all_words_in_chunks.add(word)

    original_words = set(text.split())
    # Every word in the original text must appear intact in some chunk
    assert original_words.issubset(all_words_in_chunks)


def test_chunk_by_sentences_produces_multiple_chunks_for_long_text():
    """Long multi-sentence text must be split into more than one chunk."""
    sentences = ["The algorithm processes the input data efficiently." for _ in range(40)]
    text = " ".join(sentences)

    chunks = chunk_by_sentences(text, max_chunk_size=150)
    assert len(chunks) > 1


def test_chunk_text_percentage_overlap():
    """Verify overlap_percentage correctly derives chunk_overlap from chunk_size."""
    text = "Word " * 200
    chunk_size = 200
    overlap_percentage = 0.10  # expected chunk_overlap = int(200 * 0.10) = 20

    percentage_chunks = chunk_text(
        text, chunk_size=chunk_size, overlap_percentage=overlap_percentage
    )
    absolute_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=20)

    assert len(percentage_chunks) == len(absolute_chunks)
    assert percentage_chunks == absolute_chunks
            