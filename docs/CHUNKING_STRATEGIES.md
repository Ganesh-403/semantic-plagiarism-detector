# Chunking Strategies

`src/core/text_chunking.py` provides three distinct strategies for splitting a
document into smaller pieces before embedding. Each strategy makes a different
trade-off between speed, sentence integrity, and chunk-size consistency. This
guide explains what each strategy does, when to reach for it, and how to tune
its parameters.

All three strategies return a list of `ChunkString` objects (a plain `str`
subclass that can carry optional metadata).

---

## 1. `chunk_text` - fixed-size chunking with sentence-aware padding

**Purpose:** The default, general-purpose chunker. Splits text into
fixed-length character windows, but (by default) extends each window's
boundary out to the nearest sentence terminator so chunks don't cut off
mid-sentence.

**Signature:**

```python
chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_words: int = 5,
    overlap_percentage: float | None = None,
    max_chunks: int = 1000,
    sentence_padding: bool = True,
) -> List[ChunkString]
```

**Key parameters:**

- `chunk_size` - target character length per chunk. Enforced minimum of 50
  characters (`MIN_CHUNK_SIZE`); smaller values are clamped up with a warning.
- `chunk_overlap` - number of characters shared between consecutive chunks,
  used to avoid losing context that straddles a chunk boundary. Must be
  strictly less than `chunk_size`.
- `overlap_percentage` - if set, overrides `chunk_overlap` as a fraction of
  `chunk_size` (e.g. `0.1` = 10% overlap).
- `min_words` - chunks with fewer words than this are dropped, filtering out
  noise like headers or page numbers.
- `max_chunks` - hard cap on the number of chunks generated per document, to
  bound memory use on very large inputs. Logged as a warning if hit.
- `sentence_padding` - when `True` (default), chunk boundaries are pushed
  outward to the nearest sentence boundary (searching up to 100 characters
  forward, 50 backward) so chunks end on `.`, `!`, or `?`. When `False`, the
  function falls back to plain word-boundary chunking (splits on whitespace,
  no sentence awareness).
- Falls back to raw character-based chunking (`_character_fallback_chunking`)
  for CJK text, emoji-heavy text, or single unbroken tokens where word/sentence
  boundaries don't apply.

**When to use:** Default choice for most documents. Good balance of
predictable chunk size (for consistent embedding batch shapes) and semantic
coherence via sentence padding.

**Pros:**

- Predictable, tunable chunk size - useful when you need consistent memory/
  compute per chunk.
- Sentence padding avoids truncating mid-sentence, improving embedding quality.
- Overlap preserves context across chunk boundaries.
- Robust fallback path for non-space-delimited or degenerate text.

**Cons:**

- Chunk size is only a soft target once sentence padding is enabled - actual
  chunk length can grow up to `2 × chunk_size` in the worst case (hard-capped).
- More parameters to tune than the other two strategies.

---

## 2. `chunk_by_sentences` - sentence-boundary-aware chunking

**Purpose:** Groups whole sentences into blocks, guaranteeing that no sentence
is ever split across two chunks. Uses NLTK's `sent_tokenize` when available,
falling back to a regex-based sentence splitter otherwise.

**Signature:**

```python
chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500,
    min_sentences: int = 1,
    min_words: int = 3,
) -> List[str]
```

**Key parameters:**

- `max_chunk_size` - soft character limit per chunk. It's a *soft* limit: a
  single sentence longer than `max_chunk_size` is still emitted as its own
  chunk rather than being truncated or dropped.
- `min_sentences` - minimum number of sentences required before a block is
  flushed. Trailing blocks below this threshold are still emitted (to avoid
  losing content) as long as they meet `min_words`.
- `min_words` - minimum word count to keep a chunk, filtering degenerate
  fragments (e.g. a lone page number that tokenizes as a "sentence").

**When to use:** Best when sentence integrity matters more than exact chunk
size - for example, when downstream analysis (citation extraction, semantic
alignment) depends on complete grammatical units, or when documents have
highly variable sentence lengths that would be awkwardly split by a
fixed-size window.

**Pros:**

- Never splits a sentence mid-clause.
- Simple mental model - one or more complete sentences per chunk.
- No overlap logic to tune.

**Cons:**

- No overlap between chunks, so context that spans a chunk boundary can be
  lost.
- Chunk sizes are less uniform - a chunk containing several short sentences
  will be much smaller than one containing a single long sentence.
- Depends on NLTK's `punkt`/`punkt_tab` corpus for best sentence-boundary
  accuracy (regex fallback is less precise on abbreviations, quotes, etc.).

---

## 3. `chunk_text_dynamic` - sliding window with adaptive boundary snapping

**Purpose:** A sliding-window chunker that aims for a target size but snaps
the actual boundary to the nearest sentence-ending punctuation within a
tolerance margin, balancing uniform chunk size against sentence integrity.

**Signature:**

```python
chunk_text_dynamic(
    text: str,
    target_size: int = 500,
    min_overlap: int = 50,
) -> List[str]
```

**Key parameters:**

- `target_size` - target character length per chunk.
- `min_overlap` - minimum character overlap enforced between consecutive
  chunks.
- Internally, a search margin of 20% of `target_size` is used: the algorithm
  looks for `.`, `!`, or `?` within `target_size ± 20%` and snaps to whichever
  punctuation mark lands closest to `target_size`. If no punctuation is found
  in that window, it falls back to a hard cut at `target_size`.

**When to use:** A middle ground between `chunk_text` and
`chunk_by_sentences` - use it when you want chunk sizes to stay close to a
target value (more uniform than `chunk_by_sentences`) while still preferring
sentence-aligned cuts (more semantically coherent than a hard fixed-size cut).
Well suited to large documents where you need consistent chunk counts for
batching or progress estimation.

**Pros:**

- More size-uniform than `chunk_by_sentences`, since it always tries to land
  near `target_size`.
- Still prefers sentence boundaries over arbitrary character cuts, unlike
  disabling `sentence_padding` in `chunk_text`.
- Simple two-parameter interface.

**Cons:**

- Boundary search only looks for punctuation, not full NLTK sentence
  tokenization - less robust than `chunk_by_sentences` on abbreviations
  (e.g. "Dr.", "U.S.") since any `.`/`!`/`?` character counts as a candidate
  boundary.
- No `min_words` filtering - very short trailing chunks are not dropped.

---

## Comparison Table

| Strategy             | Sentence-Aware                        | Overlap                          | Speed     | Best For                                                                 |
|-----------------------|----------------------------------------|-----------------------------------|-----------|---------------------------------------------------------------------------|
| `chunk_text`          | Yes (optional, via `sentence_padding`) | Yes (`chunk_overlap`, tunable)    | Fast      | General-purpose default; consistent chunk size with context overlap.     |
| `chunk_by_sentences`  | Yes (strict — never splits a sentence) | No                                | Fast      | Preserving grammatical units; citation/alignment tasks; variable-length documents. |
| `chunk_text_dynamic`  | Partial (punctuation-snap, not full NLTK tokenization) | Yes (`min_overlap`, tunable) | Fast      | Large documents needing near-uniform chunk sizes with lightweight sentence awareness. |

---

## Choosing a Strategy: Quick Guide

- **Not sure? Use `chunk_text`.** It's the default used by `chunk_documents`
  and covers the common case well.
- **Need guaranteed complete sentences?** Use `chunk_by_sentences`.
- **Need chunk counts/sizes to stay predictable across a large batch of
  documents (e.g. for progress bars or fixed-size embedding batches)?** Use
  `chunk_text_dynamic`.
- **Working with CJK text, emoji-heavy text, or text without clear word
  boundaries?** `chunk_text` will automatically fall back to character-based
  chunking; the other two strategies rely on sentence tokenization and may
  behave unpredictably on such input.

See also: [ALGORITHMS.md](ALGORITHMS.md) for how chunk embeddings feed into
the semantic and hybrid similarity scoring pipeline.
