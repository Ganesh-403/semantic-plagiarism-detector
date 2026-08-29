## Cross-Lingual Back-Translation Pipeline

To detect plagiarism that has been translated from one language into another (a common evasion technique), the system supports a cross-lingual detection workflow. Non-English text is first identified, checked against a translation cache to avoid redundant API/model calls, back-translated into the corpus's reference language, and then run through the standard embedding and FAISS similarity search pipeline.

The following Mermaid flowchart illustrates this pipeline:

```mermaid
flowchart LR
    A[Input Document Chunk] --> B[Language Detection]
    B -->|Non-reference language| C{Cache Check (translation_cache)}
    B -->|Reference language| E[Embedding Generation]
    C -->|Cache hit| D[Cached Translation (SQLite translation_cache)]
    C -->|Cache miss| F[Back-Translation]
    F --> G[Store Translation in Cache (SQLite translation_cache)]
    G --> D
    D --> E
    E --> H[FAISS Similarity Search]
    H --> I[Candidate Matches & Scoring]
```

**Notes on cache and implementation:**

1. **Translation cache (SQLite)** — The translation cache used by the back-translation flow is the SQLite-backed translation cache implemented at `src/db/translation_cache.py` (see functions like `get_cached_translation` and `save_translation`). A migration helper exists at `scripts/migrate_translation_cache.py` for moving legacy rows into the modern cache schema.

2. **Why SQLite?** — The current back-translation pipeline uses a local SQLite DB for durable, queryable translation lookups keyed by a hash of the source text and target language. This is resilient for the offline and single-node deployment scenarios and supports the repository's LRU/last-accessed bookkeeping.

3. **Redis is used elsewhere, not for translation cache** — The repository also contains `src/utils/redis_cache.py` which provides Redis-based utilities (session state, FAISS result caching, and other scalable caches). The translation cache described above is distinct from those Redis utilities; the back-translation pipeline uses the SQLite translation cache, not Redis.

4. **Workflow steps:**

    1. **Language Detection**: Each document chunk is analyzed to determine its source language before further processing.
    2. **Cache Check**: If the chunk's language differs from the corpus reference language, the system checks the SQLite translation cache for a previously computed translation, keyed by a hash of the chunk text and target language.
    3. **Translation (Back-Translation)**: On a cache miss, the chunk is back-translated into the reference language using the configured translation backend, and the result is stored in the translation cache for future reuse.
    4. **Embedding**: The (translated or original) text is passed through the `SentenceTransformer` embedding model to produce a semantic vector, identical to the standard pipeline.
    5. **FAISS Search**: The resulting embedding is compared against the FAISS index to retrieve candidate matches, which then flow into the standard similarity scoring pipeline.

This design ensures that cross-lingual plagiarism is detected with the same accuracy as same-language plagiarism, while the cache layer keeps repeated translations fast and cost-efficient.
