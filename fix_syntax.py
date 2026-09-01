import os
import re

# faiss_index.py
with open("src/core/faiss_index.py", "r") as f:
    text = f.read()
text = text.replace(
    "from src.core.faiss_index_metadata import FAISSIndexMetadatafrom src.core.metrics import faiss_vectors_gauge",
    "from src.core.faiss_index_metadata import FAISSIndexMetadata\nfrom src.core.metrics import faiss_vectors_gauge"
)
text = text.replace(
    "def _rebuild_index_from_disk(self) -> None:        if not self.index_path.exists():",
    "def _rebuild_index_from_disk(self) -> None:\n        if not self.index_path.exists():"
)
text = text.replace(
    "            self.metadata.save(self.metadata_path)        logger.info(",
    "            self.metadata.save(self.metadata_path)\n        logger.info("
)
text = text.replace(
    "        faiss_vectors_gauge.set(self.ntotal)        return result",
    "        faiss_vectors_gauge.set(self.ntotal)\n        return result"
)
text = text.replace(
    ")        norms = np.linalg.norm(arr, axis=1, keepdims=True)",
    ")\n        norms = np.linalg.norm(arr, axis=1, keepdims=True)"
)
text = text.replace(
    ")        for i, (vec, chunk) in enumerate(zip(emb, chunks)):",
    ")\n        for i, (vec, chunk) in enumerate(zip(emb, chunks)):"
)
text = text.replace(
    "return faiss.IndexFlatIP(active_metadata.dimension), registry    matrix = np.vstack(all_vectors)",
    "return faiss.IndexFlatIP(active_metadata.dimension), registry\n    matrix = np.vstack(all_vectors)"
)
with open("src/core/faiss_index.py", "w") as f:
    f.write(text)

# faiss_index_metadata.py
with open("src/core/faiss_index_metadata.py", "r") as f:
    text = f.read()
text = text.replace(")                logger.info", ")\n                logger.info")
with open("src/core/faiss_index_metadata.py", "w") as f:
    f.write(text)

# similarity.py
with open("src/core/similarity.py", "r") as f:
    text = f.read()
text = text.replace(
    "find_optimal_threshold)                                            find_optimal_threshold)",
    "find_optimal_threshold)"
)
text = text.replace(
    ") -> list[dict]:    \"\"\"Identify document pairs",
    ") -> list[dict]:\n    \"\"\"Identify document pairs"
)
text = text.replace(
    "if is_plagiarism(score, effective_threshold):            doc_b = doc_names[j]",
    "if is_plagiarism(score, effective_threshold):\n            doc_a = doc_names[i]\n            doc_b = doc_names[j]"
)
text = text.replace(
    "                                \"chunk_matches\": [",
    "                \"chunk_matches\": ["
)
with open("src/core/similarity.py", "w") as f:
    f.write(text)

with open("src/core/similarity.py", "r") as f:
    lines = f.readlines()
for i in range(885, 922):
    if len(lines) > i and not lines[i].startswith("    "):
        lines[i] = "    " + lines[i]
with open("src/core/similarity.py", "w") as f:
    f.writelines(lines)

# hybrid_scorer.py
with open("src/core/hybrid_scorer.py", "r") as f:
    text = f.read()
text = text.replace(")    else:", ")\n    else:")
with open("src/core/hybrid_scorer.py", "w") as f:
    f.write(text)

# corpus.py
with open("src/db/migrations/corpus.py", "r") as f:
    text = f.read()
text = text.replace("    def migration_021_add_corpus_duplicate_detection(", "def migration_021_add_corpus_duplicate_detection(")
text = text.replace("    def down_020_add_corpus_duplicate_detection(", "def down_020_add_corpus_duplicate_detection(")
with open("src/db/migrations/corpus.py", "w") as f:
    f.write(text)

# corpus_db.py
with open("src/db/corpus_db.py", "r") as f:
    lines = f.readlines()
with open("src/db/corpus_db.py", "w") as f:
    for i, line in enumerate(lines):
        if 788 <= i <= 800:
            continue
        f.write(line)
with open("src/db/corpus_db.py", "r") as f:
    text = f.read()
text = text.replace("        def get_document_embeddings_for_migration(", "def get_document_embeddings_for_migration(")
with open("src/db/corpus_db.py", "w") as f:
    f.write(text)

with open("src/db/corpus_db.py", "a") as f:
    f.write("\n\ndef get_documents_with_embeddings() -> list[str]:\n")
    f.write("    with _connect() as conn:\n")
    f.write("        rows = conn.execute(\n")
    f.write('            \"\"\"\n')
    f.write("            SELECT DISTINCT filename\n")
    f.write("            FROM chunks\n")
    f.write("            WHERE embedding IS NOT NULL\n")
    f.write("            ORDER BY filename ASC\n")
    f.write('            \"\"\"\n')
    f.write("        ).fetchall()\n")
    f.write("    return [row[\"filename\"] for row in rows]\n")

