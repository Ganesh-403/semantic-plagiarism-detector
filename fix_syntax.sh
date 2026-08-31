#!/bin/bash
# faiss_index.py fixes
sed -i 's/from src.core.faiss_index_metadata import FAISSIndexMetadatafrom src.core.metrics import faiss_vectors_gauge/from src.core.faiss_index_metadata import FAISSIndexMetadata\nfrom src.core.metrics import faiss_vectors_gauge/g' src/core/faiss_index.py
sed -i 's/def _rebuild_index_from_disk(self) -> None:        if not self.index_path.exists():/def _rebuild_index_from_disk(self) -> None:\n        if not self.index_path.exists():/g' src/core/faiss_index.py
sed -i 's/            self.metadata.save(self.metadata_path)        logger.info(/            self.metadata.save(self.metadata_path)\n        logger.info(/g' src/core/faiss_index.py
sed -i 's/        faiss_vectors_gauge.set(self.ntotal)        return result/        faiss_vectors_gauge.set(self.ntotal)\n        return result/g' src/core/faiss_index.py

# faiss_index_metadata.py fixes
sed -i 's/)                logger.info/)\n                logger.info/g' src/core/faiss_index_metadata.py

# similarity.py fixes
sed -i 's/find_optimal_threshold)                                            find_optimal_threshold)/find_optimal_threshold)/g' src/core/similarity.py
sed -i 's/) -> list\[dict\]:    """Identify document pairs/) -> list[dict]:\n    """Identify document pairs/g' src/core/similarity.py
sed -i 's/if is_plagiarism(score, effective_threshold):            doc_b = doc_names\[j\]/if is_plagiarism(score, effective_threshold):\n            doc_a = doc_names\[i\]\n            doc_b = doc_names\[j\]/g' src/core/similarity.py
sed -i 's/                                "chunk_matches": \[/                "chunk_matches": \[/g' src/core/similarity.py

python3 -c '
with open("src/core/similarity.py", "r") as f:
    lines = f.readlines()
for i in range(885, 922):
    lines[i] = "    " + lines[i]
with open("src/core/similarity.py", "w") as f:
    f.writelines(lines)
'

# corpus_db.py fixes
sed -i 's/    def migration_021_add_corpus_duplicate_detection(/def migration_021_add_corpus_duplicate_detection(/g' src/db/migrations/corpus.py
sed -i 's/    def down_020_add_corpus_duplicate_detection(/def down_020_add_corpus_duplicate_detection(/g' src/db/migrations/corpus.py
sed -i '789,801d' src/db/corpus_db.py
cat << 'EOF2' >> src/db/corpus_db.py

def get_documents_with_embeddings() -> list[str]:
    """Return documents that have persisted chunk embeddings."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT filename
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY filename ASC
            """
        ).fetchall()

    return [row["filename"] for row in rows]
EOF2
sed -i 's/        def get_document_embeddings_for_migration(/def get_document_embeddings_for_migration(/g' src/db/corpus_db.py

# hybrid_scorer.py fixes
sed -i 's/)    else:/)\n    else:/g' src/core/hybrid_scorer.py
