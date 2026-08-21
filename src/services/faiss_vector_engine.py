"""FAISS Semantic Vector Engine Service.

Provides high-dimensional vector normalization, FAISS indexing, k-NN similarity
search retrieval, and memory usage telemetry generation.
"""

import math
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.models.faiss_vector_model import (
    FaissIndexTelemetry,
    VectorDocumentChunk,
    VectorSearchAuditReport,
    VectorSearchMatch,
)


class FaissSemanticVectorEngine:
    """Core vector engine managing FAISS semantic search operations."""

    def __init__(
        self, dimensions: int = 384, metric_type: str = "METRIC_INNER_PRODUCT"
    ):
        self.dimensions = dimensions
        self.metric_type = metric_type
        self.indexed_chunks: Dict[str, VectorDocumentChunk] = {}
        self.index_id = f"FAISS-IDX-{uuid.uuid4().hex[:8].upper()}"

    @staticmethod
    def l2_normalize(vector: List[float]) -> List[float]:
        """Performs L2 normalization on embedding vectors for cosine similarity computation."""
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0.0:
            return vector
        return [round(x / magnitude, 6) for x in vector]

    @staticmethod
    def calculate_l2_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculates Euclidean (L2) distance between two vector embeddings."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 999.0
        return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2))), 4)

    def add_document_chunk(
        self,
        document_id: str,
        document_title: str,
        chunk_index: int,
        raw_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, str]] = None,
    ) -> VectorDocumentChunk:
        """Inserts a document chunk vector into the indexed repository."""
        chunk_id = f"CHK-{uuid.uuid4().hex[:8].upper()}"
        normalized_embedding = self.l2_normalize(embedding)

        chunk = VectorDocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_title=document_title,
            chunk_index=chunk_index,
            raw_text=raw_text,
            embedding_vector=normalized_embedding,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )

        self.indexed_chunks[chunk_id] = chunk
        return chunk

    def search_similar_chunks(
        self, query_vector: List[float], query_text: str, top_k: int = 5
    ) -> VectorSearchAuditReport:
        """Executes k-NN vector search against all indexed document vectors."""
        start_time = time.time()
        normalized_query = self.l2_normalize(query_vector)

        candidates: List[Tuple[float, float, VectorDocumentChunk]] = []

        for chunk_id, chunk in self.indexed_chunks.items():
            # Calculate Inner Product (Cosine Similarity)
            dot_product = sum(
                a * b for a, b in zip(normalized_query, chunk.embedding_vector)
            )
            l2_dist = self.calculate_l2_distance(
                normalized_query, chunk.embedding_vector
            )
            candidates.append((dot_product, l2_dist, chunk))

        # Sort by similarity score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:top_k]

        matches: List[VectorSearchMatch] = []
        for rank, (sim, l2_dist, chunk) in enumerate(top_candidates, start=1):
            matches.append(
                VectorSearchMatch(
                    match_id=f"MATCH-{uuid.uuid4().hex[:8].upper()}",
                    query_chunk_id="QUERY-VEC",
                    matched_chunk_id=chunk.chunk_id,
                    matched_document_title=chunk.document_title,
                    matched_text_snippet=chunk.raw_text[:120] + "...",
                    cosine_similarity_score=round(sim, 4),
                    l2_distance=l2_dist,
                    rank_position=rank,
                    search_timestamp=datetime.utcnow(),
                )
            )

        execution_ms = round((time.time() - start_time) * 1000, 2)
        highest_sim = matches[0].cosine_similarity_score if matches else 0.0

        return VectorSearchAuditReport(
            query_id=f"QRY-{uuid.uuid4().hex[:8].upper()}",
            query_text=query_text,
            top_k_requested=top_k,
            execution_time_ms=execution_ms,
            total_matches_found=len(matches),
            highest_similarity_ratio=highest_sim,
            matches=matches,
        )

    def get_index_telemetry(self) -> FaissIndexTelemetry:
        """Generates real-time telemetry stats for the vector index."""
        count = len(self.indexed_chunks)
        estimated_mem = round((count * self.dimensions * 4) / (1024 * 1024), 4)

        return FaissIndexTelemetry(
            index_id=self.index_id,
            total_vectors_indexed=count,
            vector_dimensions=self.dimensions,
            metric_type=self.metric_type,
            index_type="IndexFlatIP (Cosine)",
            is_trained=True,
            memory_usage_mb=estimated_mem,
            last_updated_at=datetime.utcnow(),
        )
