"""
src/core/faiss_index_metadata.py
---------------------------------
Maintains mapping between FAISS vector IDs and document/chunk records,
enabling incremental index updates without full rebuild (Issue #3913).

Metadata is persisted as JSON alongside the FAISS index on disk.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class VectorMapping:
    """Maps a FAISS vector ID to its source document/chunk."""

    vector_id: int
    doc_name: str
    chunk_index: int
    embedding_text: str
    timestamp: str


@dataclass
class IndexMetadata:
    """State, embedding schema, and version information for a FAISS index."""

    index_id: str
    created_at: str
    last_updated: str
    total_vectors: int
    vector_mappings: Dict[int, Dict[str, Any]]
    deleted_vector_ids: List[int]
    corpus_hash: Optional[str]

    embedding_model_identifier: Optional[str] = None
    embedding_model_version: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_normalization_strategy: Optional[str] = None
    vector_schema_version: Optional[int] = None

class FAISSIndexMetadata:
    """
    Manages FAISS vector-to-document mappings and index state.
    
    Enables incremental operations: add, update, delete vectors without
    rebuilding the entire index.
    """
    def set_embedding_metadata(self, metadata: Any) -> None:
        """Attach the embedding schema used by this FAISS index."""
        if self.metadata is None:
            self._init_new()

        self.metadata.embedding_model_identifier = metadata.model_identifier
        self.metadata.embedding_model_version = metadata.model_version
        self.metadata.embedding_dimension = metadata.dimension
        self.metadata.embedding_normalization_strategy = (
            metadata.normalization_strategy
        )
        self.metadata.vector_schema_version = metadata.vector_schema_version
    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize metadata manager.
        
        Args:
            metadata_path: Path to faiss_index_metadata.json. If None, uses default.
        """
        if metadata_path is None:
            from src.core.config import FAISS_INDEX_METADATA_PATH
            metadata_path = FAISS_INDEX_METADATA_PATH
        
        self.metadata_path = metadata_path
        self.metadata: Optional[IndexMetadata] = None
        self.load()

    def load(self) -> bool:
        """
        Load metadata from disk, or initialize new if doesn't exist.
        
        Returns:
            True if loaded from disk, False if newly initialized.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.metadata = IndexMetadata(
                    index_id=data.get("index_id", "default"),
                    created_at=data.get("created_at", datetime.now().isoformat()),
                    last_updated=data.get("last_updated", datetime.now().isoformat()),
                    total_vectors=data.get("total_vectors", 0),
                    vector_mappings={
                        int(k): v for k, v in data.get("vector_mappings", {}).items()
                    },
                    deleted_vector_ids=data.get("deleted_vector_ids", []),
                    corpus_hash=data.get("corpus_hash"),
                    embedding_model_identifier=data.get(
                        "embedding_model_identifier"
                    ),
                    embedding_model_version=data.get(
                        "embedding_model_version"
                    ),
                    embedding_dimension=data.get("embedding_dimension"),
                    embedding_normalization_strategy=data.get(
                        "embedding_normalization_strategy"
                    ),
                    vector_schema_version=data.get(
                        "vector_schema_version"
                    ),
                )
                logger.info("Loaded FAISS index metadata from %s", self.metadata_path)
                return True
            except Exception as e:
                logger.warning("Failed to load metadata: %s. Starting fresh.", e)
                self._init_new()
                return False
        else:
            self._init_new()
            return False

    def _init_new(self) -> None:
        """Initialize new metadata."""
        self.metadata = IndexMetadata(
            index_id=f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_vectors=0,
            vector_mappings={},
            deleted_vector_ids=[],
                    corpus_hash=data.get("corpus_hash"),
                    embedding_model_identifier=data.get(
                        "embedding_model_identifier"
                    ),
                    embedding_model_version=data.get(
                        "embedding_model_version"
                    ),
                    embedding_dimension=data.get("embedding_dimension"),
                    embedding_normalization_strategy=data.get(
                        "embedding_normalization_strategy"
                    ),
                    vector_schema_version=data.get(
                        "vector_schema_version"
                    ),
                )
    def save(self) -> str:
        """
        Persist metadata to disk.
        
        Returns:
            Path where metadata was saved.
        """
        if self.metadata is None:
            self._init_new()

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        data = {
            "index_id": self.metadata.index_id,
            "created_at": self.metadata.created_at,
            "last_updated": datetime.now().isoformat(),
            "total_vectors": self.metadata.total_vectors,
            "vector_mappings": self.metadata.vector_mappings,
            "deleted_vector_ids": self.metadata.deleted_vector_ids,
            "corpus_hash": self.metadata.corpus_hash,
            "embedding_model_identifier": (
                self.metadata.embedding_model_identifier
            ),
            "embedding_model_version": self.metadata.embedding_model_version,
            "embedding_dimension": self.metadata.embedding_dimension,
            "embedding_normalization_strategy": (
                self.metadata.embedding_normalization_strategy
            ),
            "vector_schema_version": self.metadata.vector_schema_version,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved FAISS index metadata to %s", self.metadata_path)
        return self.metadata_path

    def add_vector(
        self,
        vector_id: int,
        doc_name: str,
        chunk_index: int,
        embedding_text: str,
    ) -> None:
        """
        Register a new vector in the index.
        
        Args:
            vector_id: FAISS vector ID (assigned by index).
            doc_name: Name of source document.
            chunk_index: Chunk index within document.
            embedding_text: The text that was embedded.
        """
        if self.metadata is None:
            self._init_new()

        self.metadata.vector_mappings[vector_id] = {
            "doc_name": doc_name,
            "chunk_index": chunk_index,
            "embedding_text": embedding_text,
            "timestamp": datetime.now().isoformat(),
        }
        self.metadata.total_vectors = len(self.metadata.vector_mappings)

    def remove_vector(self, vector_id: int) -> None:
        """
        Mark a vector as deleted (soft delete for consistency).
        
        Args:
            vector_id: FAISS vector ID to remove.
        """
        if self.metadata is None:
            return

        if vector_id in self.metadata.vector_mappings:
            del self.metadata.vector_mappings[vector_id]
        
        if vector_id not in self.metadata.deleted_vector_ids:
            self.metadata.deleted_vector_ids.append(vector_id)
        
        self.metadata.total_vectors = len(self.metadata.vector_mappings)

    def update_vector(
        self,
        vector_id: int,
        embedding_text: str,
    ) -> None:
        """
        Update embedding text for an existing vector.
        
        Args:
            vector_id: FAISS vector ID.
            embedding_text: New embedding text.
        """
        if self.metadata is None or vector_id not in self.metadata.vector_mappings:
            logger.warning("Vector %d not found in metadata", vector_id)
            return

        self.metadata.vector_mappings[vector_id]["embedding_text"] = embedding_text
        self.metadata.vector_mappings[vector_id]["timestamp"] = datetime.now().isoformat()

    def get_vector_mapping(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Look up mapping for a single vector.
        
        Args:
            vector_id: FAISS vector ID.
        
        Returns:
            Mapping dict or None if not found.
        """
        if self.metadata is None:
            return None
        return self.metadata.vector_mappings.get(vector_id)

    def get_vectors_for_document(self, doc_name: str) -> List[int]:
        """
        Find all vector IDs belonging to a document.
        
        Args:
            doc_name: Document name.
        
        Returns:
            List of vector IDs.
        """
        if self.metadata is None:
            return []

        return [
            vid
            for vid, mapping in self.metadata.vector_mappings.items()
            if mapping.get("doc_name") == doc_name
        ]

    def validate_consistency(self, index_size: int) -> bool:
        """
        Check if metadata matches actual index size.
        
        Args:
            index_size: Number of vectors in FAISS index.
        
        Returns:
            True if consistent, False if mismatch.
        """
        if self.metadata is None:
            return True

        expected_size = len(self.metadata.vector_mappings)
        if expected_size != index_size:
            logger.warning(
                "Index/metadata mismatch: index has %d vectors, metadata tracks %d",
                index_size,
                expected_size,
            )
            return False

        return True

    def reset(self) -> None:
        """Clear all metadata (used before full rebuild)."""
        self._init_new()
        self.metadata.vector_mappings.clear()
        self.metadata.deleted_vector_ids.clear()
        self.metadata.total_vectors = 0
        logger.info("Reset FAISS index metadata")