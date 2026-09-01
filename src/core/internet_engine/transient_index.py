"""
Transient in-memory index for internet scraping results.
Isolates internet search results from the persistent corpus FAISS index.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except ImportError:
    faiss = None

from src.core.internet_engine.models import FetchedSource, TransientIndexConfig, TransientMatch
from src.core.text_chunking import chunk_documents
from src.core.embedding_model import embed_documents

logger = logging.getLogger(__name__)

class TransientInternetIndex:
    """
    Manages an isolated, temporary FAISS index for comparing a single 
    submitted document against internet search results.
    """
    
    def __init__(self, config: TransientIndexConfig):
        self.config = config
        self.index = None
        self.registry: List[Dict] = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2 used in embedding_model.py
        
    def build_index(self, sources: List[FetchedSource]) -> None:
        """Chunks fetched sources, generates embeddings, and builds the transient index."""
        if faiss is None:
            logger.error("FAISS is not installed. Cannot build transient index.")
            return

        valid_sources = [s for s in sources if s.is_valid and s.content]
        if not valid_sources:
            logger.warning("No valid internet sources to index.")
            return

        # Prepare raw texts dict for the chunking utility
        raw_texts = {s.url: s.content for s in valid_sources}
        
        # 1. Chunk documents
        chunked_docs = chunk_documents(
            raw_texts, 
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
        
        # 2. Embed chunks
        # embed_documents expects Dict[str, List[str]] and returns Dict[str, np.ndarray]
        embeddings = embed_documents(chunked_docs)
        
        # 3. Flatten into FAISS registry
        self.index = faiss.IndexFlatIP(self.dimension)
        self.registry = []
        
        vectors = []
        for url, chunks in chunked_docs.items():
            emb_matrix = embeddings.get(url)
            if emb_matrix is not None and len(emb_matrix) > 0:
                for i, chunk_text in enumerate(chunks):
                    vectors.append(emb_matrix[i])
                    self.registry.append({
                        "url": url,
                        "chunk_idx": i,
                        "text": chunk_text
                    })
                    
        if vectors:
            vector_matrix = np.vstack(vectors).astype('float32')
            self.index.add(vector_matrix)
            logger.info(f"Built transient FAISS index with {len(self.registry)} chunks.")

    def search(self, submitted_doc_id: str, submitted_text: str) -> List[TransientMatch]:
        """
        Compares the submitted document against the transient internet index.
        Returns a list of TransientMatches that exceed the similarity threshold.
        """
        if self.index is None or not self.registry or not submitted_text:
            return []
            
        # 1. Chunk and embed the submitted document
        chunked = chunk_documents(
            {submitted_doc_id: submitted_text}, 
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
        submitted_chunks = chunked.get(submitted_doc_id, [])
        if not submitted_chunks:
            return []
            
        embeddings = embed_documents(chunked)
        query_vectors = embeddings.get(submitted_doc_id)
        if query_vectors is None or len(query_vectors) == 0:
            return []
            
        # 2. Search the transient FAISS index
        k = 3  # Top 3 matches per submitted chunk
        distances, indices = self.index.search(query_vectors.astype('float32'), k)
        
        matches = []
        seen_pairs = set()
        
        # 3. Process results and filter by threshold
        for q_idx, query_chunk in enumerate(submitted_chunks):
            for i, rank_idx in enumerate(indices[q_idx]):
                if rank_idx == -1 or rank_idx >= len(self.registry):
                    continue
                    
                score = float(distances[q_idx][i])
                if score >= self.config.similarity_threshold:
                    target = self.registry[rank_idx]
                    url = target["url"]
                    
                    # Deduplicate: only keep the highest scoring match between the submitted doc and the URL
                    # in a real system we might keep all chunk matches, but for report clarity we keep unique URLs
                    pair_key = (submitted_doc_id, url)
                    if pair_key not in seen_pairs:
                        matches.append(TransientMatch(
                            submitted_doc_id=submitted_doc_id,
                            internet_url=url,
                            similarity_score=round(score, 4),
                            matched_chunk_submitted=query_chunk,
                            matched_chunk_internet=target["text"]
                        ))
                        seen_pairs.add(pair_key)
                        
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches
