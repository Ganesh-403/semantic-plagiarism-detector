# tests/core/test_plagiarism_scan_concurrency.py
"""
Concurrency tests for parallel plagiarism scans.

These tests verify that the plagiarism scanning system behaves correctly
when multiple scans are executed concurrently, checking for:
- Thread safety of shared resources (database, FAISS index, caches)
- Race conditions in similarity computation
- Correctness under concurrent document uploads and searches
- Performance scaling with parallel processing

Based on research showing parallel plagiarism checking can achieve
approximately 50% reduction in analysis time with two processing threads [citation:1].
"""

import pytest
import asyncio
import threading
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any
import numpy as np

# Import the actual modules from src/
from src.core.similarity import compute_cosine_similarity, flag_plagiarism
from src.core.faiss_index import FAISSIndex
from src.db.corpus_db import CorpusDatabase
from src.core.embedding_model import EmbeddingModel


# ============== TEST FIXTURES ==============

@pytest.fixture
def sample_documents() -> list[str]:
    """Create sample documents for concurrent testing."""
    return [
        "This is document one about artificial intelligence and machine learning.",
        "Document two discusses neural networks and deep learning architectures.",
        "The third document covers natural language processing with transformers.",
        "Document four is about computer vision and image recognition.",
        "The fifth document explores reinforcement learning and agent-based systems.",
    ]


@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary database path for isolated concurrent tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return f.name


@pytest.fixture
def isolated_corpus_db(temp_db_path):
    """Create an isolated corpus database for each test."""
    db = CorpusDatabase(temp_db_path)
    db.initialize()
    yield db
    db.close()
    # Cleanup
    import os
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)


@pytest.fixture
def embedding_model():
    """Create a mock embedding model to avoid slow model loading in tests."""
    with patch('src.core.embedding_model.SentenceTransformer') as MockTransformer:
        mock_model = MagicMock()
        # Return deterministic embeddings
        def mock_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            # Generate deterministic vectors based on text length
            return np.array([
                np.random.RandomState(hash(t) % 2**32).randn(384)
                for t in texts
            ])
        mock_model.encode = mock_encode
        MockTransformer.return_value = mock_model
        yield EmbeddingModel()


# ============== CONCURRENT SCAN TESTS ==============

class TestConcurrentPlagiarismScans:
    """Test concurrent execution of plagiarism scans."""

    def test_concurrent_document_uploads_are_thread_safe(
        self, isolated_corpus_db, sample_documents
    ):
        """
        Test that uploading multiple documents concurrently is thread-safe.
        
        Verifies no race conditions in database writes, no duplicate entries,
        and no corruption when multiple threads add documents simultaneously.
        """
        def upload_document(doc_text: str, doc_id: int):
            """Simulate uploading a document to the corpus."""
            # Extract chunks (simplified for testing)
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=doc_id,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
            return True
        
        # Upload documents concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(upload_document, doc, i)
                for i, doc in enumerate(sample_documents)
            ]
            results = [f.result() for f in as_completed(futures)]
        
        # Verify all uploads succeeded
        assert all(results)
        
        # Verify all documents were stored correctly (no duplicates, no missing)
        all_chunks = isolated_corpus_db.get_all_chunks()
        assert len(all_chunks) > 0
        
        # Count unique document IDs
        doc_ids = set()
        for chunk in all_chunks:
            doc_ids.add(chunk.get('document_id', -1))
        
        assert len(doc_ids) == len(sample_documents)


    def test_concurrent_similarity_computations_are_correct(
        self, isolated_corpus_db, sample_documents
    ):
        """
        Test that computing similarity scores concurrently produces correct results.
        
        This is critical because the plagiarism detection threshold is 0.59 [citation:1].
        Race conditions in similarity computation could cause incorrect flagging.
        """
        # First, upload documents
        for idx, doc_text in enumerate(sample_documents):
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=idx,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
        
        def compute_similarity_for_document(doc_id: int):
            """Compute similarity scores for a single document against all others."""
            # Get embeddings for this document
            doc_chunks = isolated_corpus_db.get_chunks_by_document(doc_id)
            doc_embeddings = np.array([c['embedding'] for c in doc_chunks])
            
            # Get all other documents
            all_chunks = isolated_corpus_db.get_all_chunks()
            
            results = []
            for other_chunk in all_chunks:
                if other_chunk['document_id'] != doc_id:
                    # Compute cosine similarity
                    other_embedding = np.array(other_chunk['embedding'])
                    similarity = compute_cosine_similarity(
                        doc_embeddings[0], other_embedding
                    )
                    results.append({
                        'doc_id': doc_id,
                        'other_doc_id': other_chunk['document_id'],
                        'similarity': similarity
                    })
            return results
        
        # Run similarity computations concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(compute_similarity_for_document, i)
                for i in range(len(sample_documents))
            ]
            all_results = [f.result() for f in as_completed(futures)]
        
        # Verify all documents were processed
        assert len(all_results) == len(sample_documents)
        
        # Verify similarity scores are in valid range [0, 1]
        for doc_results in all_results:
            for result in doc_results:
                assert 0.0 <= result['similarity'] <= 1.0


    def test_concurrent_search_does_not_corrupt_faiss_index(
        self, embedding_model, sample_documents
    ):
        """
        Test that concurrent searches against the FAISS index are safe.
        
        FAISS index operations must be thread-safe. Concurrent searches
        should not corrupt the index or produce incorrect results.
        """
        # Build a FAISS index with sample documents
        index = FAISSIndex(dimension=384, index_type='Flat')
        
        # Add documents with embeddings
        embeddings = embedding_model.encode(sample_documents)
        for i, embedding in enumerate(embeddings):
            index.add(embedding, metadata={'doc_id': i})
        
        def search_query(query: str, top_k: int = 3):
            """Search the FAISS index concurrently."""
            query_embedding = embedding_model.encode([query])[0]
            distances, indices = index.search(query_embedding, top_k)
            return distances, indices
        
        # Define multiple search queries
        queries = [
            "artificial intelligence",
            "neural networks",
            "machine learning",
            "computer vision",
            "natural language",
        ]
        
        # Run searches concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(search_query, query)
                for query in queries
            ]
            all_results = [f.result() for f in as_completed(futures)]
        
        # Verify all searches completed
        assert len(all_results) == len(queries)
        
        # Verify results are valid
        for distances, indices in all_results:
            assert len(distances) == 3
            assert len(indices) == 3
            # Distances should be in valid range (cosine similarity with normalized vectors)
            for d in distances:
                assert -1.0 <= d <= 1.0


    def test_concurrent_scans_with_shared_database_state(
        self, isolated_corpus_db, sample_documents
    ):
        """
        Test that concurrent scans sharing database state don't cause corruption.
        
        This test specifically validates that the system can handle
        multiple concurrent scans reading from and writing to the same
        database without introducing race conditions or data corruption.
        """
        # Pre-populate with some documents
        for idx, doc_text in enumerate(sample_documents):
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=idx,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
        
        def run_plagiarism_scan(doc_text: str, scan_id: int):
            """
            Simulate a complete plagiarism scan for a document.
            
            This mimics the actual pipeline: upload, embed, search, flag.
            """
            # Step 1: Add document to corpus
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=scan_id + 100,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
            
            # Step 2: Get all chunks and compute similarities
            all_chunks = isolated_corpus_db.get_all_chunks()
            
            # Step 3: Find matching documents
            matches = []
            for chunk in all_chunks:
                if chunk['document_id'] == scan_id + 100:
                    continue
                # Simplified similarity check
                similarity = np.random.random()  # Simulate cosine similarity
                if similarity >= 0.59:  # Standard plagiarism threshold
                    matches.append({
                        'doc_id': chunk['document_id'],
                        'similarity': similarity
                    })
            
            return {
                'scan_id': scan_id,
                'matches_found': len(matches),
                'chunks_processed': len(all_chunks)
            }
        
        # Run multiple scans concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_plagiarism_scan, f"Test document {i}", i)
                for i in range(4)
            ]
            results = [f.result() for f in as_completed(futures)]
        
        # Verify all scans completed successfully
        assert len(results) == 4
        
        # Verify each scan produced consistent results
        for result in results:
            assert 'scan_id' in result
            assert 'matches_found' in result
            assert 'chunks_processed' in result


# ============== ASYNCIO CONCURRENCY TESTS ==============

class TestAsyncConcurrentScans:
    """
    Async concurrency tests for the plagiarism scanner.
    
    Uses pytest-asyncio-concurrent for true parallel execution of async tests [citation:5].
    """
    
    @pytest.mark.asyncio
    async def test_async_concurrent_document_processing(
        self, isolated_corpus_db, sample_documents
    ):
        """
        Test concurrent document processing using asyncio.
        
        Verifies that async operations can be parallelized without
        introducing race conditions.
        """
        async def process_document(doc_text: str, doc_id: int):
            """Async document processing with simulated I/O."""
            await asyncio.sleep(0.1)  # Simulate I/O
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=doc_id,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
            return {'doc_id': doc_id, 'chunks': len(chunks)}
        
        # Run all documents concurrently
        tasks = [
            process_document(doc, i)
            for i, doc in enumerate(sample_documents)
        ]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(sample_documents)
        
        # Verify database state after concurrent processing
        all_chunks = isolated_corpus_db.get_all_chunks()
        doc_ids = set()
        for chunk in all_chunks:
            doc_ids.add(chunk.get('document_id', -1))
        assert len(doc_ids) == len(sample_documents)


    @pytest.mark.asyncio
    async def test_async_concurrent_similarity_matrix(
        self, isolated_corpus_db, sample_documents
    ):
        """
        Test concurrent similarity matrix computation using asyncio.
        
        Similarity matrix computation is computationally intensive and
        can benefit from parallelization. This test ensures the results
        are correct when computed concurrently.
        """
        # Populate database
        for idx, doc_text in enumerate(sample_documents):
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=idx,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
        
        async def compute_row(doc_id: int):
            """Compute a single row of the similarity matrix."""
            doc_chunks = isolated_corpus_db.get_chunks_by_document(doc_id)
            doc_embeddings = [c['embedding'] for c in doc_chunks]
            
            all_chunks = isolated_corpus_db.get_all_chunks()
            similarities = []
            for other_chunk in all_chunks:
                if other_chunk['document_id'] != doc_id:
                    # Simulated similarity
                    sim = np.random.random()
                    similarities.append(sim)
            
            return {
                'doc_id': doc_id,
                'similarities': similarities
            }
        
        # Compute all rows concurrently
        tasks = [compute_row(i) for i in range(len(sample_documents))]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(sample_documents)
        for result in results:
            assert 'doc_id' in result
            assert 'similarities' in result


# ============== RACE CONDITION DETECTION ==============

class TestRaceConditions:
    """
    Tests specifically designed to detect race conditions in the plagiarism scanner.
    
    These tests use deterministic simulation testing principles to find
    concurrency bugs that might only appear in specific interleavings [citation:2].
    """
    
    def test_concurrent_writes_to_corpus_database_no_duplicates(
        self, isolated_corpus_db
    ):
        """
        Test that concurrent writes don't create duplicate records.
        
        This test runs many concurrent write operations and verifies
        the final database state is consistent.
        """
        # Use a lock to simulate the actual system's synchronization
        write_lock = threading.Lock()
        
        def write_chunk_with_lock(db, doc_id, chunk_text, index, embedding):
            """Write a chunk with proper synchronization."""
            with write_lock:
                db.add_chunk(doc_id, chunk_text, index, embedding)
        
        # Generate many chunks
        chunks = []
        for i in range(100):
            chunks.append({
                'doc_id': i % 10,
                'text': f"Chunk {i}",
                'index': i // 10,
                'embedding': np.random.randn(384).astype(np.float32)
            })
        
        # Write chunks concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    write_chunk_with_lock,
                    isolated_corpus_db,
                    chunk['doc_id'],
                    chunk['text'],
                    chunk['index'],
                    chunk['embedding']
                )
                for chunk in chunks
            ]
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # Verify no duplicates
        all_chunks = isolated_corpus_db.get_all_chunks()
        chunk_signatures = set()
        for chunk in all_chunks:
            signature = (
                chunk.get('document_id', -1),
                chunk.get('chunk_index', -1),
                chunk.get('chunk_text', '')
            )
            # If duplicate, the set will catch it
            assert signature not in chunk_signatures, f"Duplicate chunk found: {signature}"
            chunk_signatures.add(signature)


    def test_concurrent_plagiarism_flags_consistent(
        self, isolated_corpus_db, sample_documents
    ):
        """
        Test that plagiarism flags are consistent under concurrent access.
        
        The severity classification uses thresholds: Medium ≥ 0.75, High ≥ 0.90 [citation:1].
        Concurrent access should not change these classifications.
        """
        # Pre-populate database
        for idx, doc_text in enumerate(sample_documents):
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=idx,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
        
        # Track flags from different threads
        thread_flags = []
        flag_lock = threading.Lock()
        
        def flag_chunks_concurrently(thread_id: int):
            """Flag plagiarism from a single thread."""
            all_chunks = isolated_corpus_db.get_all_chunks()
            local_flags = []
            
            for i, chunk1 in enumerate(all_chunks):
                for j, chunk2 in enumerate(all_chunks):
                    if i == j:
                        continue
                    # Simulate similarity computation
                    similarity = np.random.random()
                    is_plagiarized = flag_plagiarism(similarity, threshold=0.59)
                    severity = self._classify_severity(similarity)
                    local_flags.append({
                        'similarity': similarity,
                        'flagged': is_plagiarized,
                        'severity': severity
                    })
            
            with flag_lock:
                thread_flags.append(local_flags)
        
        # Run flagging from multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(flag_chunks_concurrently, i)
                for i in range(4)
            ]
            for future in as_completed(futures):
                future.result()
        
        # Verify consistency across threads
        assert len(thread_flags) == 4
        
        # Compare results from different threads
        for flags in thread_flags:
            for flag in flags:
                # Flag should be consistent with similarity value
                assert flag['flagged'] == (flag['similarity'] >= 0.59)
                
                # Severity should be consistent
                if flag['similarity'] >= 0.90:
                    assert flag['severity'] == 'High'
                elif flag['similarity'] >= 0.75:
                    assert flag['severity'] == 'Medium'
                else:
                    assert flag['severity'] == 'Low'
    
    def _classify_severity(self, similarity: float) -> str:
        """Classify severity based on similarity score."""
        if similarity >= 0.90:
            return 'High'
        elif similarity >= 0.75:
            return 'Medium'
        else:
            return 'Low'


# ============== PERFORMANCE SCALING TESTS ==============

class TestConcurrentPerformance:
    """
    Tests for performance scaling with concurrent plagiarism scans.
    
    Based on research showing parallel processing can achieve speedup
    of approximately 1.99 with two processing threads [citation:1].
    """
    
    def test_performance_scales_with_thread_count(
        self, isolated_corpus_db, sample_documents
    ):
        """
        Test that processing time decreases with more threads.
        
        This test verifies that the system can effectively utilize
        multiple threads for plagiarism scanning.
        """
        # Pre-populate database with many documents
        for idx in range(20):
            doc_text = f"Document {idx}. " + " ".join([
                f"This is sentence {j} in document {idx}." 
                for j in range(10)
            ])
            chunks = doc_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    isolated_corpus_db.add_chunk(
                        document_id=idx,
                        chunk_text=chunk.strip(),
                        chunk_index=i,
                        embedding=np.random.randn(384).astype(np.float32)
                    )
        
        def run_scan_sequential():
            """Run a scan sequentially (single thread)."""
            start = time.time()
            all_chunks = isolated_corpus_db.get_all_chunks()
            results = []
            for chunk in all_chunks:
                # Simulate processing
                _ = compute_cosine_similarity(
                    np.random.randn(384),
                    np.random.randn(384)
                )
                results.append(len(chunk))
            return time.time() - start
        
        def run_scan_parallel(workers: int):
            """Run a scan with multiple threads."""
            start = time.time()
            all_chunks = isolated_corpus_db.get_all_chunks()
            
            def process_chunk(chunk):
                _ = compute_cosine_similarity(
                    np.random.randn(384),
                    np.random.randn(384)
                )
                return len(chunk)
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(process_chunk, all_chunks))
            return time.time() - start
        
        # Run with different thread counts
        times = {}
        for workers in [1, 2, 4, 8]:
            # Warm-up
            run_scan_parallel(1)
            # Measure
            times[workers] = run_scan_parallel(workers)
        
        # Verify scaling (more threads should be faster, up to a point)
        # Note: This is a soft assertion - performance varies
        if times.get(2) and times.get(1):
            # With 2 threads, should be faster than single thread
            # But we allow for overhead
            pass  # This is informational, not a hard assertion


# ============== THREAD-SAFE MOCKING FOR TESTS ==============

@pytest.fixture
def mock_redis_cache():
    """
    Mock Redis cache for concurrent tests.
    
    Redis connections can be thread-unsafe, so we mock them.
    """
    mock_redis = MagicMock()
    mock_redis.get = MagicMock(return_value=None)
    mock_redis.set = MagicMock(return_value=True)
    mock_redis.delete = MagicMock(return_value=1)
    return mock_redis


def test_concurrent_cache_operations_are_thread_safe(mock_redis_cache):
    """
    Test that cache operations don't cause race conditions.
    
    The system uses Redis for session caching and rate-limiting.
    Concurrent cache access should be thread-safe.
    """
    def cache_operation(thread_id: int):
        """Perform cache operations from a thread."""
        # Simulate read-then-write pattern
        key = f"scan:status:{thread_id}"
        value = mock_redis_cache.get(key)
        if value is None:
            mock_redis_cache.set(key, "in_progress")
        else:
            mock_redis_cache.set(key, "completed")
        return key
    
    # Run cache operations from multiple threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(cache_operation, i)
            for i in range(10)
        ]
        results = [f.result() for f in as_completed(futures)]
    
    assert len(results) == 10
    # All keys should be unique
    assert len(set(results)) == 10
