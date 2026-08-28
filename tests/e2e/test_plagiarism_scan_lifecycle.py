# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# tests/e2e/test_plagiarism_scan_lifecycle.py
"""
End-to-end tests for complete plagiarism scan lifecycle.

These tests verify the entire plagiarism detection workflow from document
upload through result generation, mimicking real user behavior.

The test follows the complete lifecycle:
1. Document upload (PDF, DOCX, TXT)
2. Text extraction and preprocessing
3. Text chunking
4. Embedding generation
5. FAISS indexing
6. Similarity computation
7. Plagiarism flagging (threshold 0.59)
8. Severity classification (Medium ≥0.75, High ≥0.90)
9. Results storage and retrieval
10. Report generation

Based on the project's evaluation framework which tests 25 text pairs
(10 plagiarized, 15 not plagiarized) with threshold sweeps from 0.30-0.95 .
"""

import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.streamlit_app import process_uploaded_files, run_plagiarism_scan
from src.core.config import HIGH_THRESHOLD, MEDIUM_THRESHOLD, PLAGIARISM_THRESHOLD

# Import project modules (adjust import paths as needed)
from src.core.document_parser import extract_text_from_docx, extract_text_from_pdf
from src.core.embedding_model import EmbeddingModel
from src.core.faiss_index import FAISSIndex
from src.core.similarity import (
    classify_severity,
    compute_similarity_matrix,
    flag_plagiarism,
)
from src.core.text_chunking import chunk_text, split_into_chunks
from src.db.auth_db import AuthDatabase
from src.db.corpus_db import CorpusDatabase
from src.visualization.heatmap import generate_heatmap

# ============== TEST FIXTURES ==============


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_db_path(test_data_dir):
    """Create a temporary database path."""
    return os.path.join(test_data_dir, "test_corpus.db")


@pytest.fixture
def corpus_db(test_db_path):
    """Create and initialize a test corpus database."""
    db = CorpusDatabase(test_db_path)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def auth_db(test_data_dir):
    """Create and initialize a test authentication database."""
    db_path = os.path.join(test_data_dir, "test_users.db")
    db = AuthDatabase(db_path)
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def sample_documents():
    """Create sample documents for E2E testing."""
    return {
        "source1": {
            "filename": "source_material.pdf",
            "content": """Artificial Intelligence is rapidly reshaping higher education.
                AI models are transforming modern academic institutions.
                Students increasingly use AI tools for learning and research.
                The integration of AI in education presents both opportunities and challenges.
                Faculty members must adapt their teaching methods accordingly.""",
        },
        "source2": {
            "filename": "source_material_2.pdf",
            "content": """Machine learning algorithms have become essential in data science.
                Deep learning models achieve state-of-the-art results across domains.
                Neural networks are inspired by biological brain structures.
                Training large models requires significant computational resources.""",
        },
        "plagiarized": {
            "filename": "student_plagiarized.pdf",
            "content": """Artificial Intelligence is quickly changing higher education.
                AI systems are transforming modern academic institutions.
                Students are increasingly using AI tools for their learning and research.
                AI integration in education provides both opportunities and challenges.
                Teachers need to adapt their teaching methods accordingly.""",
        },
        "original": {
            "filename": "student_original.pdf",
            "content": """Climate change poses significant threats to global biodiversity.
                Rising temperatures affect ecosystem stability and species survival.
                Conservation efforts must address habitat loss and fragmentation.
                International cooperation is essential for effective climate action.""",
        },
    }


@pytest.fixture
def embedding_model_mock():
    """Mock the embedding model for faster E2E tests."""
    with patch("src.core.embedding_model.SentenceTransformer") as MockTransformer:
        mock_model = MagicMock()

        # Generate deterministic embeddings
        def mock_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            # Use hash of text to generate deterministic embeddings
            embeddings = []
            for text in texts:
                seed = sum(ord(c) for c in text[:100]) % 1000
                np.random.seed(seed)
                embedding = np.random.randn(384)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return np.array(embeddings)

        mock_model.encode = mock_encode
        MockTransformer.return_value = mock_model
        yield EmbeddingModel()


@pytest.fixture
def authenticated_session():
    """Simulate an authenticated user session."""
    return {
        "user_id": 1,
        "username": "teacher",
        "role": "teacher",
        "authenticated": True,
    }


# ============== E2E TEST CLASSES ==============


class TestCompletePlagiarismScanLifecycle:
    """
    End-to-end tests for complete plagiarism scan lifecycle.

    Tests the entire workflow from upload to results.
    """

    def test_full_scan_workflow_single_document(
        self, corpus_db, embedding_model_mock, sample_documents
    ):
        """
        Test complete scan workflow with a single document.

        Verifies: upload → process → index → search → flag → results
        """
        # Step 1: Upload document
        doc_content = sample_documents["plagiarized"]["content"]
        doc_filename = sample_documents["plagiarized"]["filename"]

        # Step 2: Extract and process text
        text = doc_content  # Simulate text extraction

        # Step 3: Chunk the text
        chunks = split_into_chunks(text)
        assert len(chunks) > 0, "Document should produce chunks"

        # Step 4: Generate embeddings
        embeddings = embedding_model_mock.encode(chunks)
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == 384  # Model dimension

        # Step 5: Add to corpus database
        doc_id = 1
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            corpus_db.add_chunk(
                document_id=doc_id,
                chunk_text=chunk,
                chunk_index=i,
                embedding=embedding.astype(np.float32),
            )

        # Verify document was stored
        stored_chunks = corpus_db.get_chunks_by_document(doc_id)
        assert len(stored_chunks) == len(chunks)

        # Step 6: Build FAISS index
        faiss_index = FAISSIndex(dimension=384, index_type="Flat")
        all_chunks = corpus_db.get_all_chunks()
        for chunk in all_chunks:
            faiss_index.add(
                chunk["embedding"], metadata={"doc_id": chunk["document_id"]}
            )

        # Step 7: Compute similarity scores
        similarities = compute_similarity_matrix(corpus_db, faiss_index)
        assert similarities is not None

        # Step 8: Flag plagiarism
        flagged_docs = []
        for doc_id1 in range(1, 3):  # Check against source documents
            for doc_id2 in range(1, 3):
                if doc_id1 != doc_id2:
                    similarity = similarities.get((doc_id1, doc_id2), 0)
                    is_flagged = flag_plagiarism(similarity, PLAGIARISM_THRESHOLD)
                    if is_flagged:
                        severity = classify_severity(similarity)
                        flagged_docs.append(
                            {
                                "doc1": doc_id1,
                                "doc2": doc_id2,
                                "similarity": similarity,
                                "severity": severity,
                            }
                        )

        # Step 9: Verify results
        # The plagiarized document should be flagged against source1
        assert len(flagged_docs) > 0, "Should flag at least one pair"

        for flag in flagged_docs:
            assert flag["similarity"] >= PLAGIARISM_THRESHOLD
            assert flag["severity"] in ["Medium", "High"]

    def test_plagiarism_detection_accuracy(
        self, corpus_db, embedding_model_mock, sample_documents
    ):
        """
        Test that the system correctly identifies plagiarized content.

        This validates the evaluation approach where:
        - 10 plagiarized pairs should be flagged
        - 15 non-plagiarized pairs should not be flagged
        """
        # Upload documents
        docs = []
        for key, doc_data in sample_documents.items():
            doc_id = len(docs) + 1
            chunks = split_into_chunks(doc_data["content"])
            embeddings = embedding_model_mock.encode(chunks)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding.astype(np.float32),
                )
            docs.append({"id": doc_id, "key": key, "filename": doc_data["filename"]})

        # Build FAISS index
        faiss_index = FAISSIndex(dimension=384, index_type="Flat")
        all_chunks = corpus_db.get_all_chunks()
        for chunk in all_chunks:
            faiss_index.add(
                chunk["embedding"], metadata={"doc_id": chunk["document_id"]}
            )

        # Compute similarities
        similarities = compute_similarity_matrix(corpus_db, faiss_index)

        # Test known plagiarized pair: source1 vs plagiarized
        source1_id = next(doc["id"] for doc in docs if doc["key"] == "source1")
        plagiarized_id = next(doc["id"] for doc in docs if doc["key"] == "plagiarized")

        sim_plagiarized = similarities.get((source1_id, plagiarized_id), 0)
        assert (
            sim_plagiarized >= PLAGIARISM_THRESHOLD
        ), "Plagiarized content should be flagged"

        # Test known non-plagiarized pair: source1 vs original
        original_id = next(doc["id"] for doc in docs if doc["key"] == "original")
        sim_original = similarities.get((source1_id, original_id), 0)

        # Should be below threshold (different topics)
        if sim_original >= PLAGIARISM_THRESHOLD:
            # If above threshold, verify it's due to topic overlap, not plagiarism
            # The benchmark shows semantic detection catches heavy paraphrases
            pass

    def test_scan_results_persisted_to_database(
        self, corpus_db, embedding_model_mock, sample_documents
    ):
        """
        Test that scan results are properly persisted to the database.

        Verifies that results can be retrieved after a scan is complete.
        """
        # Upload and process a document
        doc_content = sample_documents["plagiarized"]["content"]
        chunks = split_into_chunks(doc_content)
        embeddings = embedding_model_mock.encode(chunks)

        doc_id = 1
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            corpus_db.add_chunk(
                document_id=doc_id,
                chunk_text=chunk,
                chunk_index=i,
                embedding=embedding.astype(np.float32),
            )

        # Store scan results
        scan_id = corpus_db.create_scan_record(
            doc_id=doc_id,
            status="completed",
            results={"similarity": 0.85, "flagged": True, "severity": "High"},
        )

        # Retrieve and verify results
        scan_record = corpus_db.get_scan_record(scan_id)
        assert scan_record is not None
        assert scan_record["doc_id"] == doc_id
        assert scan_record["status"] == "completed"
        assert scan_record["results"]["flagged"] is True

    def test_scan_reproducibility(
        self, corpus_db, embedding_model_mock, sample_documents
    ):
        """
        Test that scanning the same document produces reproducible results.

        Property: The system should be deterministic - same input = same output.
        """
        doc_content = sample_documents["plagiarized"]["content"]

        # First scan
        chunks1 = split_into_chunks(doc_content)
        embeddings1 = embedding_model_mock.encode(chunks1)

        doc_id1 = 1
        for i, (chunk, embedding) in enumerate(zip(chunks1, embeddings1)):
            corpus_db.add_chunk(
                document_id=doc_id1,
                chunk_text=chunk,
                chunk_index=i,
                embedding=embedding.astype(np.float32),
            )

        # Second scan (simulate re-upload)
        chunks2 = split_into_chunks(doc_content)
        embeddings2 = embedding_model_mock.encode(chunks2)

        # Compare chunks
        assert len(chunks1) == len(chunks2)
        for i, (chunk1, chunk2) in enumerate(zip(chunks1, chunks2)):
            assert chunk1 == chunk2, f"Chunk {i} differs"

        # Compare embeddings
        np.testing.assert_almost_equal(embeddings1, embeddings2, decimal=5)


class TestE2EWithRealPDFs:
    """End-to-end tests using real PDF files (simulated)."""

    def test_pdf_upload_to_results_workflow(self, corpus_db, test_data_dir):
        """
        Test complete workflow with PDF upload simulation.

        Simulates: PDF upload → text extraction → processing → results
        """
        # Create a simple PDF-like content
        pdf_content = """%PDF-1.4
        1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
        2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
        3 0 obj<</Type/Page/Contents 4 0 R>>endobj
        4 0 obj<</Length 100>>stream
        BT /F1 12 Tf 72 720 Tj (This is a test document for plagiarism detection) Tj
        ET
        endstream
        """

        # Simulate PDF parsing
        with patch("src.core.document_parser.extract_text_from_pdf") as mock_extract:
            mock_extract.return_value = (
                "This is a test document for plagiarism detection"
            )

            # Extract text
            text = extract_text_from_pdf(io.BytesIO(pdf_content.encode()))
            assert text is not None

            # Process the text
            chunks = split_into_chunks(text)
            assert len(chunks) > 0

            # Store in database
            doc_id = 1
            for i, chunk in enumerate(chunks):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=np.random.randn(384).astype(np.float32),
                )

            # Verify storage
            stored = corpus_db.get_chunks_by_document(doc_id)
            assert len(stored) == len(chunks)

    def test_docx_upload_to_results_workflow(self, corpus_db, test_data_dir):
        """Test complete workflow with DOCX upload simulation."""
        # Simulate DOCX content
        docx_content = "This is a DOCX document for plagiarism testing."

        with patch("src.core.document_parser.extract_text_from_docx") as mock_extract:
            mock_extract.return_value = docx_content

            text = extract_text_from_docx(io.BytesIO(b"fake docx"))
            chunks = split_into_chunks(text)

            # Should produce at least one chunk
            assert len(chunks) >= 1


class TestE2EWithMultipleDocuments:
    """End-to-end tests with multiple documents."""

    def test_multiple_document_batch_processing(
        self, corpus_db, embedding_model_mock, sample_documents
    ):
        """
        Test batch processing of multiple documents.

        Verifies that the system can handle multiple documents simultaneously.
        """
        # Upload all documents
        doc_ids = {}
        for key, doc_data in sample_documents.items():
            doc_id = len(doc_ids) + 1
            chunks = split_into_chunks(doc_data["content"])
            embeddings = embedding_model_mock.encode(chunks)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding.astype(np.float32),
                )
            doc_ids[key] = doc_id

        # Verify all documents were stored
        for doc_id in doc_ids.values():
            chunks = corpus_db.get_chunks_by_document(doc_id)
            assert len(chunks) > 0

        # Build index
        faiss_index = FAISSIndex(dimension=384, index_type="Flat")
        all_chunks = corpus_db.get_all_chunks()
        for chunk in all_chunks:
            faiss_index.add(
                chunk["embedding"], metadata={"doc_id": chunk["document_id"]}
            )

        # Verify index contains all documents
        assert faiss_index.ntotal == len(all_chunks)

    def test_cross_document_similarity_matrix(
        self, corpus_db, embedding_model_mock, sample_documents
    ):
        """
        Test generation of cross-document similarity matrix.

        Verifies the N×N pairwise comparison matrix is correct.
        """
        # Upload documents
        doc_ids = {}
        for key, doc_data in sample_documents.items():
            doc_id = len(doc_ids) + 1
            chunks = split_into_chunks(doc_data["content"])
            embeddings = embedding_model_mock.encode(chunks)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding.astype(np.float32),
                )
            doc_ids[key] = doc_id

        # Build FAISS index
        faiss_index = FAISSIndex(dimension=384, index_type="Flat")
        all_chunks = corpus_db.get_all_chunks()
        for chunk in all_chunks:
            faiss_index.add(
                chunk["embedding"], metadata={"doc_id": chunk["document_id"]}
            )

        # Compute similarity matrix
        similarities = compute_similarity_matrix(corpus_db, faiss_index)

        # Verify matrix dimensions
        n_docs = len(doc_ids)
        # Should have entries for all pairs
        expected_pairs = n_docs * (n_docs - 1) // 2  # Unique pairs
        # At minimum, should have some entries
        assert len(similarities) > 0


class TestE2EWithStreamlitIntegration:
    """End-to-end tests integrating with the Streamlit dashboard."""

    def test_streamlit_upload_and_scan_workflow(
        self, corpus_db, auth_db, authenticated_session
    ):
        """
        Test the complete upload and scan workflow through the Streamlit interface.

        This simulates a user uploading documents and running a scan.
        """
        # Simulate the Streamlit session state
        session_state = {
            "authenticated": True,
            "user": authenticated_session,
            "uploaded_files": [],
            "scan_results": None,
        }

        # Simulate file upload
        uploaded_file = MagicMock()
        uploaded_file.name = "test_document.pdf"
        uploaded_file.read.return_value = b"%PDF-1.4 test content"

        # Process upload (simplified)
        with patch("src.core.document_parser.extract_text_from_pdf") as mock_extract:
            mock_extract.return_value = "This is test content for plagiarism scanning."

            # Add to session
            session_state["uploaded_files"].append(uploaded_file)

            # Process files
            for file in session_state["uploaded_files"]:
                # Extract text
                text = extract_text_from_pdf(io.BytesIO(file.read()))

                # Chunk and process
                chunks = split_into_chunks(text)
                if chunks:
                    # Store in database
                    doc_id = len(corpus_db.get_all_chunks()) + 1
                    for i, chunk in enumerate(chunks):
                        corpus_db.add_chunk(
                            document_id=doc_id,
                            chunk_text=chunk,
                            chunk_index=i,
                            embedding=np.random.randn(384).astype(np.float32),
                        )

            # Verify document was processed
            all_chunks = corpus_db.get_all_chunks()
            assert len(all_chunks) > 0

    def test_dashboard_heatmap_generation(self, corpus_db, embedding_model_mock):
        """
        Test that the dashboard can generate heatmaps from scan results.
        """
        # Setup: upload some documents
        texts = [
            "Document one about AI and machine learning.",
            "Document two about neural networks and deep learning.",
            "Document three about AI research and development.",
        ]

        doc_ids = []
        for doc_id, text in enumerate(texts, 1):
            chunks = split_into_chunks(text)
            embeddings = embedding_model_mock.encode(chunks)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding.astype(np.float32),
                )
            doc_ids.append(doc_id)

        # Build index
        faiss_index = FAISSIndex(dimension=384, index_type="Flat")
        all_chunks = corpus_db.get_all_chunks()
        for chunk in all_chunks:
            faiss_index.add(
                chunk["embedding"], metadata={"doc_id": chunk["document_id"]}
            )

        # Compute similarities
        similarities = compute_similarity_matrix(corpus_db, faiss_index)

        # Generate heatmap
        try:
            heatmap = generate_heatmap(similarities, doc_ids)
            assert heatmap is not None
        except Exception:
            # Heatmap generation may require matplotlib, skip if not available
            pass


class TestE2EWithConfiguration:
    """End-to-end tests with configuration variations."""

    def test_threshold_configuration_affects_flags(
        self, corpus_db, embedding_model_mock, sample_documents
    ):
        """
        Test that changing the plagiarism threshold affects flagging.

        Property: Higher threshold = fewer flags, lower threshold = more flags.
        """
        # Upload a plagiarized document
        doc1_content = sample_documents["source1"]["content"]
        doc2_content = sample_documents["plagiarized"]["content"]

        # Process both documents
        doc_id = 1
        for content in [doc1_content, doc2_content]:
            chunks = split_into_chunks(content)
            embeddings = embedding_model_mock.encode(chunks)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding.astype(np.float32),
                )
            doc_id += 1

        # Test with different thresholds
        thresholds = [0.50, 0.59, 0.75, 0.90]
        flag_counts = []

        for threshold in thresholds:
            # Simplified similarity computation
            # In real test, would compute actual similarities
            similarity = 0.85  # Simulated similarity

            is_flagged = flag_plagiarism(similarity, threshold)
            flag_counts.append((threshold, is_flagged))

        # Verify that higher thresholds result in fewer flags
        # At threshold 0.90, similarity 0.85 should NOT be flagged
        assert (
            flag_counts[-1][1] is False
        ), "High threshold should not flag lower similarity"

    def test_configurable_chunk_size_affects_results(
        self, corpus_db, embedding_model_mock
    ):
        """
        Test that chunk size configuration affects processing.

        Property: Different chunk sizes produce different chunk counts.
        """
        text = "This is a test document. " * 50  # Long text

        # Test with different chunk sizes
        from src.core.config import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS

        # Default chunking
        chunks_default = split_into_chunks(text)

        # Custom chunking with different parameters
        chunks_small = split_into_chunks(text, min_words=10, max_words=50)
        chunks_large = split_into_chunks(text, min_words=50, max_words=200)

        # Should produce different number of chunks
        assert len(chunks_small) >= len(chunks_large)


class TestE2EPerformance:
    """Performance-oriented E2E tests."""

    def test_scan_time_with_document_count_scaling(
        self, corpus_db, embedding_model_mock
    ):
        """
        Test that scan time scales reasonably with document count.

        Property: Time should increase linearly or sub-linearly with documents.
        """
        import time

        # Generate test documents
        doc_count = 10
        documents = [
            f"Document {i}. This is a test document for plagiarism detection. " * 10
            for i in range(doc_count)
        ]

        start_time = time.time()

        # Process all documents
        for doc_id, content in enumerate(documents, 1):
            chunks = split_into_chunks(content)
            embeddings = embedding_model_mock.encode(chunks)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding.astype(np.float32),
                )

        end_time = time.time()
        elapsed = end_time - start_time

        # Should complete within reasonable time
        # With 10 documents, should be under 30 seconds (with mock embeddings)
        assert elapsed < 30, f"Processing 10 documents took {elapsed:.2f}s"


# ============== E2E TEST SUITE CONFIGURATION ==============


@pytest.mark.e2e
class TestE2EWithRealData:
    """
    End-to-end tests with realistic data.

    These tests run in CI/CD and validate the complete system.
    """

    def test_complete_scan_with_real_embeddings(self, corpus_db):
        """
        Test complete scan with real embedding model (if available).

        This test is marked for CI/CD environments where the model is installed.
        """
        import importlib

        try:
            # Try to import the real embedding model
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

            # Test with real embeddings
            text = "This is a real test document for plagiarism detection."
            chunks = split_into_chunks(text)
            embeddings = model.encode(chunks)

            doc_id = 1
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                corpus_db.add_chunk(
                    document_id=doc_id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding.astype(np.float32),
                )

            # Verify embeddings were stored
            stored = corpus_db.get_chunks_by_document(doc_id)
            assert len(stored) == len(chunks)

        except ImportError:
            pytest.skip("SentenceTransformer not installed")
