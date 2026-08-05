import time
import pytest
from src.core.text_chunking import chunk_documents

def test_text_chunking_performance_benchmark():
    """Ensure text chunking execution time stays under 1.0 second for 100,000 character inputs."""
    # Generate synthetic text buffer of 100,000 characters
    sample_paragraph = "This is a sentence used for testing the semantic text chunking performance benchmark pipeline. "
    repeat_count = (100000 // len(sample_paragraph)) + 1
    synthetic_text = (sample_paragraph * repeat_count)[:100000]

    assert len(synthetic_text) == 100000, "Synthetic text must be exactly 100,000 characters."

    # Measure execution time of chunk_documents
    start_time = time.perf_counter()
    
    # chunk_documents typically accepts a dict or list depending on implementation; 
    # wrapping text in a document dictionary representation
    document_input = {"benchmark_doc.txt": synthetic_text}
    chunks = chunk_documents(document_input, chunk_size=500, chunk_overlap=50)
    
    elapsed_time = time.perf_counter() - start_time

    # Assert execution time is strictly less than 1.0 second
    assert elapsed_time < 1.0, f"Chunking took {elapsed_time:.4f} seconds, exceeding the 1.0 second threshold."
    assert len(chunks) > 0, "Chunks should be successfully generated."
    