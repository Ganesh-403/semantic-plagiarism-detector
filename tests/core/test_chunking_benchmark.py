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

import time

from src.core.text_chunking import chunk_documents


def test_text_chunking_performance_benchmark():
    """Ensure text chunking execution time stays under 1.0 second for 100,000 character inputs."""
    # Generate synthetic text buffer of 100,000 characters
    sample_paragraph = "This is a sentence used for testing the semantic text chunking performance benchmark pipeline. "
    repeat_count = (100000 // len(sample_paragraph)) + 1
    synthetic_text = (sample_paragraph * repeat_count)[:100000]

    assert (
        len(synthetic_text) == 100000
    ), "Synthetic text must be exactly 100,000 characters."

    # Measure execution time of chunk_documents
    start_time = time.perf_counter()

    # chunk_documents typically accepts a dict or list depending on implementation;
    # wrapping text in a document dictionary representation
    document_input = {"benchmark_doc.txt": synthetic_text}
    chunks = chunk_documents(document_input, chunk_size=500, chunk_overlap=50)

    elapsed_time = time.perf_counter() - start_time

    # Assert execution time is strictly less than 1.0 second
    assert (
        elapsed_time < 1.0
    ), f"Chunking took {elapsed_time:.4f} seconds, exceeding the 1.0 second threshold."
    assert len(chunks) > 0, "Chunks should be successfully generated."
