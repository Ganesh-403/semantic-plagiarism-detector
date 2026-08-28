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

from src.core.embeddings import generate_embeddings


def test_embedding_generation_performance():
    """Ensure embedding generation remains within acceptable latency."""

    texts = [
        "Artificial intelligence is transforming software development.",
        "Python is widely used for backend services.",
        "Machine learning enables predictive analytics.",
        "Vector databases improve semantic search.",
        "Cloud computing powers scalable applications.",
        "Cybersecurity protects sensitive information.",
        "Open source software accelerates innovation.",
        "Unit testing improves software reliability.",
        "Continuous integration catches regressions early.",
        "Performance optimization enhances user experience.",
    ]

    start_time = time.perf_counter()

    embeddings = generate_embeddings(texts)

    elapsed = time.perf_counter() - start_time

    assert len(embeddings) == len(texts)
    assert elapsed < 5.0, (
        f"Embedding generation took {elapsed:.2f}s " "which exceeds the 5.0s threshold."
    )
