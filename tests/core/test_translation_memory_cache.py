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

import threading

import pytest  # noqa: F401

from src.core.cross_lingual import TranslationMemoryCache


def test_concurrent_translations_writing():
    """Test that spawns 10 threads writing different translations simultaneously and asserts correct storage."""
    cache = TranslationMemoryCache()
    threads = []
    num_threads = 10

    def worker(thread_idx):
        source_text = f"source_phrase_{thread_idx}"
        translated_text = f"translated_phrase_{thread_idx}"
        cache.set(source_text, translated_text)

    # Spawn 10 concurrent writing threads
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Assert all translations are stored correctly
    for i in range(num_threads):
        source_text = f"source_phrase_{i}"
        expected_translation = f"translated_phrase_{i}"
        assert cache.get(source_text) == expected_translation


def test_concurrent_reads_and_writes_no_deadlock():
    """Test that reads and writes occur concurrently without deadlocking."""
    cache = TranslationMemoryCache()
    # Pre-populate some keys
    for i in range(20):
        cache.set(f"key_{i}", f"value_{i}")

    stop_event = threading.Event()

    def writer_worker():
        counter = 0
        while not stop_event.is_set():
            cache.set(f"dynamic_key_{counter}", f"dynamic_value_{counter}")
            counter += 1

    def reader_worker():
        while not stop_event.is_set():
            for i in range(20):
                cache.get(f"key_{i}")

    writer_thread = threading.Thread(target=writer_worker)
    reader_thread = threading.Thread(target=reader_worker)

    writer_thread.start()
    reader_thread.start()

    # Run concurrent read/write operations for 0.5 seconds
    threading.Event().wait(0.5)
    stop_event.set()

    writer_thread.join(timeout=2.0)
    reader_thread.join(timeout=2.0)

    # Ensure threads finished successfully without deadlocking
    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
