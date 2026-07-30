import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.core.concurrency import (ConcurrencyTimeoutError, FAISSLock,
                                  faiss_write_lock)

# ---------------------------------------------------------------------------
# Test Locking Mechanics
# ---------------------------------------------------------------------------

def test_faiss_lock_acquisition_and_release(tmp_path):
    """
    Test basic lock acquire and release functions properly.
    """
    lock_file = tmp_path / "test.lock"
    lock = FAISSLock(lock_file=str(lock_file), timeout=5)
    
    # Acquire
    lock.acquire()
    assert os.path.exists(lock_file)
    
    # Release
    lock.release()
    assert not os.path.exists(lock_file)

def test_faiss_lock_timeout(tmp_path):
    """
    Test that a locked file causes another instance to raise ConcurrencyTimeoutError.
    """
    lock_file = tmp_path / "test_timeout.lock"
    
    # Lock it manually
    fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, b"locked")
    os.close(fd)
    
    lock = FAISSLock(lock_file=str(lock_file), timeout=1)
    
    start_time = time.time()
    with pytest.raises(ConcurrencyTimeoutError):
        lock.acquire()
    
    assert time.time() - start_time >= 1.0

def test_faiss_write_lock_context_manager(tmp_path):
    """
    Test the context manager properly acquires and automatically releases.
    """
    lock_file = tmp_path / "context.lock"
    
    with faiss_write_lock(lock_path=str(lock_file), timeout=2):
        assert os.path.exists(lock_file)
        
    assert not os.path.exists(lock_file)

# ---------------------------------------------------------------------------
# Test Concurrent Threading
# ---------------------------------------------------------------------------

def mock_rebuild_task(lock_file: str, shared_resource: list, thread_id: int):
    """
    A simulated FAISS rebuild task. It attempts to acquire the lock, appends to
    the shared list, sleeps slightly, and releases.
    If the lock fails, it appends a corruption marker.
    """
    try:
        with faiss_write_lock(lock_path=lock_file, timeout=10):
            # Critical section
            time.sleep(0.05) # Simulate IO
            shared_resource.append(thread_id)
            # If not thread-safe, multiple threads will append at the same index
            # or cause race conditions.
    except ConcurrencyTimeoutError:
        shared_resource.append(-1) # Timeout failure

def test_concurrent_faiss_rebuild_sequencing(tmp_path):
    """
    Spawn 10 simultaneous threads attempting to "rebuild FAISS".
    The lock should sequence them perfectly so the shared resource has exactly 10 distinct entries.
    """
    lock_file = str(tmp_path / "concurrent.lock")
    shared_resource = []
    
    num_threads = 10
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            futures.append(executor.submit(mock_rebuild_task, lock_file, shared_resource, i))
            
        # Wait for all to finish
        for f in futures:
            f.result()
            
    # Verification
    assert len(shared_resource) == num_threads
    assert -1 not in shared_resource # No timeouts occurred
    assert sorted(shared_resource) == list(range(num_threads)) # All threads executed sequentially


def test_faiss_lock_configurable_timeout(mocker):
    mocker.patch('src.core.app_config.get_lock_timeout', return_value=15)
    from src.core.concurrency import FAISSLock
    lock = FAISSLock()
    assert lock.timeout == 15

def test_faiss_write_lock_context_configurable_timeout(mocker):
    mocker.patch('src.core.app_config.get_lock_timeout', return_value=45)
    from src.core.concurrency import faiss_write_lock
    with mocker.patch('src.core.concurrency.FAISSLock') as mock_lock:
        mock_lock.return_value.acquire.return_value = None
        with faiss_write_lock():
            pass
        mock_lock.assert_called_with(lock_file='corpus.index.lock', timeout=45)
