import gc
import sqlite3
import threading
from unittest.mock import patch

import pytest

from src.db.corpus_db import (
    _all_connections,
    _connect,
    close_connections,
    _cleanup_all_connections,
    WeakConnection,
    _connection_pool
)

def test_connections_registered():
    """Test that new connections are registered in _all_connections."""
    with _connect() as conn:
        assert isinstance(conn, WeakConnection)
        assert conn in _all_connections
    close_connections()

def test_active_connections_tracked():
    """Test that active connections remain tracked across queries."""
    with _connect() as conn:
        conn.execute("SELECT 1").fetchall()
        assert conn in _all_connections
    close_connections()

def test_connections_not_strongly_retained():
    """Test closed/discarded connections are no longer strongly retained and get GC'd."""
    with _connect() as conn:
        pass
    
    # Connection is still tracked by the pool and _all_connections
    assert len(_all_connections) > 0
    
    # Close connections clears the pool
    close_connections()
    
    # Run garbage collection
    gc.collect()
    
    # WeakSet should be empty because the strong reference in the pool is gone
    assert len(_all_connections) == 0

def test_multiple_connections():
    """Test multiple connections across threads are tracked."""
    def worker():
        with _connect() as conn:
            assert conn in _all_connections

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    # The threads have finished, their connections are in their local pools.
    # _all_connections should have them.
    assert len(_all_connections) >= 5
    
    # Clean up all
    close_connections(all_threads=True)
    gc.collect()
    assert len(_all_connections) == 0

def test_repeated_connection_creation():
    """Test repeated connection creation and cleanup doesn't leak memory."""
    for _ in range(10):
        with _connect() as conn:
            conn.execute("SELECT 1")
        close_connections()
        gc.collect()
        
    assert len(_all_connections) == 0

def test_connection_tracking_normal_ops():
    """Test tracking remains correct after normal ops."""
    with _connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test_leak (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO test_leak DEFAULT VALUES")
        assert conn in _all_connections
    
    # Check that after context manager it's still alive (in pool)
    gc.collect()
    assert len([c for c in _all_connections]) > 0
    close_connections()
    gc.collect()
    assert len([c for c in _all_connections]) == 0

def test_cleanup_behavior():
    """Test _cleanup_all_connections closes and removes everything."""
    with _connect() as conn:
        pass
    
    _cleanup_all_connections()
    # It clears the weakset entirely
    assert len(_all_connections) == 0
    # Also verify connection is closed
    with pytest.raises(sqlite3.ProgrammingError):
        # We need a reference to the conn to test it, so let's get it again
        pass

def test_cleanup_behavior_close():
    with _connect() as conn:
        pass
        
    assert conn in _all_connections
    _cleanup_all_connections()
    assert len(_all_connections) == 0
    
    with pytest.raises(sqlite3.ProgrammingError, match="Cannot operate on a closed database"):
        conn.execute("SELECT 1")
        
def test_regression_memory_leak():
    """Regression test for the original memory leak scenario."""
    # Simulate a web request or background task pattern
    def simulated_task():
        with _connect() as conn:
            conn.execute("SELECT 1")
        # Framework drops the thread or we manually close pool
        close_connections()
        
    for _ in range(50):
        t = threading.Thread(target=simulated_task)
        t.start()
        t.join()
        
    gc.collect()
    # If there was a memory leak, _all_connections would grow to 50
    assert len(_all_connections) == 0
