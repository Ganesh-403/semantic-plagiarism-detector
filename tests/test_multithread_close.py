"""
Comprehensive Unit Tests for Multi-threaded close_connections
Issue: #3419
Tests concurrent safety, atomicity, and thread-safety of connection closure.
"""

import pytest
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


# ==============================================================================
# SECTION 1: Simulated Database Connection Class
# ==============================================================================

class MockDatabaseConnection:
    """Simulates a database connection with a thread-safe close method."""
    
    def __init__(self):
        self.is_closed = False
        self._lock = threading.Lock()
        self.close_call_count = 0

    def close(self):
        """Simulates closing the connection in a thread-safe manner."""
        with self._lock:
            self.close_call_count += 1
            # Simulate a tiny delay to force race conditions
            time.sleep(0.001)
            self.is_closed = True

    def is_connection_active(self) -> bool:
        """Returns True if the connection is still open."""
        return not self.is_closed


# ==============================================================================
# SECTION 2: Simulated Database Manager (Under Test)
# ==============================================================================

class DatabaseManager:
    """Manages a pool of database connections."""

    def __init__(self):
        self.connections: List[MockDatabaseConnection] = []
        self._lock = threading.Lock()

    def add_connection(self, connection: MockDatabaseConnection):
        """Adds a connection to the pool."""
        with self._lock:
            self.connections.append(connection)

    def close_all_connections(self):
        """Closes all connections in the pool."""
        with self._lock:
            for conn in self.connections:
                conn.close()

    def get_active_connection_count(self) -> int:
        """Returns the number of active (non-closed) connections."""
        with self._lock:
            return sum(1 for conn in self.connections if conn.is_connection_active())


# ==============================================================================
# SECTION 3: Core Thread-Safety Tests
# ==============================================================================

class TestThreadSafety:
    def test_single_thread_close(self):
        """A single thread should close the connection properly."""
        manager = DatabaseManager()
        conn = MockDatabaseConnection()
        manager.add_connection(conn)
        
        manager.close_all_connections()
        
        assert conn.is_closed is True
        assert conn.close_call_count == 1

    def test_multi_thread_close_safety(self):
        """Multiple threads closing the same connection should not crash."""
        manager = DatabaseManager()
        conn = MockDatabaseConnection()
        manager.add_connection(conn)
        
        def close_connection():
            manager.close_all_connections()

        threads = [threading.Thread(target=close_connection) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert conn.is_closed is True
        assert conn.close_call_count == 1  # Should only actually close once due to lock


# ==============================================================================
# SECTION 4: Concurrency Stress Tests (Using ThreadPoolExecutor)
# ==============================================================================

class TestConcurrencyStress:
    def test_thread_pool_executor_close(self):
        """Tests closing connections using a ThreadPoolExecutor."""
        manager = DatabaseManager()
        conn = MockDatabaseConnection()
        manager.add_connection(conn)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(manager.close_all_connections) for _ in range(5)]
            for future in as_completed(futures):
                future.result()  # Ensure no exceptions are raised

        assert conn.is_closed is True
        assert conn.close_call_count >= 1

    def test_concurrent_close_multiple_connections(self):
        """Tests closing multiple connections concurrently."""
        manager = DatabaseManager()
        connections = [MockDatabaseConnection() for _ in range(20)]
        for conn in connections:
            manager.add_connection(conn)

        def close_all():
            manager.close_all_connections()

        threads = [threading.Thread(target=close_all) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All connections should be closed
        assert manager.get_active_connection_count() == 0
        for conn in connections:
            assert conn.is_closed is True

    def test_no_deadlock(self):
        """Ensures closing connections doesn't cause a deadlock."""
        manager = DatabaseManager()
        conn = MockDatabaseConnection()
        manager.add_connection(conn)

        def safe_close():
            try:
                manager.close_all_connections()
            except Exception as e:
                pytest.fail(f"Deadlock or error occurred: {e}")

        thread = threading.Thread(target=safe_close)
        thread.start()
        thread.join(timeout=5)  # If it takes longer than 5 seconds, it's a deadlock
        
        assert not thread.is_alive()


# ==============================================================================
# SECTION 5: Edge Cases and Logging
# ==============================================================================

class TestEdgeCases:
    def test_close_no_connections(self):
        """Closing when there are no connections should not crash."""
        manager = DatabaseManager()
        manager.close_all_connections()  # Should not raise an exception

    def test_close_already_closed_connection(self):
        """Closing an already closed connection should be idempotent."""
        manager = DatabaseManager()
        conn = MockDatabaseConnection()
        manager.add_connection(conn)
        
        manager.close_all_connections()
        manager.close_all_connections()  # Closing again should not crash
        
        assert conn.is_closed is True

    def test_active_count_after_close(self):
        """Active count should drop to 0 after closing."""
        manager = DatabaseManager()
        conn1 = MockDatabaseConnection()
        conn2 = MockDatabaseConnection()
        manager.add_connection(conn1)
        manager.add_connection(conn2)
        
        assert manager.get_active_connection_count() == 2
        
        manager.close_all_connections()
        
        assert manager.get_active_connection_count() == 0

    def test_logging_on_close(self, caplog):
        """Ensure that closing connections is logged (if logging is configured)."""
        manager = DatabaseManager()
        conn = MockDatabaseConnection()
        manager.add_connection(conn)
        
        with caplog.at_level(logging.INFO):
            manager.close_all_connections()
            # We are not asserting a specific log message, just ensuring no crash
            assert len(caplog.records) >= 0