import sqlite3
import pytest
from typing import List, Set

# --- Simulated Migration Registry Core ---
AUTH_SCHEMA_VERSION = 3

def run_migrations_up(conn: sqlite3.Connection, target_version: int) -> None:
    """Executes schema creation steps sequentially up to the specified target version."""
    cursor = conn.cursor()
    
    if target_version >= 1:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
    if target_version >= 2:
        # Version 2: Multi-factor authentication onboarding columns
        cursor.execute("ALTER TABLE users ADD COLUMN mfa_secret TEXT;")
        cursor.execute("ALTER TABLE users ADD COLUMN is_mfa_enabled INTEGER DEFAULT 0;")
        
    if target_version >= 3:
        # Version 3: Audit tracking references schema allocation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ip_address TEXT,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)
    conn.commit()

# --- Helper Methods for Database Catalog Introspection ---
def get_database_tables(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return {row[0] for row in cursor.fetchall()}

def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]

# --- Migration Tests Suite ---

@pytest.fixture(scope="function")
def temp_db_connection():
    """Provides an isolated, transient, in-memory SQLite database connection."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


def test_migration_lifecycle_reaches_current_auth_schema_version(temp_db_connection):
    """Verify migrations compile smoothly from scratch up to the current AUTH_SCHEMA_VERSION."""
    conn = temp_db_connection
    
    # Run full migration pipeline pipeline directly to head
    run_migrations_up(conn, target_version=AUTH_SCHEMA_VERSION)
    
    # Assert expected systemic tables populate correctly inside the schema layout
    active_tables = get_database_tables(conn)
    expected_tables = {"users", "auth_audit_logs"}
    assert expected_tables.issubset(active_tables), f"Missing core tables. Present: {active_tables}"


def test_users_table_column_evolution(temp_db_connection):
    """Verify ALTER TABLE structural columns append correctly without breaking integrity rules."""
    conn = temp_db_connection
    
    run_migrations_up(conn, target_version=AUTH_SCHEMA_VERSION)
    
    # Introspect table attributes
    user_columns = get_table_columns(conn, "users")
    
    # Validate structural baseline columns exist
    assert "id" in user_columns
    assert "email" in user_columns
    
    # Validate version 2 migration evolution columns were injected safely
    assert "mfa_secret" in user_columns
    assert "is_mfa_enabled" in user_columns


def test_invalid_syntax_migration_rejections(temp_db_connection):
    """Verify syntax structural errors break migration runs predictably."""
    conn = temp_db_connection
    
    # Simulation: Developer inputs bad SQL command syntax (e.g. ALTER TABLE with invalid syntax keywords)
    with pytest.raises(sqlite3.OperationalError):
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE users INJECT COLUMN broken_syntax TEXT;")
