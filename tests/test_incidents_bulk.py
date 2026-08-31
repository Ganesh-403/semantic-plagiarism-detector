import pytest

from src import database  # Adjust based on your module structure


@pytest.fixture(autouse=True)
def setup_teardown(monkeypatch, tmp_path):
    """
    Refactored fixture to ensure isolated execution using either an in-memory SQLite database
    or a temporary database file that is automatically destroyed upon test completion.
    """
    # Option A: Using an in-memory SQLite database (or Option B using tmp_path)
    test_db = ":memory:"

    # Patch the database path or connection reference used by your app
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", test_db)

    # Initialize the database schema for the test session
    database.init_db()

    yield

    # Teardown logic if required (in-memory DBs destroy themselves automatically)
