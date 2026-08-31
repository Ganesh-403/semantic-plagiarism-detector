import pytest
from src.database import _connect  # Adjust import based on your project structure


def test_sqlite_transaction_rollback_on_exception():
    """
    Verify that transactions are automatically rolled back when an exception
    is raised inside a `with _connect() as conn:` block, preventing partial data writes.
    """
    # 1. Ensure the test user does not exist beforehand
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = 'rollback_test_user'")
        conn.commit()

    # 2. Trigger an exception inside the context manager after attempting an insert
    with pytest.raises(RuntimeError, match="Forced exception for rollback test"):
        with _connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                ("rollback_test_user", "rollback@example.com"),
            )
            # Raise an exception before transaction commits
            raise RuntimeError("Forced exception for rollback test")

    # 3. Assert that the user was NOT persisted to the database due to rollback
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = 'rollback_test_user'")
        user = cursor.fetchone()

    assert user is None, (
        "Transaction failed to roll back; user record exists in the database!"
    )
