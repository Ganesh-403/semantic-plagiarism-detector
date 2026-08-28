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

"""
test_legacy_bcrypt_migration.py
----------------------------------
Comprehensive unit tests for Issue #2706: Verifying automatic transparent migration
from legacy bcrypt hashes ($2b$, $2a$, $2y$) to modern Argon2id hashes during user login authentication.
"""

import os
import tempfile

import bcrypt
import pytest

from src.db.auth import (
    _connect,
    _verify_password_hash,
    configure_db_path,
    init_db,
    verify_user,
)


@pytest.fixture(autouse=True)
def temp_auth_db():
    """Fixture providing an isolated temporary SQLite authentication database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    configure_db_path(db_path)
    init_db()

    yield db_path

    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except OSError:
        pass


def test_legacy_bcrypt_2b_hash_transparent_migration_to_argon2():
    """Verify that a user inserted into the database with a legacy $2b$ bcrypt hash
    authenticates successfully and transparently migrates their stored hash to Argon2 ($argon2id...).
    """
    username = "legacy_bcrypt_user"
    plain_password = "LegacyBcryptPassword123!"

    # Step 1: Generate a valid legacy $2b$ bcrypt hash
    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")
    assert bcrypt_hash.startswith("$2b$")

    # Step 2: Directly insert mock user with legacy $2b$ hash into database
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, 'active')
            """,
            (username, bcrypt_hash, "teacher"),
        )
        conn.commit()

    # Step 3: Verify initial state in DB
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        assert row[0] == bcrypt_hash
        assert row[0].startswith("$2b$")

    # Step 4: Perform user login via verify_user()
    is_authenticated = verify_user(username, plain_password)
    assert is_authenticated is True

    # Step 5: Assert that stored password hash in DB has been transparently migrated to Argon2
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        updated_hash = row[0]

    assert updated_hash.startswith("$argon2")
    assert updated_hash != bcrypt_hash

    # Step 6: Verify future login succeeds with migrated Argon2 hash
    reauth_success = verify_user(username, plain_password)
    assert reauth_success is True


def test_legacy_bcrypt_2a_hash_transparent_migration_to_argon2():
    """Verify that a user inserted with a legacy $2a$ bcrypt hash authenticates
    successfully and migrates their hash to Argon2id."""
    username = "legacy_2a_user"
    plain_password = "Legacy2aPassword123!"

    salt = bcrypt.gensalt(rounds=10, prefix=b"2a")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")
    assert bcrypt_hash.startswith("$2a$")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, 'active')
            """,
            (username, bcrypt_hash, "teacher"),
        )
        conn.commit()

    # Perform login
    res = verify_user(username, plain_password, return_details=True)
    assert isinstance(res, dict)
    assert res["authenticated"] is True

    # Check migration
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        migrated_hash = row[0]

    assert migrated_hash.startswith("$argon2")


def test_legacy_bcrypt_2y_hash_transparent_migration_to_argon2():
    """Verify that a user inserted with a legacy $2y$ bcrypt hash authenticates
    successfully and migrates their hash to Argon2id."""
    username = "legacy_2y_user"
    plain_password = "Legacy2yPassword123!"

    # Generate a $2b$ hash and convert prefix to $2y$ for PHP-compatibility testing
    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")
    bcrypt_2y_hash = "$2y$" + bcrypt_hash[4:]

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, 'active')
            """,
            (username, bcrypt_2y_hash, "teacher"),
        )
        conn.commit()

    # Perform login
    assert verify_user(username, plain_password) is True

    # Check migration to argon2
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        migrated_hash = row[0]

    assert migrated_hash.startswith("$argon2")


def test_legacy_bcrypt_invalid_password_does_not_migrate():
    """Verify that an invalid password attempt against a legacy bcrypt account fails
    and does NOT update/migrate the stored password hash."""
    username = "bcrypt_invalid_pass_user"
    correct_password = "CorrectPassword123!"
    wrong_password = "WrongPassword999!"

    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(correct_password.encode("utf-8"), salt).decode("utf-8")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, 'active')
            """,
            (username, bcrypt_hash, "teacher"),
        )
        conn.commit()

    # Attempt login with wrong password
    assert verify_user(username, wrong_password) is False

    # Assert hash remains unchanged legacy bcrypt hash
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        assert row[0] == bcrypt_hash
        assert row[0].startswith("$2b$")


def test_legacy_bcrypt_suspended_user_does_not_migrate():
    """Verify that a suspended user with a legacy bcrypt hash is rejected and hash is not migrated."""
    username = "bcrypt_suspended_user"
    plain_password = "SuspendedPass123!"

    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, 'suspended')
            """,
            (username, bcrypt_hash, "teacher"),
        )
        conn.commit()

    assert verify_user(username, plain_password) is False

    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        assert row[0] == bcrypt_hash


def test_legacy_bcrypt_inactive_user_does_not_migrate():
    """Verify that an inactive/suspended user with a legacy bcrypt hash is rejected and not migrated."""
    username = "bcrypt_inactive_user"
    plain_password = "InactivePass123!"

    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, 'suspended')
            """,
            (username, bcrypt_hash, "teacher"),
        )
        conn.commit()

    assert verify_user(username, plain_password) is False

    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        assert row[0] == bcrypt_hash


def test_legacy_bcrypt_verify_password_hash_direct_helper():
    """Test the internal helper _verify_password_hash with bcrypt hashes."""
    plain_password = "HelperPassword123!"
    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

    assert _verify_password_hash(plain_password, bcrypt_hash) is True
    assert _verify_password_hash("WrongPass", bcrypt_hash) is False
    assert _verify_password_hash(plain_password, "") is False
    assert _verify_password_hash(plain_password, "malformed_hash") is False


def test_legacy_bcrypt_migration_with_must_change_password_flag():
    """Verify that transparent migration preserves must_change_password requirement during auth."""
    username = "bcrypt_must_change_user"
    plain_password = "MustChangePass123!"

    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status, must_change_password)
            VALUES (?, ?, ?, 'active', 1)
            """,
            (username, bcrypt_hash, "teacher"),
        )
        conn.commit()

    details = verify_user(username, plain_password, return_details=True)
    assert isinstance(details, dict)
    assert details["authenticated"] is True
    assert details["must_change_password"] is True

    # Confirm migration happened
    with _connect() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        assert row[0].startswith("$argon2")


def test_multiple_legacy_bcrypt_logins_idempotency():
    """Verify that multiple consecutive logins after initial migration stay on Argon2."""
    username = "bcrypt_idempotent_user"
    plain_password = "IdempotentPass123!"

    salt = bcrypt.gensalt(rounds=10, prefix=b"2b")
    bcrypt_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, 'active')
            """,
            (username, bcrypt_hash, "teacher"),
        )
        conn.commit()

    # First login (triggers migration)
    assert verify_user(username, plain_password) is True

    with _connect() as conn:
        hash_after_first = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()[0]
    assert hash_after_first.startswith("$argon2")

    # Second login
    assert verify_user(username, plain_password) is True

    with _connect() as conn:
        hash_after_second = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()[0]
    assert hash_after_second.startswith("$argon2")


def test_batch_legacy_bcrypt_users_migration_simulation():
    """Simulate migrating a batch of legacy bcrypt users across multiple environments."""
    users_data = [
        ("user1", "PassOne123!", "2b"),
        ("user2", "PassTwo123!", "2a"),
        ("user3", "PassThree123!", "2b"),
    ]

    for uname, pword, prefix in users_data:
        salt = bcrypt.gensalt(rounds=8, prefix=prefix.encode("utf-8"))
        b_hash = bcrypt.hashpw(pword.encode("utf-8"), salt).decode("utf-8")
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password, role, status) VALUES (?, ?, 'teacher', 'active')",
                (uname, b_hash),
            )
            conn.commit()

    # Authenticate all
    for uname, pword, _ in users_data:
        assert verify_user(uname, pword) is True

    # Assert all migrated to Argon2
    with _connect() as conn:
        rows = conn.execute(
            "SELECT username, password FROM users WHERE username IN ('user1', 'user2', 'user3')"
        ).fetchall()
        for u, h in rows:
            assert h.startswith("$argon2")
