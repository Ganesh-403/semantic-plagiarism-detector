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
tests/db/test_revoke_sso_access.py
----------------------------------
Regression tests for revoke_sso_access() and the duplicate demote_user
(Issue #3566).

revoke_sso_access() carried promote_user()'s success branch verbatim:

    if affected > 0:
        log_security_event(
            event_type="user_role_changed",
            username=username,
            details=f"Role changed to {new_role.value} by {admin_username}",
        )
        return True

Neither `new_role` nor `admin_username` exists in that scope, so evaluating
the f-string raised NameError. The bare `except Exception` swallowed it,
logged "Failed to promote user", and returned False -- *after* conn.commit()
had already cleared the SSO columns. The revocation succeeded, the caller was
told it had not, and no audit entry was written.

`demote_user` was also defined twice; the shadowed first definition targeted
`UserRole.MEMBER`, which the enum does not have.
"""

import sqlite3
import uuid

import pytest

from src.db.auth import (
    UserRole,
    _connect,
    add_user,
    demote_user,
    get_security_audit_logs,
    get_user_role,
    init_db,
    promote_user,
    revoke_sso_access,
)


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """Isolate DB operations and make sure the SSO columns exist.

    ``init_db()`` does not create ``sso_provider``, ``sso_provider_user_id``
    or ``updated_at``, and nothing in ``src/db/migrations`` adds them either
    -- a separate pre-existing gap that leaves every SSO function in this
    module unable to run against a freshly initialised database. These tests
    add the columns so the functions have a table to work against; nothing
    below depends on how the columns come to exist.
    """
    init_db()

    with _connect() as conn:
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        for column in ("sso_provider", "sso_provider_user_id", "updated_at"):
            if column not in existing:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT DEFAULT NULL")
        conn.commit()

    yield


def _make_sso_user(provider: str = "github") -> str:
    """Create a local user and link an SSO provider to it directly."""
    username = f"sso_{uuid.uuid4().hex[:8]}"
    add_user(username, "SecurePass123!")

    with _connect() as conn:
        conn.execute(
            """UPDATE users
               SET sso_provider = ?,
                   sso_provider_user_id = ?
               WHERE username = ?""",
            (provider, f"{provider}-12345", username),
        )
        conn.commit()

    return username


def _sso_columns(username: str):
    """Read the two SSO columns straight from the table."""
    with _connect() as conn:
        return conn.execute(
            "SELECT sso_provider, sso_provider_user_id FROM users WHERE username = ?",
            (username,),
        ).fetchone()


class TestRevokeSsoAccessReturnValue:
    """The return value must describe what actually happened to the row."""

    def test_revoking_a_linked_account_returns_true(self):
        """The regression itself: this returned False after committing."""
        username = _make_sso_user()

        assert revoke_sso_access(username) is True

    def test_revoking_clears_both_sso_columns(self):
        """The columns the function exists to clear are actually cleared."""
        username = _make_sso_user()

        revoke_sso_access(username)

        provider, provider_user_id = _sso_columns(username)
        assert provider is None
        assert provider_user_id is None

    def test_return_value_agrees_with_the_row(self):
        """True must not be reported for a row that did not change, or vice versa.

        This is the shape of the original bug: the commit landed and the
        return value contradicted it. Asserting both together is what catches
        a success path that reports failure.
        """
        username = _make_sso_user()

        result = revoke_sso_access(username)
        provider, _ = _sso_columns(username)

        assert result is (provider is None)

    def test_unknown_username_returns_false(self):
        """No such user is not a revocation."""
        assert revoke_sso_access(f"missing_{uuid.uuid4().hex[:8]}") is False

    def test_local_only_account_returns_false(self):
        """A user with no provider linked has nothing to revoke."""
        username = f"local_{uuid.uuid4().hex[:8]}"
        add_user(username, "SecurePass123!")

        assert revoke_sso_access(username) is False

    def test_second_revocation_returns_false(self):
        """Revoking twice is idempotent: the second call has nothing to do."""
        username = _make_sso_user()

        assert revoke_sso_access(username) is True
        assert revoke_sso_access(username) is False


class TestRevokeSsoAccessAuditTrail:
    """A revocation must leave a trace in the security audit log."""

    def test_revocation_writes_an_audit_event(self):
        """No entry was written at all while the NameError was being swallowed."""
        username = _make_sso_user()

        revoke_sso_access(username)

        events = get_security_audit_logs(username=username)
        assert any(event["event_type"] == "sso_access_revoked" for event in events)

    def test_audit_event_names_the_revoked_provider(self):
        """The provider is read before the update, while it is still readable."""
        username = _make_sso_user(provider="gitlab")

        revoke_sso_access(username)

        events = get_security_audit_logs(username=username)
        revocations = [e for e in events if e["event_type"] == "sso_access_revoked"]
        assert revocations
        assert "gitlab" in revocations[0]["details"]

    def test_audit_event_is_not_a_role_change(self):
        """The pasted branch logged 'user_role_changed', which is the wrong event."""
        username = _make_sso_user()

        revoke_sso_access(username)

        events = get_security_audit_logs(username=username)
        assert not any(e["event_type"] == "user_role_changed" for e in events)

    def test_no_audit_event_when_there_was_nothing_to_revoke(self):
        """A no-op must not pollute the audit log."""
        username = f"local_{uuid.uuid4().hex[:8]}"
        add_user(username, "SecurePass123!")

        revoke_sso_access(username)

        events = get_security_audit_logs(username=username)
        assert not any(e["event_type"] == "sso_access_revoked" for e in events)


class TestRevokeSsoAccessLeavesTheAccountIntact:
    """Unlinking a provider is not the same as disabling the account."""

    def test_role_is_unchanged(self):
        """Revocation touches the SSO columns only."""
        username = _make_sso_user()
        role_before = get_user_role(username)

        revoke_sso_access(username)

        assert get_user_role(username) == role_before

    def test_user_still_exists(self):
        """The local account survives, so the user can still log in with a password."""
        username = _make_sso_user()

        revoke_sso_access(username)

        assert get_user_role(username) is not None

    def test_row_stops_matching_the_sso_predicate(self):
        """Every SSO lookup filters on `sso_provider IS NOT NULL`.

        Asserting against that predicate directly, rather than through
        get_sso_user_info(), keeps this test about the revocation instead of
        about the other columns that lookup happens to select.
        """
        username = _make_sso_user()

        def is_sso_account() -> bool:
            with _connect() as conn:
                return (
                    conn.execute(
                        "SELECT 1 FROM users "
                        "WHERE username = ? AND sso_provider IS NOT NULL",
                        (username,),
                    ).fetchone()
                    is not None
                )

        assert is_sso_account() is True
        revoke_sso_access(username)
        assert is_sso_account() is False

    def test_other_users_are_untouched(self):
        """The UPDATE is scoped to one username."""
        target = _make_sso_user(provider="github")
        bystander = _make_sso_user(provider="google")

        revoke_sso_access(target)

        provider, provider_user_id = _sso_columns(bystander)
        assert provider == "google"
        assert provider_user_id == "google-12345"


class TestRevokeSsoAccessErrorHandling:
    """Genuine failures still return False, and say what actually failed."""

    def test_database_error_returns_false(self, monkeypatch):
        """A real DB error is caught, not re-raised."""
        username = _make_sso_user()

        def failing_connect(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr("src.db.auth._connect", failing_connect)

        assert revoke_sso_access(username) is False

    def test_error_message_names_the_revocation(self, monkeypatch, caplog):
        """The handler used to log 'Failed to promote user'."""
        username = _make_sso_user()

        def failing_connect(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr("src.db.auth._connect", failing_connect)

        with caplog.at_level("ERROR"):
            revoke_sso_access(username)

        assert "revoke SSO access" in caplog.text
        assert "promote" not in caplog.text


class TestDemoteUserIsDefinedOnce:
    """The shadowed definition referenced a UserRole member that never existed."""

    def test_user_role_has_no_member_attribute(self):
        """UserRole is USER/TEACHER/ADMIN/SUPER_ADMIN -- there is no MEMBER."""
        assert not hasattr(UserRole, "MEMBER")
        assert {role.name for role in UserRole} == {
            "USER",
            "TEACHER",
            "ADMIN",
            "SUPER_ADMIN",
        }

    def test_module_defines_demote_user_exactly_once(self):
        """A second def would shadow the first and hide whichever one is wrong."""
        import ast
        import inspect

        import src.db.auth as auth_module

        tree = ast.parse(inspect.getsource(auth_module))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "demote_user"
        ]

        assert len(definitions) == 1

    def test_demote_user_sets_the_user_role(self):
        """The surviving definition demotes to 'user', not to a missing member."""
        admin = f"admin_{uuid.uuid4().hex[:8]}"
        add_user(admin, "SecurePass123!", role="admin")

        target = f"teacher_{uuid.uuid4().hex[:8]}"
        add_user(target, "SecurePass123!")
        promote_user(target, UserRole.TEACHER, admin)
        assert get_user_role(target) == "teacher"

        assert demote_user(target, admin) is True
        assert get_user_role(target) == UserRole.USER.value
