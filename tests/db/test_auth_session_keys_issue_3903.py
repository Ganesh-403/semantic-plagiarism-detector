"""Regression coverage for issue #3903.

``src/db/auth.py`` read the logged-in username out of Streamlit session state
through a ``SessionKeys`` enum in three places::

    2031:  username = st.session_state.get(SessionKeys.USERNAME)  # noqa: F821
    2060:  username = st.session_state.get(SessionKeys.USERNAME)  # noqa: F821
    2783:  admin    = st.session_state.get(SessionKeys.USERNAME)  # noqa: F821

``SessionKeys`` lives in ``app/session_keys.py`` and was never imported into
``src/db/auth.py``.  The names sit inside function bodies, so the module still
imported cleanly -- the ``NameError`` only fired the first time a decorated view
was actually called.  That covers the two RBAC decorators that gate every
privileged screen (``require_permission``, ``require_role``) and the admin role
selector (``render_role_selector``).

The three ``# noqa: F821`` comments are why this survived review: flake8 had
flagged all three undefined names and the warnings were suppressed.

These tests do not merely assert that the name now resolves.  Each of the three
call sites is driven with a faked ``st.session_state`` through the
authenticated, unauthenticated and insufficient-privilege paths, so a future
change that resolves the name but breaks the gate still fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from app.session_keys import SessionKeys

AUTH_PATH = Path(__file__).resolve().parents[2] / "src" / "db" / "auth.py"


class FakeSessionState(dict):
    """Minimal stand-in for ``st.session_state``.

    Streamlit's real session state supports both attribute and item access.
    ``SessionKeys`` subclasses ``str``, so a member and its plain-string value
    address the same slot -- the fake inherits that from ``dict`` for free,
    which is exactly the property the production code relies on.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class RecordingStreamlit:
    """Captures the ``st.error`` / ``st.success`` calls auth makes."""

    def __init__(self, session_state: FakeSessionState) -> None:
        self.session_state = session_state
        self.errors: list[str] = []
        self.successes: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(str(message))

    def success(self, message: str) -> None:
        self.successes.append(str(message))

    def rerun(self) -> None:  # pragma: no cover - not reached in these tests
        raise AssertionError("st.rerun() should not be called in these tests")


@pytest.fixture()
def auth():
    import src.db.auth as auth_module

    return auth_module


@pytest.fixture()
def fake_st(auth, monkeypatch):
    """Swap auth's module-level ``st`` for a recorder with empty session state."""
    stub = RecordingStreamlit(FakeSessionState())
    monkeypatch.setattr(auth, "st", stub)
    return stub


class TestSessionKeysIsImported:
    """The literal defect: the name had no binding in the module."""

    def test_auth_module_binds_session_keys(self, auth) -> None:
        assert hasattr(auth, "SessionKeys"), (
            "src.db.auth does not bind SessionKeys; the RBAC decorators will "
            "raise NameError as soon as they are called"
        )

    def test_it_is_the_real_enum(self, auth) -> None:
        assert auth.SessionKeys is SessionKeys
        assert auth.SessionKeys.USERNAME == "username"

    def test_import_is_at_module_scope(self) -> None:
        """A function-local import would leave the three call sites broken."""
        tree = ast.parse(AUTH_PATH.read_text(encoding="utf-8-sig"))
        module_scope_imports = [
            node for node in tree.body if isinstance(node, ast.ImportFrom)
        ]
        sources = {
            node.module
            for node in module_scope_imports
            if any(a.name == "SessionKeys" for a in node.names)
        }
        assert sources == {"app.session_keys"}, (
            f"expected a module-scope 'from app.session_keys import SessionKeys', "
            f"found imports of SessionKeys from {sorted(sources) or 'nowhere'}"
        )

    def test_no_f821_suppressions_remain(self) -> None:
        """The `# noqa: F821` comments are how #3903 got past the linter."""
        offenders = [
            f"line {n}: {line.strip()}"
            for n, line in enumerate(
                AUTH_PATH.read_text(encoding="utf-8-sig").splitlines(), start=1
            )
            if "F821" in line
        ]
        assert not offenders, (
            "F821 (undefined name) suppressions remain in src/db/auth.py:\n"
            + "\n".join(offenders)
        )

    def test_no_bare_sessionkeys_reference_is_unbound(self, auth) -> None:
        """Every ``SessionKeys.X`` the module names is a real member."""
        tree = ast.parse(AUTH_PATH.read_text(encoding="utf-8-sig"))
        referenced = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "SessionKeys"
        }
        assert referenced, "expected at least one SessionKeys.* reference"
        unknown = referenced - set(SessionKeys.__members__)
        assert not unknown, f"auth.py references non-existent SessionKeys members: {sorted(unknown)}"


class TestRequirePermission:
    """Call site 1 -- ``require_permission`` (auth.py:2033)."""

    def test_allows_a_user_with_the_permission(self, auth, fake_st, monkeypatch) -> None:
        fake_st.session_state[SessionKeys.USERNAME] = "alice"
        monkeypatch.setattr(auth, "has_permission", lambda user, perm: True)

        @auth.require_permission(auth.Permission.VIEW_AUDIT_LOGS)
        def view() -> str:
            return "audit logs"

        assert view() == "audit logs"
        assert fake_st.errors == []

    def test_blocks_a_user_without_the_permission(self, auth, fake_st, monkeypatch) -> None:
        fake_st.session_state[SessionKeys.USERNAME] = "bob"
        monkeypatch.setattr(auth, "has_permission", lambda user, perm: False)

        @auth.require_permission(auth.Permission.VIEW_AUDIT_LOGS)
        def view() -> str:  # pragma: no cover - must not run
            raise AssertionError("wrapped function ran despite denial")

        assert view() is None
        assert any("Permission denied" in msg for msg in fake_st.errors)

    def test_blocks_an_anonymous_caller(self, auth, fake_st) -> None:
        """Empty session state -- this is the path that raised NameError."""

        @auth.require_permission(auth.Permission.VIEW_AUDIT_LOGS)
        def view() -> str:  # pragma: no cover - must not run
            raise AssertionError("wrapped function ran without a session")

        assert view() is None
        assert any("Authentication required" in msg for msg in fake_st.errors)

    def test_reads_the_username_under_the_sessionkeys_value(self, auth, fake_st, monkeypatch) -> None:
        """A plain-string key must satisfy the lookup.

        ``SessionKeys`` subclasses ``str``, so real Streamlit code that stored
        the username under ``"username"`` must still be seen by the decorator.
        """
        fake_st.session_state["username"] = "carol"
        seen: list[str] = []
        monkeypatch.setattr(
            auth, "has_permission", lambda user, perm: seen.append(user) or True
        )

        @auth.require_permission(auth.Permission.VIEW_AUDIT_LOGS)
        def view() -> str:
            return "ok"

        assert view() == "ok"
        assert seen == ["carol"]

    def test_preserves_function_metadata_and_arguments(self, auth, fake_st, monkeypatch) -> None:
        fake_st.session_state[SessionKeys.USERNAME] = "alice"
        monkeypatch.setattr(auth, "has_permission", lambda user, perm: True)

        @auth.require_permission(auth.Permission.VIEW_AUDIT_LOGS)
        def view(a: int, b: int = 2) -> int:
            """A documented view."""
            return a + b

        assert view.__name__ == "view"
        assert view.__doc__ == "A documented view."
        assert view(1) == 3
        assert view(1, b=10) == 11


class TestRequireRole:
    """Call site 2 -- ``require_role`` (auth.py:2062)."""

    def test_allows_a_sufficiently_privileged_role(self, auth, fake_st, monkeypatch) -> None:
        fake_st.session_state[SessionKeys.USERNAME] = "admin"
        monkeypatch.setattr(auth, "get_user_role", lambda user: "admin")

        @auth.require_role(auth.UserRole.ADMIN)
        def view() -> str:
            return "admin panel"

        assert view() == "admin panel"
        assert fake_st.errors == []

    def test_blocks_an_insufficient_role(self, auth, fake_st, monkeypatch) -> None:
        fake_st.session_state[SessionKeys.USERNAME] = "teacher"
        monkeypatch.setattr(auth, "get_user_role", lambda user: "user")

        @auth.require_role(auth.UserRole.ADMIN)
        def view() -> str:  # pragma: no cover - must not run
            raise AssertionError("wrapped function ran despite an insufficient role")

        assert view() is None
        assert any("Role required" in msg for msg in fake_st.errors)

    def test_blocks_an_anonymous_caller(self, auth, fake_st) -> None:
        @auth.require_role(auth.UserRole.ADMIN)
        def view() -> str:  # pragma: no cover - must not run
            raise AssertionError("wrapped function ran without a session")

        assert view() is None
        assert any("Authentication required" in msg for msg in fake_st.errors)

    def test_looks_the_role_up_for_the_session_user(self, auth, fake_st, monkeypatch) -> None:
        """The username the decorator resolves is the one it authorises."""
        fake_st.session_state[SessionKeys.USERNAME] = "dave"
        asked: list[str] = []

        def get_user_role(user: str) -> str:
            asked.append(user)
            return "admin"

        monkeypatch.setattr(auth, "get_user_role", get_user_role)

        @auth.require_role(auth.UserRole.ADMIN)
        def view() -> str:
            return "ok"

        assert view() == "ok"
        assert asked == ["dave"]


class TestRenderRoleSelector:
    """Call site 3 -- ``render_role_selector`` (auth.py:2785)."""

    @pytest.fixture()
    def selector_st(self, auth, monkeypatch):
        """``st`` with the widget surface ``render_role_selector`` uses."""
        stub = RecordingStreamlit(FakeSessionState())
        stub.selectbox = lambda *a, **k: "admin"   # the newly chosen role
        stub.button = lambda *a, **k: True          # "Update Role" was clicked
        stub.rerun = lambda: None
        monkeypatch.setattr(auth, "st", stub)
        return stub

    def test_passes_the_acting_admin_to_promote_user(
        self, auth, selector_st, monkeypatch
    ) -> None:
        """The audit trail depends on this username resolving.

        Before the fix this raised NameError the moment an administrator
        clicked "Update Role".
        """
        selector_st.session_state[SessionKeys.USERNAME] = "root"
        calls: list[tuple[str, Any, str]] = []

        def promote_user(username, new_role, admin_username):
            calls.append((username, new_role, admin_username))
            return True

        monkeypatch.setattr(auth, "promote_user", promote_user)
        monkeypatch.setattr(auth, "get_valid_roles", lambda: ["user", "teacher", "admin"])

        auth.render_role_selector("target_user", "user")

        assert len(calls) == 1
        username, new_role, admin_username = calls[0]
        assert username == "target_user"
        assert new_role == auth.UserRole.ADMIN
        assert admin_username == "root", (
            "the acting administrator was not read out of session state"
        )

    def test_reports_a_failed_promotion(self, auth, selector_st, monkeypatch) -> None:
        selector_st.session_state[SessionKeys.USERNAME] = "root"
        monkeypatch.setattr(auth, "promote_user", lambda *a, **k: False)
        monkeypatch.setattr(auth, "get_valid_roles", lambda: ["user", "teacher", "admin"])

        auth.render_role_selector("target_user", "user")

        assert any("Failed to update role" in msg for msg in selector_st.errors)
        assert selector_st.successes == []
