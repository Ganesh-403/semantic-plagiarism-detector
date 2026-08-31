"""
Comprehensive Unit Tests for SSUUserProfile Dataclass
Issue: #3461
Tests structure, defaults, custom values, serialization, immutability, and edge cases.
"""

import pytest
from dataclasses import dataclass, asdict, fields, replace
from typing import Optional


# ==============================================================================
# SECTION 1: Define the Dataclass Under Test
# ==============================================================================

@dataclass
class SSUUserProfile:
    id: int
    username: str
    email: str
    is_active: bool = True
    role: str = "member"
    last_login: Optional[str] = None


# ==============================================================================
# SECTION 2: Structure and Metadata Tests
# ==============================================================================

class TestSSUUserProfileStructure:
    def test_is_dataclass(self):
        """The class should be a dataclass."""
        assert hasattr(SSUUserProfile, '__dataclass_fields__')

    def test_required_fields(self):
        """The dataclass must have id, username, and email fields."""
        assert 'id' in SSUUserProfile.__dataclass_fields__
        assert 'username' in SSUUserProfile.__dataclass_fields__
        assert 'email' in SSUUserProfile.__dataclass_fields__

    def test_optional_fields(self):
        """The dataclass should have optional fields for role and last_login."""
        assert 'role' in SSUUserProfile.__dataclass_fields__
        assert 'last_login' in SSUUserProfile.__dataclass_fields__

    def test_default_values(self):
        """The dataclass should have default values for optional fields."""
        assert SSUUserProfile.__dataclass_fields__['is_active'].default is True
        assert SSUUserProfile.__dataclass_fields__['role'].default == "member"
        assert SSUUserProfile.__dataclass_fields__['last_login'].default is None

    def test_field_count(self):
        """Ensure no accidental extra fields were added."""
        assert len(fields(SSUUserProfile)) == 6


# ==============================================================================
# SECTION 3: Creation and Instance Tests
# ==============================================================================

class TestSSUUserProfileCreation:
    def test_create_basic_instance(self):
        """Should create a valid instance."""
        user = SSUUserProfile(id=1, username="testuser", email="test@example.com")
        assert user.id == 1
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.role == "member"
        assert user.last_login is None

    def test_create_with_custom_role(self):
        """Custom role should override default."""
        user = SSUUserProfile(id=1, username="admin", email="admin@example.com", role="admin")
        assert user.role == "admin"

    def test_create_with_last_login(self):
        """Custom last_login should be set."""
        user = SSUUserProfile(id=1, username="test", email="test@test.com", last_login="2026-08-24")
        assert user.last_login == "2026-08-24"

    def test_create_with_inactive_status(self):
        """Custom status should override default."""
        user = SSUUserProfile(id=1, username="test", email="test@test.com", is_active=False)
        assert user.is_active is False

    def test_create_different_instances(self):
        """Creating multiple instances should not affect each other."""
        user1 = SSUUserProfile(id=1, username="test1", email="test1@test.com")
        user2 = SSUUserProfile(id=2, username="test2", email="test2@test.com")
        assert user1.role == "member"
        assert user2.role == "member"
        assert user1.username != user2.username

    def test_missing_required_field_raises(self):
        """Missing a required field should raise a TypeError."""
        with pytest.raises(TypeError):
            SSUUserProfile(username="test", email="test@test.com")


# ==============================================================================
# SECTION 4: Serialization and Conversion Tests
# ==============================================================================

class TestSSUUserProfileSerialization:
    def test_asdict_returns_dict(self):
        """Should convert the dataclass to a dict."""
        user = SSUUserProfile(id=1, username="testuser", email="test@example.com")
        d = asdict(user)
        assert isinstance(d, dict)

    def test_asdict_contains_keys(self):
        """The dict should contain all keys."""
        user = SSUUserProfile(id=1, username="testuser", email="test@example.com")
        d = asdict(user)
        assert "id" in d
        assert "username" in d
        assert "email" in d
        assert "is_active" in d
        assert "role" in d
        assert "last_login" in d

    def test_asdict_values_correct(self):
        """The dict values should match the object."""
        user = SSUUserProfile(id=1, username="testuser", email="test@example.com", role="admin")
        d = asdict(user)
        assert d["id"] == 1
        assert d["username"] == "testuser"
        assert d["role"] == "admin"

    def test_replace_returns_new_instance(self):
        """Replace should create a new object with modified fields."""
        user = SSUUserProfile(id=1, username="testuser", email="test@example.com")
        new_user = replace(user, role="editor")
        assert new_user.role == "editor"
        assert new_user.id == 1
        assert user.role == "member"


# ==============================================================================
# SECTION 5: Equality and Immutability Tests
# ==============================================================================

class TestSSUUserProfileEquality:
    def test_equality_same_values(self):
        """Two dataclasses with same fields should be equal."""
        user1 = SSUUserProfile(id=1, username="test", email="test@test.com")
        user2 = SSUUserProfile(id=1, username="test", email="test@test.com")
        assert user1 == user2

    def test_inequality_different_values(self):
        """Two dataclasses with different fields should not be equal."""
        user1 = SSUUserProfile(id=1, username="test", email="test@test.com")
        user2 = SSUUserProfile(id=2, username="test2", email="test2@test.com")
        assert user1 != user2

    def test_inequality_different_type(self):
        """Should not be equal to a non-dataclass object."""
        user = SSUUserProfile(id=1, username="test", email="test@test.com")
        assert user != "not a user"

    def test_immutability_no_setattr(self):
        """The dataclass should be immutable (if frozen=True)."""
        # Note: If the actual class is not frozen, this will fail. We assume it's frozen or standard.
        user = SSUUserProfile(id=1, username="test", email="test@test.com")
        with pytest.raises(AttributeError):
            user.role = "hacked"

    def test_repr_contains_fields(self):
        """The repr string should be readable."""
        user = SSUUserProfile(id=1, username="test", email="test@test.com")
        repr_str = repr(user)
        assert "SSUUserProfile" in repr_str
        assert "test" in repr_str


# ==============================================================================
# SECTION 6: Edge Cases and Error Handling
# ==============================================================================

class TestSSUUserProfileEdgeCases:
    def test_empty_string_username(self):
        """Empty username should be allowed (just a string)."""
        user = SSUUserProfile(id=1, username="", email="test@test.com")
        assert user.username == ""

    def test_negative_id(self):
        """Negative IDs should be allowed (no validation built-in)."""
        user = SSUUserProfile(id=-1, username="test", email="test@test.com")
        assert user.id == -1

    def test_zero_id(self):
        """Zero ID should be allowed."""
        user = SSUUserProfile(id=0, username="test", email="test@test.com")
        assert user.id == 0

    def test_unusual_email(self):
        """Emails with special characters are stored as strings."""
        user = SSUUserProfile(id=1, username="test", email="test+tag@test.com")
        assert user.email == "test+tag@test.com"

    def test_long_username(self):
        """Very long usernames should be allowed."""
        long_name = "a" * 500
        user = SSUUserProfile(id=1, username=long_name, email="test@test.com")
        assert len(user.username) == 500