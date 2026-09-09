"""Provider profile regressions against SSOUserProfile, not a local stand-in."""

from dataclasses import asdict
from src.utils.sso import SSOUserProfile


def test_provider_profile_round_trip():
    profile = SSOUserProfile("alice@example.com", "alice", "Alice", "avatar", "provider-123")
    restored = SSOUserProfile(**asdict(profile))
    assert restored == profile
    assert restored.provider_user_id == "provider-123"


def test_legacy_profile_constructor_remains_compatible():
    profile = SSOUserProfile("alice@example.com", "alice", "Alice", "")
    assert profile.provider_user_id == ""
    assert profile.email == "alice@example.com"


def test_profile_mutation_does_not_change_other_users():
    alice = SSOUserProfile("alice@example.com", "alice", "Alice", "")
    bob = SSOUserProfile("bob@example.com", "bob", "Bob", "")
    alice.avatar = "alice-avatar"
    assert bob.avatar == ""
    assert alice != bob
