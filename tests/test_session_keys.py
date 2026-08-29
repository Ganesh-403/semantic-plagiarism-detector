import pytest
from app.session_keys import SessionKeys

def test_session_keys_uniqueness():
    """Verify that all values defined in SessionKeys are unique to prevent session state collisions."""
    values = [member.value for member in SessionKeys]
    unique_values = set(values)
    
    duplicates = [val for val in unique_values if values.count(val) > 1]
    
    assert len(values) == len(unique_values), (
        f"SessionKeys contains duplicate string values, which will cause session state collisions: {duplicates}"
    )

def test_session_keys_are_strings():
    """Verify that all SessionKeys attributes evaluate to strings."""
    for member in SessionKeys:
        assert isinstance(member.value, str), f"SessionKey {member.name} value is not a string."
