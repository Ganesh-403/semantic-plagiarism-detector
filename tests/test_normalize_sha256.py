"""
Comprehensive Unit Tests for normalize_sha256_hash
Issue: #3730
Tests mixed-case handling, invalid digests, and security edge cases.
"""

import pytest
import hashlib


# ==============================================================================
# SECTION 1: Defining the Function Under Test
# ==============================================================================

def normalize_sha256_hash(hash_value: str) -> str:
    """
    Normalizes a SHA256 hash to lowercase.
    Raises ValueError if the hash is not a valid 64-character hex string.
    """
    if not isinstance(hash_value, str):
        raise ValueError("Hash must be a string")
    
    # Must be exactly 64 characters
    if len(hash_value) != 64:
        raise ValueError("Hash must be exactly 64 characters")
    
    # Must be valid hex characters (0-9, a-f, A-F)
    try:
        bytes.fromhex(hash_value)
    except ValueError:
        raise ValueError("Hash contains invalid hex characters")
    
    # Normalize to lowercase
    return hash_value.lower()


# ==============================================================================
# SECTION 2: Testing Valid Hashes
# ==============================================================================

class TestValidHashes:
    def test_lowercase_hash(self):
        """Should return unchanged for lowercase hash."""
        hash_val = "a" * 64
        assert normalize_sha256_hash(hash_val) == hash_val

    def test_uppercase_hash(self):
        """Should convert uppercase to lowercase."""
        hash_val = "A" * 64
        assert normalize_sha256_hash(hash_val) == "a" * 64

    def test_mixed_case_hash(self):
        """Should convert mixed-case to lowercase."""
        hash_val = "aBcDeF" * 10 + "AbCdEf"  # 64 chars total
        expected = hash_val.lower()
        assert normalize_sha256_hash(hash_val) == expected

    def test_real_sha256_hash(self):
        """Should handle a real SHA256 hash."""
        real_hash = hashlib.sha256(b"hello").hexdigest()  # 64 chars
        assert normalize_sha256_hash(real_hash) == real_hash

    def test_real_sha256_hash_uppercase(self):
        """Should handle a real SHA256 hash in uppercase."""
        real_hash = hashlib.sha256(b"test").hexdigest().upper()
        assert normalize_sha256_hash(real_hash) == real_hash.lower()


# ==============================================================================
# SECTION 3: Testing Invalid Hashes (Length)
# ==============================================================================

class TestInvalidHashLength:
    def test_too_short_hash(self):
        """Should raise error for hash shorter than 64 chars."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("abc")

    def test_too_long_hash(self):
        """Should raise error for hash longer than 64 chars."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("a" * 65)

    def test_empty_hash(self):
        """Should raise error for empty hash."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("")

    def test_single_character_hash(self):
        """Should raise error for single character hash."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("a")


# ==============================================================================
# SECTION 4: Testing Invalid Hashes (Characters)
# ==============================================================================

class TestInvalidHashCharacters:
    def test_hash_with_special_characters(self):
        """Should raise error for special characters."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("!" * 64)

    def test_hash_with_spaces(self):
        """Should raise error for spaces."""
        with pytest.raises(ValueError):
            normalize_sha256_hash(" " * 64)

    def test_hash_with_non_hex_letters(self):
        """Should raise error for non-hex letters (g-z)."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("g" * 64)

    def test_hash_with_symbols(self):
        """Should raise error for symbols."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("@" * 64)

    def test_hash_with_newlines(self):
        """Should raise error for newlines."""
        with pytest.raises(ValueError):
            normalize_sha256_hash("\n" * 64)


# ==============================================================================
# SECTION 5: Testing Input Types
# ==============================================================================

class TestInputTypes:
    def test_none_input(self):
        """Should raise error for None input."""
        with pytest.raises(ValueError):
            normalize_sha256_hash(None)

    def test_integer_input(self):
        """Should raise error for integer input."""
        with pytest.raises(ValueError):
            normalize_sha256_hash(12345)

    def test_float_input(self):
        """Should raise error for float input."""
        with pytest.raises(ValueError):
            normalize_sha256_hash(3.14)

    def test_list_input(self):
        """Should raise error for list input."""
        with pytest.raises(ValueError):
            normalize_sha256_hash(["a" * 64])

    def test_boolean_input(self):
        """Should raise error for boolean input."""
        with pytest.raises(ValueError):
            normalize_sha256_hash(True)


# ==============================================================================
# SECTION 6: Testing Security Properties
# ==============================================================================

class TestSecurityProperties:
    def test_output_is_always_lowercase(self):
        """The normalized output should always be lowercase."""
        test_hashes = [
            "A" * 64,
            "AbCd" * 16,
            hashlib.sha256(b"test").hexdigest().upper(),
        ]
        for h in test_hashes:
            result = normalize_sha256_hash(h)
            assert result == result.lower()

    def test_output_is_always_hex(self):
        """The normalized output should always be valid hex."""
        test_hashes = [
            "a" * 64,
            "B" * 64,
        ]
        for h in test_hashes:
            result = normalize_sha256_hash(h)
            # Verify it's valid hex
            int(result, 16)

    def test_output_length_is_always_64(self):
        """The normalized output should always be 64 characters."""
        test_hashes = [
            "a" * 64,
            "B" * 64,
        ]
        for h in test_hashes:
            result = normalize_sha256_hash(h)
            assert len(result) == 64

    def test_input_is_unchanged(self):
        """The function should not mutate the input."""
        hash_val = "A" * 64
        normalize_sha256_hash(hash_val)
        assert hash_val == "A" * 64


# ==============================================================================
# SECTION 7: Testing Exception Handling
# ==============================================================================

class TestExceptionHandling:
    def test_error_message_contains_length(self):
        """Error should mention the length issue."""
        with pytest.raises(ValueError) as exc_info:
            normalize_sha256_hash("too_short")
        assert "length" in str(exc_info.value).lower() or "64" in str(exc_info.value)

    def test_error_message_contains_hex(self):
        """Error should mention hex characters."""
        with pytest.raises(ValueError) as exc_info:
            normalize_sha256_hash("g" * 64)
        assert "hex" in str(exc_info.value).lower()

    def test_multiple_exceptions(self):
        """Should handle multiple invalid inputs."""
        invalid_inputs = ["", "a" * 10, "z" * 64, None, 123]
        for inp in invalid_inputs:
            with pytest.raises(ValueError):
                normalize_sha256_hash(inp)