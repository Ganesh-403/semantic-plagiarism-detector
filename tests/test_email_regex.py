"""
Comprehensive Unit Tests for Email Regex Validation
Issue: #3451
Tests valid formats, invalid formats, edge cases, and fuzz testing.
"""

import pytest
import re
import random
import string


# ==============================================================================
# SECTION 1: Defining the Validation Logic (Under Test)
# ==============================================================================

def is_valid_email(email: str) -> bool:
    """
    Validates an email address using a standard regex pattern.
    """
    if not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# ==============================================================================
# SECTION 2: Testing Valid Emails
# ==============================================================================

class TestValidEmails:
    def test_standard_email(self):
        assert is_valid_email("user@example.com") is True

    def test_email_with_dots(self):
        assert is_valid_email("first.last@domain.co") is True

    def test_email_with_plus(self):
        assert is_valid_email("user+tag@example.com") is True

    def test_email_with_numbers(self):
        assert is_valid_email("user123@example.org") is True

    def test_email_uppercase(self):
        assert is_valid_email("USER@EXAMPLE.COM") is True

    def test_email_subdomain(self):
        assert is_valid_email("user@mail.example.com") is True

    def test_email_with_underscore(self):
        assert is_valid_email("first_name@example.com") is True

    def test_email_with_hyphen(self):
        assert is_valid_email("user-name@example.com") is True

    def test_email_short_tld(self):
        assert is_valid_email("user@example.io") is True

    def test_email_long_tld(self):
        assert is_valid_email("user@example.international") is True


# ==============================================================================
# SECTION 3: Testing Invalid Emails
# ==============================================================================

class TestInvalidEmails:
    def test_missing_at_symbol(self):
        assert is_valid_email("userexample.com") is False

    def test_missing_domain(self):
        assert is_valid_email("user@") is False

    def test_missing_tld(self):
        assert is_valid_email("user@example") is False

    def test_double_at(self):
        assert is_valid_email("user@@example.com") is False

    def test_space_in_email(self):
        assert is_valid_email("user @example.com") is False

    def test_leading_space(self):
        assert is_valid_email(" user@example.com") is False

    def test_trailing_space(self):
        assert is_valid_email("user@example.com ") is False

    def test_empty_string(self):
        assert is_valid_email("") is False

    def test_non_string_input(self):
        assert is_valid_email(12345) is False

    def test_none_input(self):
        assert is_valid_email(None) is False

    def test_missing_username(self):
        assert is_valid_email("@example.com") is False

    def test_only_at(self):
        assert is_valid_email("@") is False

    def test_special_characters_in_domain(self):
        assert is_valid_email("user@exa!mple.com") is False

    def test_multiple_dots_at_end(self):
        assert is_valid_email("user@example..com") is False


# ==============================================================================
# SECTION 4: Edge Cases and Fuzz Testing
# ==============================================================================

class TestEdgeCases:
    def test_email_with_newline(self):
        assert is_valid_email("user@example.com\n") is False

    def test_email_with_tab(self):
        assert is_valid_email("\tuser@example.com") is False

    def test_very_long_email(self):
        long_email = "a" * 200 + "@example.com"
        assert is_valid_email(long_email) is True

    def test_shortest_valid_email(self):
        assert is_valid_email("a@b.co") is True


class TestRandomizedFuzz:
    def test_fuzz_valid_emails(self):
        """Randomly generated valid emails should pass."""
        random.seed(42)
        for _ in range(100):
            username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            domain = ''.join(random.choices(string.ascii_lowercase, k=8))
            tld = ''.join(random.choices(string.ascii_lowercase, k=3))
            email = f"{username}@{domain}.{tld}"
            assert is_valid_email(email) is True

    def test_fuzz_invalid_emails(self):
        """Randomly generated strings should almost never be valid."""
        random.seed(99)
        for _ in range(100):
            random_string = ''.join(random.choices(string.ascii_letters + string.punctuation + string.digits, k=15))
            # Very unlikely to match standard email pattern
            assert is_valid_email(random_string) is False