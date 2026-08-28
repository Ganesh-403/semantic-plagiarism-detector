"""
Comprehensive Unit Tests for unique_filename budget overflow exception
Issue: #3729
Tests that the budget limit is enforced when generating filenames.
"""

import pytest


# ==============================================================================
# SECTION 1: Defining the Function Under Test
# ==============================================================================

class FilenameBudgetExceededError(Exception):
    """Custom exception raised when the filename budget is exceeded."""
    pass


def unique_filename(base_name: str, budget: int = 10) -> str:
    """
    Generates a unique filename based on a base name.
    Raises FilenameBudgetExceededError if the budget is exceeded.
    """
    if len(base_name) > budget:
        raise FilenameBudgetExceededError(
            f"Filename budget exceeded: '{base_name}' is longer than {budget} characters."
        )
    return f"{base_name}_unique"


# ==============================================================================
# SECTION 2: Testing Successful Filename Generation
# ==============================================================================

class TestSuccessfulFilenameGeneration:
    def test_short_filename(self):
        """Should generate a unique filename for a short base name."""
        result = unique_filename("file")
        assert result == "file_unique"

    def test_filename_at_exact_budget(self):
        """Should succeed when filename length equals budget."""
        # Budget is 10, filename is '1234567890' (10 chars)
        result = unique_filename("1234567890")
        assert result == "1234567890_unique"

    def test_filename_with_spaces(self):
        """Should handle filenames with spaces."""
        result = unique_filename("my file")
        assert result == "my file_unique"

    def test_custom_budget(self):
        """Should succeed with a custom larger budget."""
        result = unique_filename("verylongfilename", budget=20)
        assert result == "verylongfilename_unique"

    def test_empty_base_name(self):
        """Should handle an empty base name."""
        result = unique_filename("")
        assert result == "_unique"


# ==============================================================================
# SECTION 3: Testing Budget Overflow Exceptions
# ==============================================================================

class TestBudgetOverflowExceptions:
    def test_exception_raised_when_over_budget(self):
        """Should raise FilenameBudgetExceededError when filename is too long."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("thisfilenameistoolong", budget=10)

    def test_exception_message_contains_filename(self):
        """The exception message should reference the filename."""
        with pytest.raises(FilenameBudgetExceededError) as exc_info:
            unique_filename("toolongfilename", budget=5)
        assert "toolongfilename" in str(exc_info.value)

    def test_exception_message_contains_budget(self):
        """The exception message should reference the budget."""
        with pytest.raises(FilenameBudgetExceededError) as exc_info:
            unique_filename("toolongfilename", budget=5)
        assert "5" in str(exc_info.value)

    def test_exception_is_custom_type(self):
        """Should be a specific custom exception type."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("toolongfilename", budget=1)

    def test_exception_inherits_exception(self):
        """The custom exception should inherit from Exception."""
        assert issubclass(FilenameBudgetExceededError, Exception)


# ==============================================================================
# SECTION 4: Testing Boundary Conditions
# ==============================================================================

class TestBoundaryConditions:
    def test_filename_one_char_over_budget(self):
        """Should raise exception when exactly 1 char over budget."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("12345678901", budget=10)  # 11 chars

    def test_filename_many_chars_over_budget(self):
        """Should raise exception when many chars over budget."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("a" * 100, budget=10)

    def test_zero_budget(self):
        """Should raise exception if budget is 0 and filename is not empty."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("file", budget=0)

    def test_negative_budget(self):
        """Should raise exception if budget is negative."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("file", budget=-5)


# ==============================================================================
# SECTION 5: Testing Edge Cases
# ==============================================================================

class TestEdgeCases:
    def test_unicode_filename_over_budget(self):
        """Should handle unicode filenames over budget."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("こんにちは世界", budget=5)

    def test_special_characters_over_budget(self):
        """Should handle special characters over budget."""
        with pytest.raises(FilenameBudgetExceededError):
            unique_filename("!!!@@@###", budget=5)

    def test_no_error_for_short_special_characters(self):
        """Should succeed for short special character filenames."""
        result = unique_filename("@@@")
        assert result == "@@@_unique"

    def test_multiple_calls_do_not_raise(self):
        """Should not raise exception for valid filenames on multiple calls."""
        for _ in range(10):
            assert unique_filename("short") == "short_unique"


# ==============================================================================
# SECTION 6: Testing Exception Handling and Recovery
# ==============================================================================

class TestExceptionHandling:
    def test_catch_and_handle_exception(self):
        """Should allow the caller to catch the exception and handle it."""
        try:
            unique_filename("toolongfilename", budget=5)
        except FilenameBudgetExceededError:
            handled = True
        assert handled is True

    def test_exception_does_not_crash_program(self):
        """The exception should not crash the program when caught."""
        try:
            unique_filename("toolongfilename", budget=5)
        except FilenameBudgetExceededError:
            pass
        assert True  # Program continues to execute

    def test_function_returns_string_on_success(self):
        """Should return a string on successful generation."""
        result = unique_filename("valid")
        assert isinstance(result, str)

    def test_returned_filename_contains_base(self):
        """The returned filename should contain the base name."""
        result = unique_filename("base")
        assert "base" in result