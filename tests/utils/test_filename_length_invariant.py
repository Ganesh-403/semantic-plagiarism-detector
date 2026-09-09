"""
tests/utils/test_filename_length_invariant.py
---------------------------------------------
Tests for the one promise every helper in ``src.utils.filename`` makes: a name
it returns is never longer than ``max_length``.

Two things broke that promise.

``MAX_FILENAME_LENGTH`` was lowered to 128 by the truncation fix, then quietly
raised back to 150 by an unrelated commit that added the streaming hash helper.
The tests asserting 128 were never updated, so they simply failed on main.

``unique_filename()`` computed its stem budget as
``max_length - len(extension) - len(suffix)`` and passed the result straight to
a slice. Once the collision counter grew long enough for that to go negative,
``stem[:-1]`` trimmed from the end of the stem instead of clamping to nothing,
and the candidate came out *longer* than the caller's limit.
"""

import pytest

from src.utils.filename import (
    MAX_FILENAME_LENGTH,
    sanitize_filename,
    sanitize_filename_mapping,
    unique_filename,
)


class TestMaxFilenameLengthConstant:
    """The default limit is 128 characters."""

    def test_default_limit_is_128(self):
        assert MAX_FILENAME_LENGTH == 128

    @pytest.mark.parametrize(
        "raw",
        [
            "a" * 400 + ".pdf",
            "café-résumé " * 40 + ".docx",
            "文档" * 200 + ".txt",
            "a" * 300 + "_doc1.pdf",
        ],
    )
    def test_result_fits_the_filesystem_byte_cap(self, raw):
        """Unicode names must also fit the filesystem byte limit."""
        result = sanitize_filename(raw)

        assert len(result) <= MAX_FILENAME_LENGTH
        assert len(result.encode("utf-8")) <= 255

    @pytest.mark.parametrize("length", [130, 200, 400, 1000])
    def test_long_names_are_truncated_to_the_default(self, length):
        result = sanitize_filename("a" * length + ".pdf")

        assert len(result) == MAX_FILENAME_LENGTH
        assert result.endswith(".pdf")

    def test_short_names_are_not_padded(self):
        assert sanitize_filename("report.pdf") == "report.pdf"


class TestSanitizeFilenameRespectsMaxLength:
    """Truncation holds for the default and for custom limits."""

    @pytest.mark.parametrize("max_length", [8, 12, 20, 50, 100, 128, 255])
    @pytest.mark.parametrize("stem_length", [1, 10, 200, 500])
    def test_result_never_exceeds_the_limit(self, max_length, stem_length):
        result = sanitize_filename("a" * stem_length + ".docx", max_length=max_length)
        assert len(result) <= max_length

    def test_distinct_long_names_stay_distinct(self):
        """Truncation appends a hash so two long names do not collapse."""
        first = sanitize_filename("a" * 300 + "_doc1.pdf")
        second = sanitize_filename("a" * 300 + "_doc2.pdf")

        assert first != second
        assert len(first) <= MAX_FILENAME_LENGTH
        assert len(second) <= MAX_FILENAME_LENGTH


class TestUniqueFilenameRespectsMaxLength:
    """The collision counter must never push a name past the limit."""

    def test_default_limit_holds_across_many_collisions(self):
        existing = set()
        name = None

        for _ in range(300):
            name = unique_filename("report.pdf", existing)
            assert len(name) <= MAX_FILENAME_LENGTH
            existing.add(name.casefold())

        assert name == "report_299.pdf"

    @pytest.mark.parametrize("max_length", [10, 16, 24, 40])
    def test_custom_limit_holds_across_many_collisions(self, max_length):
        existing = set()

        for _ in range(150):
            name = unique_filename("report.pdf", existing, max_length=max_length)
            assert len(name) <= max_length
            existing.add(name.casefold())

    def test_long_stem_is_trimmed_to_fit_the_counter(self):
        """The stem gives way to the suffix, not the other way round."""
        existing = {sanitize_filename("a" * 300 + ".pdf").casefold()}

        name = unique_filename("a" * 300 + ".pdf", existing)

        assert len(name) <= MAX_FILENAME_LENGTH
        assert name.endswith("_1.pdf")

    def test_zero_stem_budget_still_produces_a_valid_name(self):
        """When only the suffix and extension fit, the stem drops away."""
        existing = {"ab.docx"} | {f"a_{index}.docx" for index in range(1, 10)}

        name = unique_filename("ab.docx", existing, max_length=8)

        assert len(name) <= 8
        assert name == "_10.docx"

    def test_exhausted_budget_raises_instead_of_overflowing(self):
        """A negative budget used to return an over-long name silently."""
        existing = {"ab.docx"}
        existing |= {f"a_{index}.docx" for index in range(1, 10)}
        existing |= {f"_{index}.docx" for index in range(10, 100)}

        with pytest.raises(ValueError, match="too small to disambiguate"):
            unique_filename("ab.docx", existing, max_length=8)

    def test_no_collision_returns_the_sanitized_name_untouched(self):
        assert unique_filename("report.pdf", set()) == "report.pdf"

    def test_collision_matching_is_case_insensitive(self):
        assert unique_filename("REPORT.PDF", {"report.pdf"}) == "REPORT_1.pdf"


class TestMappingRespectsMaxLength:
    """The mapping helper inherits the same invariant."""

    def test_every_key_fits_the_default_limit(self):
        files = {f"{'a' * 300}_{index}.pdf": index for index in range(20)}

        result = sanitize_filename_mapping(files)

        assert len(result) == len(files)
        for key in result:
            assert len(key) <= MAX_FILENAME_LENGTH

    @pytest.mark.parametrize("max_length", [10, 20, 64])
    def test_every_key_fits_a_custom_limit(self, max_length):
        files = {f"report_{index}.pdf": index for index in range(30)}

        result = sanitize_filename_mapping(files, max_length=max_length)

        assert len(result) == len(files)
        for key in result:
            assert len(key) <= max_length

    def test_no_entry_is_lost_to_a_collision(self):
        files = {
            "<b>report</b>.pdf": b"one",
            "report.pdf": b"two",
            "REPORT.pdf": b"three",
        }

        result = sanitize_filename_mapping(files)

        assert sorted(result.values()) == sorted([b"one", b"two", b"three"])
