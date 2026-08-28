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
tests/utils/test_pagination.py
------------------------------
Unit tests for pagination utilities.

Validates PaginationPage dataclass behavior including __repr__, __eq__,
factory methods, and navigation helpers.
"""

import doctest

import pytest

from src.utils import pagination
from src.utils.pagination import PaginationPage, _coerce_integer, paginate_items


class TestPaginationPageReprClassName:
    """Regression tests for the class name in __repr__ (Issue #2200).

    __repr__ built its output from a hardcoded string that misspelled the
    class as "PagnationPage". Every log line, traceback, and debugger frame
    printed a class name that does not exist, so grepping logs for
    "PaginationPage" returned nothing.

    The pre-existing repr tests below all assert on substrings like
    "page=1/2" and "items=[1, 2, 3]" and never look at the class name, which
    is exactly how the typo survived. These tests close that gap.
    """

    def test_repr_starts_with_the_real_class_name(self):
        page = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )

        assert repr(page).startswith("PaginationPage(")

    def test_repr_does_not_contain_the_misspelling(self):
        page = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )

        assert "PagnationPage" not in repr(page)

    def test_repr_class_name_matches_the_type(self):
        """The name must be derived from the type, not hardcoded."""
        page = PaginationPage(
            items=[],
            page=1,
            total_pages=1,
            total_items=0,
            per_page=10,
        )

        assert repr(page).startswith(f"{type(page).__name__}(")

    def test_subclass_reports_its_own_name(self):
        """A hardcoded name would report the base class for any subclass."""

        class AuditLogPage(PaginationPage):
            pass

        page = AuditLogPage(
            items=[1, 2],
            page=1,
            total_pages=1,
            total_items=2,
            per_page=10,
        )

        assert repr(page).startswith("AuditLogPage(")
        assert "PaginationPage(" not in repr(page)

    def test_repr_includes_per_page(self):
        """per_page is a field but was invisible in the repr."""
        page = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )

        assert "per_page=5" in repr(page)

    def test_repr_distinguishes_pages_that_differ_only_by_page_size(self):
        """Without per_page these two rendered identically."""
        first = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )
        second = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=50,
        )

        assert repr(first) != repr(second)

    def test_repr_is_exactly_as_documented(self):
        """Pin the full string so the format cannot drift silently."""
        truncated = PaginationPage(
            items=[1, 2, 3, 4, 5],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )
        full = PaginationPage(
            items=[1, 2],
            page=1,
            total_pages=1,
            total_items=2,
            per_page=10,
        )

        assert repr(truncated) == "PaginationPage(page=1/2, items=5, per_page=5)"
        assert repr(full) == "PaginationPage(page=1/1, items=[1, 2], per_page=10)"


class TestPaginationModuleDoctests:
    """The __repr__ docstring examples must match real behaviour."""

    def test_docstring_examples_pass(self):
        results = doctest.testmod(pagination, verbose=False)

        assert results.failed == 0, (
            f"{results.failed} of {results.attempted} doctests in "
            "src/utils/pagination.py failed"
        )

    def test_doctests_are_actually_present(self):
        """Guard the guard: testmod passes vacuously with no examples."""
        results = doctest.testmod(pagination, verbose=False)

        assert results.attempted > 0, "expected doctest examples in pagination.py"


class TestPaginationPageRepr:
    """Test suite for custom __repr__ implementation."""

    def test_repr_with_small_items_list(self):
        """Verify __repr__ shows full list when 3 or fewer items."""
        page = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )

        repr_str = repr(page)
        assert "page=1/2" in repr_str
        assert "items=[1, 2, 3]" in repr_str

    def test_repr_with_large_items_list(self):
        """Verify __repr__ truncates to count when more than 3 items."""
        page = PaginationPage(
            items=[1, 2, 3, 4, 5, 6, 7, 8],
            page=1,
            total_pages=2,
            total_items=15,
            per_page=8,
        )

        repr_str = repr(page)
        assert "page=1/2" in repr_str
        assert "items=8" in repr_str
        # Should NOT contain the full list
        assert "[1, 2, 3, 4, 5, 6, 7, 8]" not in repr_str

    def test_repr_with_empty_items_list(self):
        """Verify __repr__ handles empty items list correctly."""
        page = PaginationPage(
            items=[],
            page=1,
            total_pages=1,
            total_items=0,
            per_page=10,
        )

        repr_str = repr(page)
        assert "page=1/1" in repr_str
        assert "items=[]" in repr_str

    def test_repr_with_exactly_three_items(self):
        """Verify __repr__ shows full list at exactly 3 items (boundary)."""
        page = PaginationPage(
            items=["a", "b", "c"],
            page=2,
            total_pages=3,
            total_items=9,
            per_page=3,
        )

        repr_str = repr(page)
        assert "items=['a', 'b', 'c']" in repr_str

    def test_repr_with_four_items(self):
        """Verify __repr__ truncates at exactly 4 items (boundary)."""
        page = PaginationPage(
            items=["a", "b", "c", "d"],
            page=1,
            total_pages=1,
            total_items=4,
            per_page=10,
        )

        repr_str = repr(page)
        assert "items=4" in repr_str
        assert "['a', 'b', 'c', 'd']" not in repr_str


class TestPaginationPageEq:
    """Test suite for __eq__ implementation."""

    def test_equal_pages_are_equal(self):
        """Verify two identical pages are equal."""
        page1 = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )
        page2 = PaginationPage(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )

        assert page1 == page2

    def test_different_items_not_equal(self):
        """Verify pages with different items are not equal."""
        page1 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page2 = PaginationPage(
            items=[4, 5, 6], page=1, total_pages=2, total_items=10, per_page=5
        )

        assert page1 != page2

    def test_different_page_number_not_equal(self):
        """Verify pages with different page numbers are not equal."""
        page1 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page2 = PaginationPage(
            items=[1, 2, 3], page=2, total_pages=2, total_items=10, per_page=5
        )

        assert page1 != page2

    def test_different_total_pages_not_equal(self):
        """Verify pages with different total_pages are not equal."""
        page1 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page2 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=3, total_items=10, per_page=5
        )

        assert page1 != page2

    def test_different_total_items_not_equal(self):
        """Verify pages with different total_items are not equal."""
        page1 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page2 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=15, per_page=5
        )

        assert page1 != page2

    def test_different_per_page_not_equal(self):
        """Verify pages with different per_page are not equal."""
        page1 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page2 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=10
        )

        assert page1 != page2

    def test_not_equal_to_non_pagination_page(self):
        """Verify page is not equal to non-PaginationPage objects."""
        page = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )

        assert page != "not a page"
        assert page != 123
        assert page != {"items": [1, 2, 3]}
        # Deliberate `!=` rather than `is not`: this exercises __eq__'s
        # non-PaginationPage branch, which `is not None` would bypass.
        assert page != None  # noqa: E711

    def test_equal_pages_have_same_hash(self):
        """Verify equal pages have the same hash (for use in sets/dicts)."""
        page1 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page2 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )

        assert hash(page1) == hash(page2)

    def test_pages_can_be_used_in_set(self):
        """Verify pages can be added to sets (requires __hash__)."""
        page1 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page2 = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=2, total_items=10, per_page=5
        )
        page3 = PaginationPage(
            items=[4, 5, 6], page=1, total_pages=2, total_items=10, per_page=5
        )

        page_set = {page1, page2, page3}
        assert len(page_set) == 2  # page1 and page2 are duplicates


class TestPaginationPageHash:
    """Tests for PaginationPage.__hash__ (Issue #3221).

    ``@dataclass(frozen=True)`` generates a hash over every field, and
    ``items`` is a ``list``, so an explicit __hash__ tuples the items before
    feeding them to ``hash()``. These tests pin down that contract:

    * equal pages hash identically,
    * pages carrying unhashable items (dicts — the shape ``warning_list``
      builds) raise ``TypeError`` instead of silently corrupting a set,
    * equality itself keeps working for such pages, because __eq__ compares
      the lists directly.
    """

    def make_page(self, items=None, **overrides):
        """Build a page with sensible defaults and optional overrides."""
        kwargs = dict(
            items=[1, 2, 3] if items is None else items,
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )
        kwargs.update(overrides)
        return PaginationPage(**kwargs)

    def test_equal_pages_hash_identically_across_construction_paths(self):
        """Pages equal via __eq__ must hash equally however they were built."""
        constructed = self.make_page(
            items=[0, 1, 2],
            total_items=3,
            total_pages=1,
            start_index=1,
            end_index=3,
        )
        created = PaginationPage.create(
            items=[0, 1, 2],
            page=1,
            per_page=5,
            total_items=3,
        )
        paginated = paginate_items([0, 1, 2], page=1, page_size=5)

        # The hash contract covers every compared field — including the
        # start_index / end_index defaults that differ between paths above.
        assert constructed == created
        assert hash(constructed) == hash(created)
        assert constructed == paginated
        assert hash(constructed) == hash(paginated)

    def test_hash_is_stable_across_repeated_calls(self):
        """The same instance must not change its hash within a process."""
        page = self.make_page()

        first = hash(page)
        assert hash(page) == first
        assert hash(page) == first

    def test_pages_with_tuple_items_are_usable_as_dict_keys(self):
        """Hashable item payloads keep the frozen-dataclass promise."""
        keyed = {self.make_page(items=(1, 2)): "value"}

        assert keyed[self.make_page(items=(1, 2))] == "value"

    def test_unhashable_items_raise_type_error(self):
        """A page of dicts is unhashable — the ordinary Python contract."""
        page = self.make_page(items=[{"id": 1}, {"id": 2}])

        with pytest.raises(TypeError):
            hash(page)

    def test_unhashable_items_error_message_names_the_offender(self):
        """The raised error should explain what could not be hashed."""
        page = self.make_page(items=[{"id": 1}])

        with pytest.raises(TypeError, match="unhashable type"):
            hash(page)

    def test_dict_item_pages_still_compare_equal(self):
        """__eq__ compares the raw lists, so dict-backed pages stay equal."""
        items = [{"id": 1}, {"id": 2}]
        page1 = self.make_page(items=list(items))
        page2 = self.make_page(items=list(items))

        assert page1 == page2

    def test_dict_item_pages_are_rejected_from_sets(self):
        """A set membership attempt surfaces the same TypeError."""
        page = self.make_page(items=[{"id": 1}])

        with pytest.raises(TypeError, match="unhashable type"):
            {page}

    def test_unequal_pages_are_valid_set_members_together(self):
        """Distinct pages may share a set; equal ones collapse to one."""
        page_a = self.make_page(items=[1])
        page_b = self.make_page(items=[2])

        assert len({page_a, page_b}) == 2


class TestPaginationPageFactory:
    """Test suite for PaginationPage.create() factory method."""

    def test_create_calculates_total_pages_correctly(self):
        """Verify create() calculates total_pages correctly."""
        page = PaginationPage.create(
            items=[1, 2, 3],
            page=1,
            per_page=10,
            total_items=25,
        )

        assert page.total_pages == 3  # 25 items / 10 per page = 3 pages

    def test_create_handles_exact_division(self):
        """Verify create() handles exact division (no remainder)."""
        page = PaginationPage.create(
            items=[1, 2, 3, 4, 5],
            page=1,
            per_page=5,
            total_items=20,
        )

        assert page.total_pages == 4  # 20 items / 5 per page = 4 pages

    def test_create_handles_empty_results(self):
        """Verify create() returns at least 1 page even with 0 items."""
        page = PaginationPage.create(
            items=[],
            page=1,
            per_page=10,
            total_items=0,
        )

        assert page.total_pages == 1
        assert page.total_items == 0

    def test_create_raises_on_invalid_page(self):
        """Verify create() raises ValueError for page < 1."""
        with pytest.raises(ValueError, match="page must be >= 1"):
            PaginationPage.create(items=[], page=0, per_page=10, total_items=0)

    def test_create_raises_on_invalid_per_page(self):
        """Verify create() raises ValueError for per_page < 1."""
        with pytest.raises(ValueError, match="per_page must be >= 1"):
            PaginationPage.create(items=[], page=1, per_page=0, total_items=0)


class TestPaginationPageNavigation:
    """Test suite for navigation helper methods."""

    def test_has_next_true_when_not_last_page(self):
        """Verify has_next() returns True when not on last page."""
        page = PaginationPage(
            items=[1], page=1, total_pages=3, total_items=10, per_page=5
        )
        assert page.has_next() is True

    def test_has_next_false_on_last_page(self):
        """Verify has_next() returns False on last page."""
        page = PaginationPage(
            items=[1], page=3, total_pages=3, total_items=10, per_page=5
        )
        assert page.has_next() is False

    def test_has_previous_true_when_not_first_page(self):
        """Verify has_previous() returns True when not on first page."""
        page = PaginationPage(
            items=[1], page=2, total_pages=3, total_items=10, per_page=5
        )
        assert page.has_previous() is True

    def test_has_previous_false_on_first_page(self):
        """Verify has_previous() returns False on first page."""
        page = PaginationPage(
            items=[1], page=1, total_pages=3, total_items=10, per_page=5
        )
        assert page.has_previous() is False

    def test_next_page_returns_correct_number(self):
        """Verify next_page() returns page + 1 when available."""
        page = PaginationPage(
            items=[1], page=2, total_pages=5, total_items=10, per_page=5
        )
        assert page.next_page() == 3

    def test_next_page_returns_none_on_last_page(self):
        """Verify next_page() returns None on last page."""
        page = PaginationPage(
            items=[1], page=5, total_pages=5, total_items=10, per_page=5
        )
        assert page.next_page() is None

    def test_previous_page_returns_correct_number(self):
        """Verify previous_page() returns page - 1 when available."""
        page = PaginationPage(
            items=[1], page=3, total_pages=5, total_items=10, per_page=5
        )
        assert page.previous_page() == 2

    def test_previous_page_returns_none_on_first_page(self):
        """Verify previous_page() returns None on first page."""
        page = PaginationPage(
            items=[1], page=1, total_pages=5, total_items=10, per_page=5
        )
        assert page.previous_page() is None


class TestPaginationPageSerialization:
    """Test suite for to_dict() serialization."""

    def test_to_dict_contains_all_fields(self):
        """Verify to_dict() includes all required fields."""
        page = PaginationPage(
            items=[1, 2, 3], page=2, total_pages=5, total_items=20, per_page=5
        )
        result = page.to_dict()

        assert "items" in result
        assert "page" in result
        assert "total_pages" in result
        assert "total_items" in result
        assert "per_page" in result
        assert "has_next" in result
        assert "has_previous" in result
        assert "next_page" in result
        assert "previous_page" in result

    def test_to_dict_values_are_correct(self):
        """Verify to_dict() returns correct values."""
        page = PaginationPage(
            items=[1, 2, 3], page=2, total_pages=5, total_items=20, per_page=5
        )
        result = page.to_dict()

        assert result["items"] == [1, 2, 3]
        assert result["page"] == 2
        assert result["total_pages"] == 5
        assert result["total_items"] == 20
        assert result["per_page"] == 5
        assert result["has_next"] is True
        assert result["has_previous"] is True
        assert result["next_page"] == 3
        assert result["previous_page"] == 1


# --- NEW TESTS ADDED FOR ISSUE #2030 ---


class TestCoerceInteger:
    """Test suite for _coerce_integer helper function."""

    def test_coerce_integer_valid_strings(self):
        """Verify valid number strings are coerced to int."""
        assert _coerce_integer("10", 1) == 10

    def test_coerce_integer_invalid_string(self):
        """Verify invalid strings return the fallback/default."""
        assert _coerce_integer("abc", 1) == 1

    def test_coerce_integer_none(self):
        """Verify None returns the fallback/default."""
        assert _coerce_integer(None, 1) == 1

    def test_coerce_integer_float(self):
        """Verify floats are coerced/truncated to int."""
        assert _coerce_integer(3.14, 1) == 3
        assert _coerce_integer(-2.9, 1) == -2


class TestPaginateItemsBoundaryConditions:
    """Test suite for paginate_items boundary conditions (Issue #2030).

    These assertions used to compare the return value against a bare list.
    ``paginate_items`` returns a ``PaginationPage`` — the page geometry is
    the point of the helper — so they now read ``.items`` and additionally
    pin the clamped page number, which is what Issue #3045 regressed.
    """

    def test_empty_list(self):
        """Verify empty list returns empty list regardless of pagination."""
        page = paginate_items([], page=1, page_size=10)

        assert page.items == []
        assert page.total_pages == 1
        assert page.total_items == 0
        assert page.start_index == 0
        assert page.end_index == 0

    def test_page_zero(self):
        """Verify page=0 is clamped to 1."""
        items = [1, 2, 3, 4, 5]
        page = paginate_items(items, page=0, page_size=2)

        assert page.items == [1, 2]
        assert page.page == 1

    def test_page_negative(self):
        """Verify page=-1 is clamped to 1."""
        items = [1, 2, 3, 4, 5]
        page = paginate_items(items, page=-1, page_size=2)

        assert page.items == [1, 2]
        assert page.page == 1

    def test_page_beyond_range(self):
        """Verify page=9999 is clamped to the last available page."""
        items = [1, 2, 3, 4, 5]
        # 5 items total, page_size=2 means 3 pages. The last page contains just [5].
        page = paginate_items(items, page=9999, page_size=2)

        assert page.items == [5]
        assert page.page == 3
        assert page.total_pages == 3

    def test_page_size_zero(self):
        """Verify page_size=0 is clamped to a minimum valid size (1)."""
        items = [1, 2, 3]
        page = paginate_items(items, page=1, page_size=0)

        assert page.items == [1]
        assert page.per_page == 1

    def test_page_size_negative(self):
        """Verify page_size=-5 is clamped to a minimum valid size (1)."""
        items = [1, 2, 3]
        page = paginate_items(items, page=1, page_size=-5)

        assert page.items == [1]
        assert page.per_page == 1

    def test_page_string(self):
        """Verify string inputs for page are handled and coerced gracefully."""
        items = [1, 2, 3, 4, 5]
        # "abc" coercion fails, defaults to 1
        assert paginate_items(items, page="abc", page_size=2).items == [1, 2]
        # A numeric string is honoured rather than discarded.
        assert paginate_items(items, page="3", page_size=2).items == [5]


class TestPaginateItemsKeywordContract:
    """Regression tests for the caller-facing signature (Issue #3045).

    ``src/utils/warning_list.py`` calls
    ``paginate_items(rows, page=..., page_size=..., max_page_size=100)``.
    The helper was rewritten with a positional ``(items, page_size,
    current_page)`` signature, so every one of those calls raised
    ``TypeError: paginate_items() got an unexpected keyword argument 'page'``
    and the plagiarism warnings list could not render at all.
    """

    def test_accepts_the_keyword_arguments_its_caller_uses(self):
        page = paginate_items(
            list(range(150)),
            page=2,
            page_size=200,
            max_page_size=100,
        )

        assert page.page == 2
        assert page.per_page == 100

    def test_max_page_size_clamps_the_requested_size(self):
        page = paginate_items(
            list(range(150)), page=1, page_size=200, max_page_size=100
        )

        assert len(page.items) == 100
        assert page.page_size == 100
        assert page.total_pages == 2

    def test_max_page_size_none_lifts_the_cap(self):
        page = paginate_items(
            list(range(150)), page=1, page_size=200, max_page_size=None
        )

        assert len(page.items) == 150
        assert page.total_pages == 1

    def test_page_size_alias_matches_per_page(self):
        page = paginate_items(list(range(10)), page=1, page_size=4)

        assert page.page_size == page.per_page == 4

    def test_start_and_end_index_are_one_based_and_inclusive(self):
        page = paginate_items(list(range(23)), page=2, page_size=10)

        assert page.start_index == 11
        assert page.end_index == 20

    def test_start_and_end_index_are_zero_for_an_empty_sequence(self):
        page = paginate_items([], page=1, page_size=10)

        assert (page.start_index, page.end_index) == (0, 0)

    def test_last_partial_page_reports_a_short_end_index(self):
        page = paginate_items(list(range(23)), page=3, page_size=10)

        assert len(page.items) == 3
        assert page.start_index == 21
        assert page.end_index == 23

    def test_source_sequence_is_not_mutated(self):
        items = [1, 2, 3, 4, 5]
        paginate_items(items, page=1, page_size=2)

        assert items == [1, 2, 3, 4, 5]

    def test_accepts_a_tuple_without_choking_on_the_slice(self):
        page = paginate_items((1, 2, 3, 4, 5), page=2, page_size=2)

        assert page.items == [3, 4]

    def test_non_numeric_page_size_falls_back_to_the_default(self):
        page = paginate_items(list(range(50)), page=1, page_size=None)

        assert page.per_page == 10


class TestPaginationPageIndexFields:
    """``start_index`` / ``end_index`` round-trip through the dataclass."""

    def test_fields_default_to_zero_for_direct_construction(self):
        page = PaginationPage(
            items=[1, 2, 3], page=1, total_pages=1, total_items=3, per_page=10
        )

        assert page.start_index == 0
        assert page.end_index == 0

    def test_create_populates_the_index_fields(self):
        page = PaginationPage.create(
            items=[11, 12, 13], page=2, per_page=10, total_items=23
        )

        assert page.start_index == 11
        assert page.end_index == 13

    def test_to_dict_exposes_the_index_fields(self):
        page = paginate_items(list(range(23)), page=2, page_size=10)
        result = page.to_dict()

        assert result["start_index"] == 11
        assert result["end_index"] == 20

    def test_pages_differing_only_by_index_fields_are_not_equal(self):
        first = PaginationPage(
            items=[1],
            page=1,
            total_pages=1,
            total_items=1,
            per_page=10,
            start_index=1,
            end_index=1,
        )
        second = PaginationPage(
            items=[1],
            page=1,
            total_pages=1,
            total_items=1,
            per_page=10,
            start_index=0,
            end_index=0,
        )

        assert first != second


class TestPaginationPageWasClamped:
    """``was_clamped`` distinguishes clamped requests from genuine ones (#3218).

    ``paginate_items`` deliberately never raises for an out-of-range page,
    which left API callers unable to tell "the user asked for the last page"
    from "an out-of-range request was silently pulled back". The flag makes
    that distinction visible on the returned page.
    """

    def make_page(self, **overrides):
        """Build a page with sensible defaults and optional overrides."""
        kwargs = dict(
            items=[1, 2, 3],
            page=1,
            total_pages=2,
            total_items=10,
            per_page=5,
        )
        kwargs.update(overrides)
        return PaginationPage(**kwargs)

    def test_page_beyond_the_end_is_flagged(self):
        page = paginate_items([1, 2, 3, 4, 5], page=9999, page_size=2)

        assert page.page == 3
        assert page.was_clamped is True

    def test_page_zero_is_flagged(self):
        page = paginate_items([1, 2, 3, 4, 5], page=0, page_size=2)

        assert page.page == 1
        assert page.was_clamped is True

    def test_negative_page_is_flagged(self):
        page = paginate_items([1, 2, 3, 4, 5], page=-7, page_size=2)

        assert page.page == 1
        assert page.was_clamped is True

    def test_in_range_request_is_not_flagged(self):
        page = paginate_items(list(range(23)), page=2, page_size=10)

        assert page.page == 2
        assert page.was_clamped is False

    def test_last_page_requested_exactly_is_not_flagged(self):
        page = paginate_items([1, 2, 3, 4, 5], page=3, page_size=2)

        assert page.page == 3
        assert page.was_clamped is False

    def test_default_page_request_is_not_flagged(self):
        page = paginate_items(list(range(50)))

        assert page.page == 1
        assert page.was_clamped is False

    def test_non_numeric_page_coerces_without_a_clamp_flag(self):
        # "abc" falls back to the default (a coercion), not a clamp.
        page = paginate_items(list(range(23)), page="abc", page_size=10)

        assert page.page == 1
        assert page.was_clamped is False

    def test_empty_sequence_with_out_of_range_page_is_flagged(self):
        page = paginate_items([], page=5, page_size=10)

        assert page.items == []
        assert page.total_pages == 1
        assert page.was_clamped is True

    def test_field_defaults_to_false_for_direct_construction(self):
        page = self.make_page()

        assert page.was_clamped is False

    def test_to_dict_exposes_the_flag(self):
        clamped = paginate_items([], page=5, page_size=10).to_dict()
        exact = paginate_items([], page=1, page_size=10).to_dict()

        assert clamped["was_clamped"] is True
        assert exact["was_clamped"] is False

    def test_pages_differing_only_by_the_flag_are_unequal(self):
        clamped = self.make_page(was_clamped=True)
        plain = self.make_page()

        assert clamped != plain

    def test_hash_covers_the_flag_consistently_with_equality(self):
        clamped = self.make_page(was_clamped=True)
        plain = self.make_page()
        same_clamp = self.make_page(was_clamped=True)

        assert hash(clamped) != hash(plain)
        assert hash(clamped) == hash(same_clamp)
