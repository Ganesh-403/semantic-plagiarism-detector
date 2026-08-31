"""
Comprehensive Unit Tests for PaginationPage Properties (Issue #3216)
Tests the newly added @property decorators for next_page and prev_page.
"""

import pytest
from src.utils.pagination import PaginationPage, paginate_items

# ==============================================================================
# SECTION 1: Testing the next_page Property
# ==============================================================================


class TestNextPageProperty:
    def test_next_page_exists(self):
        """Ensure the next_page property exists on the object."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert hasattr(page, "next_page")

    def test_next_page_is_property_not_method(self):
        """It should be accessible without parentheses (e.g., page.next_page, not page.next_page())."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert not callable(page.next_page)

    def test_next_page_returns_correct_integer(self):
        """On page 1 with total_pages=3, next_page should be 2."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.next_page == 2

    def test_next_page_returns_none_at_end(self):
        """On the last page, next_page should be None."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=3, per_page=10, total_items=25
        )
        assert page.next_page is None

    def test_next_page_return_type(self):
        """Ensure the return value is an integer."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert isinstance(page.next_page, int)

    def test_next_page_none_type_at_end(self):
        """Ensure the return value is None at the end."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=3, per_page=10, total_items=25
        )
        assert page.next_page is None

    def test_next_page_with_total_pages_one(self):
        """If there is only 1 page, next_page should be None."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=3
        )
        assert page.next_page is None

    def test_next_page_consistent_with_has_next(self):
        """next_page should only exist if has_next() is True."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        if page.has_next():
            assert page.next_page is not None
        else:
            assert page.next_page is None


# ==============================================================================
# SECTION 2: Testing the prev_page Property
# ==============================================================================


class TestPrevPageProperty:
    def test_prev_page_exists(self):
        """Ensure the prev_page property exists on the object."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert hasattr(page, "prev_page")

    def test_prev_page_is_property_not_method(self):
        """It should be accessible without parentheses (e.g., page.prev_page, not page.prev_page())."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert not callable(page.prev_page)

    def test_prev_page_returns_correct_integer(self):
        """On page 2, prev_page should be 1."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert page.prev_page == 1

    def test_prev_page_returns_none_at_start(self):
        """On the first page, prev_page should be None."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.prev_page is None

    def test_prev_page_return_type(self):
        """Ensure the return value is an integer."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert isinstance(page.prev_page, int)

    def test_prev_page_none_type_at_start(self):
        """Ensure the return value is None at the start."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.prev_page is None

    def test_prev_page_with_total_pages_one(self):
        """If there is only 1 page, prev_page should be None."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=3
        )
        assert page.prev_page is None

    def test_prev_page_consistent_with_has_previous(self):
        """prev_page should only exist if has_previous() is True."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        if page.has_previous():
            assert page.prev_page is not None
        else:
            assert page.prev_page is None


# ==============================================================================
# SECTION 3: Testing Properties with paginate_items
# ==============================================================================


class TestPropertiesWithPagination:
    def test_next_page_with_paginate_items(self):
        """Test next_page using the paginate_items factory."""
        result = paginate_items([1, 2, 3, 4, 5], page=1, page_size=2)
        assert result.next_page == 2

    def test_prev_page_with_paginate_items(self):
        """Test prev_page using the paginate_items factory."""
        result = paginate_items([1, 2, 3, 4, 5], page=2, page_size=2)
        assert result.prev_page == 1

    def test_next_page_with_paginate_items_last_page(self):
        """Test next_page is None on the last page."""
        result = paginate_items([1, 2, 3, 4, 5], page=3, page_size=2)
        assert result.next_page is None

    def test_prev_page_with_paginate_items_first_page(self):
        """Test prev_page is None on the first page."""
        result = paginate_items([1, 2, 3, 4, 5], page=1, page_size=2)
        assert result.prev_page is None

    def test_clamping_page_affects_properties(self):
        """If page is clamped to the last page, next_page should be None."""
        result = paginate_items([1, 2, 3, 4, 5], page=9999, page_size=2)
        assert result.page == 3
        assert result.next_page is None


# ==============================================================================
# SECTION 4: Immutability and Edge Cases
# ==============================================================================


class TestImmutability:
    def test_properties_do_not_mutate_object(self):
        """Accessing properties should not change the object state."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        initial_page = page.page
        initial_total_pages = page.total_pages

        _ = page.next_page
        _ = page.prev_page

        assert page.page == initial_page
        assert page.total_pages == initial_total_pages

    def test_properties_work_with_empty_items(self):
        """Properties should work even with empty items."""
        page = PaginationPage.create(items=[], page=1, per_page=10, total_items=0)
        assert page.next_page is None
        assert page.prev_page is None


# ==============================================================================
# SECTION 5: Equality and Hash Tests
# ==============================================================================


class TestPropertiesWithEquality:
    def test_pages_with_same_properties_are_equal(self):
        page1 = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        page2 = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page1 == page2

    def test_pages_with_different_properties_not_equal(self):
        page1 = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        page2 = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert page1 != page2

    def test_property_values_repr(self):
        """Ensure the repr output remains correct when properties are added."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        repr_str = repr(page)
        assert "page=1/3" in repr_str
        assert "items=[1, 2, 3]" in repr_str
