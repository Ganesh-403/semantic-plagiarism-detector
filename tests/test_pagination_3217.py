"""
Comprehensive Tests for to_dict() Helper on PaginationPage (Issue #3217)
Tests serialization, all required keys, edge cases, and JSON compatibility.
"""

import pytest
import json
from src.utils.pagination import PaginationPage, paginate_items

# ==============================================================================
# SECTION 1: Basic Functionality and Return Type
# ==============================================================================


class TestToDictBasic:
    def test_to_dict_returns_dict(self):
        """The method must return a Python dictionary."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert isinstance(page.to_dict(), dict)

    def test_to_dict_is_serializable(self):
        """The dictionary must be serializable to JSON (for API endpoints)."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        json.dumps(page.to_dict())

    def test_to_dict_is_safe_to_call_multiple_times(self):
        """Calling the method multiple times should not mutate the object."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        dict1 = page.to_dict()
        dict2 = page.to_dict()
        assert dict1 == dict2
        assert page.page == 1


# ==============================================================================
# SECTION 2: Testing All Required Keys
# ==============================================================================


class TestToDictRequiredKeys:
    def test_to_dict_has_items(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "items" in page.to_dict()

    def test_to_dict_has_page(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "page" in page.to_dict()

    def test_to_dict_has_total_pages(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "total_pages" in page.to_dict()

    def test_to_dict_has_total_items(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "total_items" in page.to_dict()

    def test_to_dict_has_per_page(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "per_page" in page.to_dict()

    def test_to_dict_has_next_page(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "next_page" in page.to_dict()

    def test_to_dict_has_previous_page(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert "previous_page" in page.to_dict()

    def test_to_dict_has_start_index(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "start_index" in page.to_dict()

    def test_to_dict_has_end_index(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert "end_index" in page.to_dict()


# ==============================================================================
# SECTION 3: Testing Correct Values
# ==============================================================================


class TestToDictValues:
    def test_to_dict_page_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert page.to_dict()["page"] == 2

    def test_to_dict_per_page_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.to_dict()["per_page"] == 10

    def test_to_dict_total_items_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.to_dict()["total_items"] == 25

    def test_to_dict_total_pages_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.to_dict()["total_pages"] == 3

    def test_to_dict_items_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.to_dict()["items"] == [1, 2, 3]

    def test_to_dict_next_page_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.to_dict()["next_page"] == 2

    def test_to_dict_previous_page_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert page.to_dict()["previous_page"] == 1

    def test_to_dict_start_index_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert page.to_dict()["start_index"] == 11

    def test_to_dict_end_index_value(self):
        page = PaginationPage.create(
            items=[1, 2, 3], page=2, per_page=10, total_items=25
        )
        assert page.to_dict()["end_index"] == 13


# ==============================================================================
# SECTION 4: Testing Edge Cases and Boundaries
# ==============================================================================


class TestToDictEdgeCases:
    def test_to_dict_at_last_page(self):
        """At the last page, next_page should be None."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=3, per_page=10, total_items=25
        )
        assert page.to_dict()["next_page"] is None

    def test_to_dict_at_first_page(self):
        """At the first page, previous_page should be None."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        assert page.to_dict()["previous_page"] is None

    def test_to_dict_empty_items(self):
        """Items should be an empty list when there are no items."""
        page = PaginationPage.create(items=[], page=1, per_page=10, total_items=0)
        assert page.to_dict()["items"] == []

    def test_to_dict_zero_total_items(self):
        """Total items should be 0 when empty, but total_pages should be 1."""
        page = PaginationPage.create(items=[], page=1, per_page=10, total_items=0)
        assert page.to_dict()["total_items"] == 0
        assert page.to_dict()["total_pages"] == 1


# ==============================================================================
# SECTION 5: Integration with paginate_items
# ==============================================================================


class TestToDictWithPaginateItems:
    def test_to_dict_with_paginate_items_page_1(self):
        """Test to_dict with paginate_items on page 1."""
        result = paginate_items([1, 2, 3, 4, 5], page=1, page_size=2)
        d = result.to_dict()
        assert d["page"] == 1
        assert d["total_pages"] == 3
        assert d["next_page"] == 2
        assert d["previous_page"] is None

    def test_to_dict_with_paginate_items_page_2(self):
        """Test to_dict with paginate_items on page 2."""
        result = paginate_items([1, 2, 3, 4, 5], page=2, page_size=2)
        d = result.to_dict()
        assert d["page"] == 2
        assert d["items"] == [3, 4]
        assert d["next_page"] == 3
        assert d["previous_page"] == 1

    def test_to_dict_with_paginate_items_last_page(self):
        """Test to_dict with paginate_items on the last page."""
        result = paginate_items([1, 2, 3, 4, 5], page=3, page_size=2)
        d = result.to_dict()
        assert d["page"] == 3
        assert d["items"] == [5]
        assert d["next_page"] is None
        assert d["previous_page"] == 2

    def test_to_dict_with_clamped_page(self):
        """If page is clamped to the end, next_page should be None."""
        result = paginate_items([1, 2, 3, 4, 5], page=9999, page_size=2)
        d = result.to_dict()
        assert d["page"] == 3
        assert d["next_page"] is None


# ==============================================================================
# SECTION 6: JSON Serialization and Formatting
# ==============================================================================


class TestToDictJSON:
    def test_to_dict_json_dumps_success(self):
        """Must be compatible with json.dumps."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        json_str = json.dumps(page.to_dict())
        assert json_str is not None

    def test_to_dict_json_loads_success(self):
        """Must be decodable back into a Python dict."""
        page = PaginationPage.create(
            items=[1, 2, 3], page=1, per_page=10, total_items=25
        )
        json_str = json.dumps(page.to_dict())
        loaded_dict = json.loads(json_str)
        assert loaded_dict["page"] == 1

    def test_to_dict_json_no_non_serializable(self):
        """The dict must not contain non-serializable types."""
        page = PaginationPage.create(
            items=["a", "b"], page=1, per_page=10, total_items=25
        )
        try:
            json.dumps(page.to_dict())
        except TypeError:
            pytest.fail("to_dict() returned a non-serializable value!")
