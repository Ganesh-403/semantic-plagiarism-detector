import pytest

from src.utils.pagination import (
    PaginationPage,
    paginate_items,
)


def test_first_page_contains_requested_slice():
    result = paginate_items(
        list(range(25)),
        page=1,
        page_size=10,
    )

    assert isinstance(result, PaginationPage)
    assert result.items == list(range(10))
    assert result.total_items == 25
    assert result.page == 1
    assert result.page_size == 10
    assert result.total_pages == 3
    assert result.start_index == 1
    assert result.end_index == 10


def test_middle_page_has_correct_bounds():
    result = paginate_items(
        list(range(25)),
        page=2,
        page_size=10,
    )

    assert result.items == list(range(10, 20))
    assert result.page == 2
    assert result.start_index == 11
    assert result.end_index == 20


def test_final_partial_page_has_correct_bounds():
    result = paginate_items(
        list(range(25)),
        page=3,
        page_size=10,
    )

    assert result.items == list(range(20, 25))
    assert result.start_index == 21
    assert result.end_index == 25


def test_page_below_one_is_clamped_to_first_page():
    result = paginate_items(
        ["a", "b", "c"],
        page=-50,
        page_size=2,
    )

    assert result.page == 1
    assert result.items == ["a", "b"]


def test_page_above_range_is_clamped_to_last_page():
    result = paginate_items(
        list(range(12)),
        page=999,
        page_size=5,
    )

    assert result.page == 3
    assert result.items == [10, 11]
    assert result.start_index == 11
    assert result.end_index == 12


def test_empty_input_returns_stable_first_page():
    result = paginate_items(
        [],
        page=99,
        page_size=10,
    )

    assert result.items == []
    assert result.total_items == 0
    assert result.page == 1
    assert result.page_size == 10
    assert result.total_pages == 1
    assert result.start_index == 0
    assert result.end_index == 0


def test_page_size_below_one_is_clamped_to_one():
    result = paginate_items(
        [1, 2, 3],
        page_size=0,
    )

    assert result.page_size == 1
    assert result.total_pages == 3
    assert result.items == [1]


def test_page_size_is_capped_by_default_limit():
    result = paginate_items(
        list(range(150)),
        page_size=500,
    )

    assert result.page_size == 100
    assert len(result.items) == 100
    assert result.total_pages == 2


def test_custom_page_size_limit_is_applied():
    result = paginate_items(
        list(range(20)),
        page_size=10,
        max_page_size=4,
    )

    assert result.page_size == 4
    assert result.items == [0, 1, 2, 3]
    assert result.total_pages == 5


def test_none_disables_page_size_limit():
    result = paginate_items(
        list(range(150)),
        page_size=125,
        max_page_size=None,
    )

    assert result.page_size == 125
    assert len(result.items) == 125
    assert result.total_pages == 2


@pytest.mark.parametrize(
    ("page", "expected_page"),
    [
        (None, 1),
        ("invalid", 1),
        (float("inf"), 1),
        ("2", 2),
    ],
)
def test_page_values_are_safely_coerced(
    page,
    expected_page,
):
    result = paginate_items(
        list(range(30)),
        page=page,
        page_size=10,
    )

    assert result.page == expected_page


@pytest.mark.parametrize(
    ("page_size", "expected_size"),
    [
        (None, 10),
        ("invalid", 10),
        (float("inf"), 10),
        ("4", 4),
    ],
)
def test_page_size_values_are_safely_coerced(
    page_size,
    expected_size,
):
    result = paginate_items(
        list(range(30)),
        page_size=page_size,
    )

    assert result.page_size == expected_size


def test_invalid_max_page_size_is_rejected():
    with pytest.raises(
        ValueError,
        match="max_page_size must be at least 1",
    ):
        paginate_items(
            [1, 2, 3],
            max_page_size=0,
        )


def test_returned_items_are_a_new_list():
    source = ["a", "b", "c"]

    result = paginate_items(
        source,
        page_size=2,
    )
    result.items.append("changed")

    assert source == ["a", "b", "c"]


def test_preserves_generic_item_objects():
    first = {"id": 1}
    second = {"id": 2}

    result = paginate_items(
        (first, second),
        page_size=1,
    )

    assert result.items == [first]
    assert result.items[0] is first
