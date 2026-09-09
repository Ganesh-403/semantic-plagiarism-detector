"""Sequence and streaming pagination preserve order, boundaries and laziness."""
import pytest
from src.utils.paginationn import (
    Page, PageInfo, PaginatedIterator, PaginationError, Paginator, SequenceLike,
    batch_process, create_paginated_response, paginate_items, stream_paginate,
)


@pytest.mark.parametrize('count', [0, 1, 3, 6, 7])
@pytest.mark.parametrize('factory', [list, iter])
def test_pages_partition_input_without_losing_or_duplicating_items(count, factory):
    values = list(range(count))
    paginator = Paginator(factory(values), page_size=3)
    pages = list(paginator.iter_pages())
    assert [value for page in pages for value in page] == values
    for i, page in enumerate(pages):
        assert page.items == values[i * 3:(i + 1) * 3]
        assert page.has_more == (i < len(pages) - 1)
        assert page.page_info.has_previous == (i > 0)
        assert page.page_info.start_index == i * 3
        assert page.page_info.end_index == i * 3 + len(page) - 1
    assert paginator.get_page(1).items == values[:3]
    assert not paginator.get_page(count + 2).items


def test_generator_random_access_only_consumes_needed_prefix():
    consumed = []
    def rows():
        for value in range(7):
            consumed.append(value)
            yield value
    paginator = Paginator(rows(), page_size=3)
    assert paginator.get_page(2).items == [3, 4, 5]
    assert consumed == list(range(7))
    assert paginator.get_page(1).items == [0, 1, 2]
    assert consumed == list(range(7))
    final = paginator.get_page(3)
    assert final.items == [6] and not final.has_more
    assert final.page_info.total_items == 7 and final.page_info.total_pages == 3


def test_page_metadata_and_legacy_response():
    values, info = paginate_items(range(7), page=2, page_size=3)
    assert values == [3, 4, 5] and info.has_next
    response = create_paginated_response(iter(range(7)), page=2, page_size=3, total_hint=7)
    assert response['data'] == values
    assert response['pagination'] == {'page': 2, 'page_size': 3, 'total_items': 7, 'total_pages': 3, 'has_next': True, 'has_previous': True, 'start_index': 3, 'end_index': 5}
    page = Page(values, PageInfo(2, 3), True)
    assert len(page) == 3 and list(page) == values
    assert 'page=2' in repr(page)


@pytest.mark.parametrize('invalid', [0, -1, 'bad'])
def test_invalid_page_has_explicit_strict_and_lenient_outcomes(invalid):
    with pytest.raises(PaginationError):
        Paginator([1]).get_page(invalid)
    assert Paginator([1], error_on_invalid_page=False).get_page(invalid).items == []


def test_small_page_size_is_clamped_and_non_integer_is_rejected():
    assert Paginator([1, 2], page_size=0).get_page(1).items == [1]
    with pytest.raises(PaginationError):
        Paginator([1], page_size='bad')


def test_streaming_interfaces_do_not_read_ahead_or_process_empty_batches():
    consumed = []
    def rows():
        for i in range(7):
            consumed.append(i)
            yield i
    pages = stream_paginate(rows(), page_size=3)
    assert next(pages) == [0, 1, 2]
    assert consumed == [0, 1, 2]
    assert list(pages) == [[3, 4, 5], [6]]
    assert list(stream_paginate([1, 2, 3], 2)) == [[1, 2], [3]]
    assert list(batch_process(iter(range(7)), 3, sum)) == [3, 12, 6]
    assert list(batch_process(range(4), 3)) == [[0, 1, 2], [3]]
    assert list(batch_process([], processor=lambda _: pytest.fail('Empty batch was processed'))) == []


def test_paginated_iterator_supports_forward_pages_and_repeated_current_page():
    paginator = PaginatedIterator(iter(range(7)), page_size=3)
    assert paginator.get_page(0) == []
    assert paginator.get_page(2) == [3, 4, 5]
    assert paginator.get_page(2) == [3, 4, 5]
    with pytest.raises(PaginationError, match='rewind'):
        paginator.get_page(1)
    assert paginator.get_page(3) == [6]
    assert paginator.get_page(4) == []
    assert list(PaginatedIterator(iter(range(7)), 3)) == list(range(7))


def test_sequence_wrapper_retains_unread_items_after_indexing():
    sequence = SequenceLike(iter(range(5)))
    assert sequence[1] == 1
    assert sequence[3] == 3
    assert list(sequence) == [0, 1, 2, 3, 4]
    assert len(sequence) == 5
    assert sequence[-1] == 4 and sequence[1:4] == [1, 2, 3]
    assert sequence[::-1] == [4, 3, 2, 1, 0]
    with pytest.raises(IndexError):
        _ = sequence[5]
    hinted = SequenceLike(iter(range(5)), total=5)
    assert len(hinted) == 5 and list(hinted) == list(range(5))
    assert SequenceLike(iter(range(5)))[1:3] == [1, 2]
    assert SequenceLike(iter(range(5)))[:] == list(range(5))


def test_iterator_failure_is_not_silently_reported_as_end_of_results():
    def broken():
        yield 1
        raise RuntimeError('cursor disconnected')
    with pytest.raises(PaginationError, match='cursor disconnected'):
        list(Paginator(broken(), page_size=2).iter_pages())
