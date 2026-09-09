"""Pagination for sequences and streams, preserving every input item.

Paginator retains the consumed prefix to support repeated/backward page access.
For bounded memory, use stream_paginate, batch_process, or PaginatedIterator;
those interfaces consume a stream forward and retain at most one page.
"""
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any, Callable, Generic, TypeVar

T = TypeVar('T')


class PaginationError(ValueError):
    """A page request cannot be served by this paginator."""


@dataclass
class PageInfo:
    page: int
    page_size: int
    total_items: int | None = None
    has_next: bool = False
    has_previous: bool = False
    total_pages: int | None = None
    start_index: int | None = None
    end_index: int | None = None


class Page(Generic[T]):
    def __init__(self, items: list[T], page_info: PageInfo, has_more: bool = False):
        self.items, self.page_info, self.has_more = items, page_info, has_more

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return f'Page(items={len(self.items)}, page={self.page_info.page}, has_more={self.has_more})'


def _page_size(size):
    if not isinstance(size, int):
        raise PaginationError('Page size must be an integer')
    return max(1, size)


class Paginator(Generic[T]):
    """Support arbitrary page access by caching only the consumed stream prefix."""
    def __init__(self, items: Sequence[T] | Iterable[T], page_size: int = 20,
                 total_hint: int | None = None, error_on_invalid_page: bool = True):
        self._items = items
        self.page_size = _page_size(page_size)
        self.error_on_invalid_page = error_on_invalid_page
        self._is_sequence = isinstance(items, Sequence)
        self._stream = iter(items)
        self._cached_items: list[T] = []
        self._exhausted = False
        self._total_items = len(items) if self._is_sequence else total_hint
        if self._total_items is not None:
            self._total_items = max(0, self._total_items)

    def _fill(self, end: int):
        while not self._exhausted and len(self._cached_items) < end:
            try:
                self._cached_items.append(next(self._stream))
            except StopIteration:
                self._exhausted = True
                self._total_items = len(self._cached_items)

    def get_page(self, page: int = 1) -> Page[T]:
        if not isinstance(page, int) or page < 1:
            if self.error_on_invalid_page:
                raise PaginationError('Page must be a positive integer')
            return Page([], PageInfo(page=page, page_size=self.page_size))
        start = (page - 1) * self.page_size
        end = start + self.page_size
        try:
            if self._is_sequence:
                try:
                    items = list(self._items[start:end])
                except (TypeError, AttributeError):
                    self._fill(end + 1)
                    items = self._cached_items[start:end]
                has_more = end < self._total_items
            else:
                # Retain a one-item lookahead instead of dropping a peeked item.
                self._fill(end + 1)
                items = self._cached_items[start:end]
                has_more = len(self._cached_items) > end
        except Exception as exc:
            raise PaginationError(f'Error retrieving page {page}: {exc}') from exc
        total = self._total_items
        info = PageInfo(page=page, page_size=self.page_size, total_items=total,
                        total_pages=None if total is None else (total + self.page_size - 1) // self.page_size,
                        has_next=has_more, has_previous=page > 1, start_index=start,
                        end_index=start + len(items) - 1 if items else None)
        return Page(items, info, has_more)

    def iter_pages(self) -> Iterator[Page[T]]:
        number = 1
        while True:
            page = self.get_page(number)
            if not page.items:
                return
            yield page
            if not page.has_more:
                return
            number += 1

    def paginate(self, page: int = 1) -> tuple[list[T], PageInfo]:
        result = self.get_page(page)
        return result.items, result.page_info


class PaginatedIterator(Generic[T]):
    """Forward-only stream paging with a repeatable current page and bounded memory."""
    def __init__(self, source: Iterator[T], page_size: int = 100, total_hint: int | None = None):
        self.source = iter(source)
        self.page_size = _page_size(page_size)
        self.total_hint = total_hint
        self._current_page: list[T] = []
        self._page_num = 0

    def __iter__(self):
        while True:
            page = self.get_page(self._page_num + 1)
            if not page:
                return
            yield from page

    def get_page(self, page_num: int) -> list[T]:
        if page_num <= 0:
            return []
        if page_num < self._page_num:
            raise PaginationError('A consumed stream cannot rewind; use Paginator for backward access')
        while self._page_num < page_num:
            self._current_page = list(islice(self.source, self.page_size))
            self._page_num += 1
            if not self._current_page:
                break
        return list(self._current_page)


def paginate_items(items, page=1, page_size=20, total_hint=None, error_on_invalid_page=True):
    return Paginator(items, page_size, total_hint, error_on_invalid_page).paginate(page)


def create_paginated_response(items, page=1, page_size=20, total_hint=None):
    data, metadata = paginate_items(items, page, page_size, total_hint)
    return {'data': data, 'pagination': asdict(metadata)}


def stream_paginate(source: Iterable[T], page_size: int = 100) -> Iterator[list[T]]:
    size = _page_size(page_size)
    stream = iter(source)
    while page := list(islice(stream, size)):
        yield page


def batch_process(items: Iterable[T], batch_size: int = 100,
                  processor: Callable[[list[T]], Any] | None = None):
    for page in stream_paginate(items, batch_size):
        yield processor(page) if processor is not None else page


class SequenceLike(Sequence[T]):
    """A lazy, repeatable sequence view over an iterator, retaining consumed items."""
    def __init__(self, iterator: Iterator[T], total: int | None = None):
        self.iterator = iter(iterator)
        self.total = total
        self._items: list[T] = []
        self._exhausted = False

    def _fill(self, stop=None):
        while not self._exhausted and (stop is None or len(self._items) < stop):
            try:
                self._items.append(next(self.iterator))
            except StopIteration:
                self._exhausted = True
                self.total = len(self._items)

    def __iter__(self):
        index = 0
        while True:
            self._fill(index + 1)
            if index >= len(self._items):
                return
            yield self._items[index]
            index += 1

    def __len__(self):
        if self.total is None:
            self._fill()
        return self.total

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.stop is None or index.stop < 0 or (index.start or 0) < 0 or (index.step or 1) < 0:
                self._fill()
            else:
                self._fill(index.stop)
        elif index < 0:
            self._fill()
        else:
            self._fill(index + 1)
        return self._items[index]
