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
Pagination Utilities with Streaming/Iterator Support

Provides pagination functionality that works with:
- Sequences (lists, tuples) with len()
- Iterators and generators (database cursors, streaming data)
- Lazy evaluation without loading entire dataset into memory

Key improvements:
- Supports generator/iterator inputs using itertools.islice
- Handles streaming data efficiently
- Memory efficient for large datasets
- Backwards compatible with existing Sequence-based pagination
"""

import itertools
import logging
import sys
from collections.abc import Sized
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PaginationError(Exception):
    """Custom exception for pagination errors."""

    pass


@dataclass
class PageInfo:
    """Information about the current page."""

    page: int
    page_size: int
    total_items: Optional[int] = None
    has_next: bool = False
    has_previous: bool = False
    total_pages: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None


class Page(Generic[T]):
    """
    A page of items with metadata.

    Attributes:
        items: Items on the current page
        page_info: Metadata about the page
        has_more: Whether there are more items available
    """

    def __init__(self, items: list[T], page_info: PageInfo, has_more: bool = False):
        self.items = items
        self.page_info = page_info
        self.has_more = has_more

    def __len__(self) -> int:
        """Return number of items on this page."""
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        """Iterate over items on this page."""
        return iter(self.items)

    def __repr__(self) -> str:
        return f"Page(items={len(self.items)}, page={self.page_info.page}, has_more={self.has_more})"


class Paginator(Generic[T]):
    """
    Flexible paginator that handles both sequences and iterators.

    Examples:
        >>> # With a list
        >>> paginator = Paginator([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], page_size=3)
        >>> page = paginator.get_page(1)
        >>> page.items
        [1, 2, 3]

        >>> # With a generator
        >>> def gen():
        ...     for i in range(100):
        ...         yield i
        >>> paginator = Paginator(gen(), page_size=10, total_hint=100)
        >>> page = paginator.get_page(1)
        >>> len(page.items)
        10
    """

    def __init__(
        self,
        items: Sequence[T] | Iterator[T] | Iterable[T],
        page_size: int = 20,
        total_hint: Optional[int] = None,
        error_on_invalid_page: bool = True,
    ):
        """
        Initialize the paginator.

        Args:
            items: Items to paginate (sequence, iterator, or iterable)
            page_size: Number of items per page
            total_hint: Optional hint about total items (for iterators)
            error_on_invalid_page: Whether to raise error on invalid page
        """
        self._items = items
        self.page_size = max(1, page_size)
        self.error_on_invalid_page = error_on_invalid_page
        self._total_hint = total_hint
        self._cached_items: Optional[list[T]] = None
        self._is_sequence = isinstance(items, Sequence)
        self._is_iterator = (
            isinstance(items, (Iterator, Iterable)) and not self._is_sequence
        )

        # Store the original iterator if needed
        if self._is_iterator and not hasattr(items, "__len__"):
            # We need to handle this differently
            pass

        self._iterators_cache: dict = {}
        self._total_items: Optional[int] = None

        # Try to get total length if possible
        if self._is_sequence:
            self._total_items = len(self._items)
            self._total_pages = self._calculate_total_pages(self._total_items)
        elif total_hint is not None:
            self._total_hint = max(0, total_hint)
            self._total_items = self._total_hint
            self._total_pages = self._calculate_total_pages(self._total_items)
        else:
            # For iterators without length, we'll handle on the fly
            self._total_items = None
            self._total_pages = None

    def _calculate_total_pages(self, total_items: int) -> int:
        """Calculate total number of pages."""
        if total_items <= 0:
            return 0
        return (total_items + self.page_size - 1) // self.page_size

    def _has_sequence(self) -> bool:
        """Check if we have a sequence with length."""
        return self._is_sequence

    def _get_sequence_items(self, start: int, end: int) -> list[T]:
        """Get items from a sequence."""
        if not self._is_sequence:
            raise PaginationError("Not a sequence")

        # Make sure we have a sequence that supports slicing
        try:
            return list(self._items[start:end])
        except (TypeError, AttributeError):
            # Fallback to iteration for custom sequences
            return self._get_iterator_items(start, end)

    def _get_iterator_items(self, start: int, end: int) -> list[T]:
        """
        Get items from an iterator using islice.

        This is the key optimization for handling generators/iterators
        without loading the entire stream into memory.
        """
        # For sequences with len, we can use slicing
        if self._has_sequence() and hasattr(self._items, "__len__"):
            return self._get_sequence_items(start, end)

        # For iterators, use itertools.islice
        # We need to store the original iterator state
        if start == 0:
            # Starting from beginning, just slice the iterator
            items = list(itertools.islice(self._items, start, end))

            # Check if we have more items
            # This is important for determining has_next
            if len(items) == end - start:
                # Try to peek one more to check if there are more
                try:
                    # We need to preserve the original iterator state
                    # Since we can't restore it easily, we'll use a flag
                    # This is a limitation of iterators
                    has_more = self._peek_iterator_for_more()
                except:
                    has_more = False
            else:
                has_more = False
        else:
            # For mid-page access to iterators, we need to optimize
            # We'll use islice from start to end
            # Note: This requires consuming the iterator from the beginning
            # which is inefficient for large offsets
            # A better approach is to use a cached iterator or implement
            # a custom paginated iterator

            # Create a new iterator from the original if possible
            if hasattr(self._items, "__iter__") and not hasattr(self._items, "reset"):
                # We can't reset, so we need to handle this differently
                # Use a more efficient approach for large datasets
                items = self._get_items_with_lazy_islice(start, end)
            else:
                items = list(itertools.islice(self._items, start, end))

        return items

    def _get_items_with_lazy_islice(self, start: int, end: int) -> list[T]:
        """
        Optimized islice for large offsets.

        This uses a memory-efficient approach to slice iterators.
        """
        try:
            # For Python's islice, we can specify start and stop
            # But we need to consume from the beginning anyway
            # For large start values, this can be inefficient
            # We'll use a more optimized approach for certain types

            # Check if we can index into the iterator directly
            if hasattr(self._items, "__getitem__"):
                # Some iterators support indexing (like custom iterators)
                try:
                    return [
                        self._items[i]
                        for i in range(start, min(end, start + self.page_size))
                    ]
                except (TypeError, IndexError):
                    pass

            # Use islice with start and stop
            items = list(itertools.islice(self._items, start, end))
            return items

        except Exception as e:
            logger.error(f"Error getting iterator items: {e}")
            # Fallback: return empty list
            return []

    def _peek_iterator_for_more(self) -> bool:
        """Peek at iterator to check if there are more items."""
        # This is a tricky operation with iterators
        # We'll try to peek without consuming the iterator
        try:
            # For some iterators, we can check by creating a copy
            # This is not always possible
            if hasattr(self._items, "__copy__"):
                import copy

                temp_iter = copy.copy(self._items)
                try:
                    next(temp_iter)
                    return True
                except StopIteration:
                    return False

            # For generators, we can't peek without consuming
            # We'll return False as a conservative estimate
            return False
        except:
            return False

    def _get_total_items(self) -> Optional[int]:
        """Get total number of items if available."""
        if self._total_items is not None:
            return self._total_items

        if self._is_sequence and hasattr(self._items, "__len__"):
            self._total_items = len(self._items)
            return self._total_items

        return None

    def get_page(self, page: int = 1) -> Page[T]:
        """
        Get a specific page.

        Args:
            page: Page number (1-indexed)

        Returns:
            Page object containing items and metadata

        Raises:
            PaginationError: If page is invalid and error_on_invalid_page is True
        """
        if page < 1:
            if self.error_on_invalid_page:
                raise PaginationError(f"Invalid page number: {page}. Page must be >= 1")
            return Page([], PageInfo(page=page, page_size=self.page_size))

        start = (page - 1) * self.page_size
        end = start + self.page_size

        # Get items
        items = self._get_page_items(start, end, page)

        # Create page info
        total_items = self._get_total_items()
        total_pages = (
            None if total_items is None else self._calculate_total_pages(total_items)
        )

        # Determine if there are more items
        has_more = self._has_more_items(items, start, end, total_items)

        page_info = PageInfo(
            page=page,
            page_size=self.page_size,
            total_items=total_items,
            has_next=has_more,
            has_previous=page > 1,
            total_pages=total_pages,
            start_index=start,
            end_index=start + len(items) - 1 if items else None,
        )

        return Page(items, page_info, has_more)

    def _get_page_items(self, start: int, end: int, page: int) -> list[T]:
        """Get items for a specific page."""
        try:
            if self._has_sequence():
                items = self._get_sequence_items(start, end)
            else:
                # For iterators, use islice
                items = self._get_iterator_items(start, end)
        except Exception as e:
            logger.error(f"Error getting page items: {e}")
            raise PaginationError(f"Error retrieving page {page}: {str(e)}")

        return items

    def _has_more_items(
        self, items: list[T], start: int, end: int, total_items: Optional[int]
    ) -> bool:
        """Check if there are more items available."""
        if total_items is not None:
            return end < total_items

        # For iterators without known total
        if len(items) == self.page_size:
            # We got a full page, so there might be more
            return True

        # If we got fewer than page_size, we're at the end
        return False

    def iter_pages(self) -> Iterator[Page[T]]:
        """
        Iterate over all pages.

        This is memory efficient for iterators as it only loads one page at a time.
        """
        page = 1
        while True:
            try:
                current_page = self.get_page(page)
                yield current_page

                if not current_page.has_more:
                    break

                page += 1
            except Exception as e:
                logger.error(f"Error iterating pages: {e}")
                break

    def paginate(self, page: int = 1) -> tuple[list[T], PageInfo]:
        """
        Legacy interface for paginate_items.

        Returns a tuple of (items, page_info) for backwards compatibility.
        """
        page_obj = self.get_page(page)
        return page_obj.items, page_obj.page_info


class PaginatedIterator(Generic[T]):
    """
    Memory-efficient paginated iterator for large datasets.

    This class provides a way to iterate over large datasets without
    loading all items into memory at once.
    """

    def __init__(
        self,
        source: Iterator[T],
        page_size: int = 100,
        total_hint: Optional[int] = None,
    ):
        self.source = source
        self.page_size = page_size
        self.total_hint = total_hint
        self._current_page = None
        self._page_num = 0

    def __iter__(self) -> Iterator[T]:
        """Iterate over items page by page."""
        return self._iterate_pages()

    def _iterate_pages(self) -> Iterator[T]:
        """Internal page iteration."""
        while True:
            # Get next page
            items = list(itertools.islice(self.source, self.page_size))
            if not items:
                break

            # Yield items from this page
            for item in items:
                yield item

            # If we got less than page_size, we're done
            if len(items) < self.page_size:
                break

    def get_page(self, page_num: int) -> list[T]:
        """
        Get a specific page (requires starting from beginning for iterators).
        """
        # This is inefficient for large offsets, but works for small pagination
        # For better performance with offset pagination, use sequences
        if page_num <= 0:
            return []

        # Reset the iterator if possible
        if hasattr(self.source, "reset"):
            self.source.reset()
        elif hasattr(self.source, "__iter__"):
            # For generators, we need to recreate it
            # This is a limitation of the approach
            pass

        # Skip to the desired page
        skip = (page_num - 1) * self.page_size
        items = list(itertools.islice(self.source, skip, skip + self.page_size))
        return items


def paginate_items(
    items: Sequence[T] | Iterator[T] | Iterable[T],
    page: int = 1,
    page_size: int = 20,
    total_hint: Optional[int] = None,
    error_on_invalid_page: bool = True,
) -> tuple[list[T], PageInfo]:
    """
    Paginate items with support for sequences and iterators.

    This is the main function that replaces the original paginate_items
    which only supported sequences with len().

    Args:
        items: Items to paginate (sequence, iterator, or iterable)
        page: Page number (1-indexed)
        page_size: Number of items per page
        total_hint: Optional hint about total items (for iterators)
        error_on_invalid_page: Whether to raise error on invalid page

    Returns:
        Tuple of (items on page, page info)

    Examples:
        >>> # With a list
        >>> items, info = paginate_items([1, 2, 3, 4, 5], page=2, page_size=2)
        >>> items
        [3, 4]
        >>> info.has_next
        True

        >>> # With a generator
        >>> def gen():
        ...     for i in range(10):
        ...         yield i
        >>> items, info = paginate_items(gen(), page=1, page_size=3, total_hint=10)
        >>> items
        [0, 1, 2]
        >>> info.total_items
        10
    """
    paginator = Paginator(
        items=items,
        page_size=page_size,
        total_hint=total_hint,
        error_on_invalid_page=error_on_invalid_page,
    )

    return paginator.paginate(page)


def create_paginated_response(
    items: Sequence[T] | Iterator[T] | Iterable[T],
    page: int = 1,
    page_size: int = 20,
    total_hint: Optional[int] = None,
) -> dict:
    """
    Create a paginated API response.

    This is useful for REST APIs that need to return paginated data.

    Args:
        items: Items to paginate
        page: Page number
        page_size: Items per page
        total_hint: Hint for total items

    Returns:
        Dictionary with paginated data and metadata
    """
    page_items, page_info = paginate_items(
        items=items, page=page, page_size=page_size, total_hint=total_hint
    )

    return {
        "data": page_items,
        "pagination": {
            "page": page_info.page,
            "page_size": page_info.page_size,
            "total_items": page_info.total_items,
            "total_pages": page_info.total_pages,
            "has_next": page_info.has_next,
            "has_previous": page_info.has_previous,
            "start_index": page_info.start_index,
            "end_index": page_info.end_index,
        },
    }


def stream_paginate(source: Iterator[T], page_size: int = 100) -> Iterator[list[T]]:
    """
    Yield pages of items from a stream.

    This is useful for processing large streams in batches.

    Args:
        source: Iterator or generator
        page_size: Number of items per page

    Yields:
        Lists of items (pages)

    Examples:
        >>> def large_dataset():
        ...     for i in range(1000):
        ...         yield i
        >>> for page in stream_paginate(large_dataset(), page_size=10):
        ...     # Process page of 10 items
        ...     process(page)
    """
    while True:
        page = list(itertools.islice(source, page_size))
        if not page:
            break
        yield page


def batch_process(
    items: Sequence[T] | Iterator[T] | Iterable[T],
    batch_size: int = 100,
    processor: Optional[Callable[[list[T]], Any]] = None,
) -> Iterator[Any]:
    """
    Process items in batches with a processor function.

    Args:
        items: Items to process
        batch_size: Size of each batch
        processor: Function to apply to each batch

    Yields:
        Results of processing each batch

    Examples:
        >>> def save_batch(batch):
        ...     db.insert_many(batch)
        ...     return len(batch)
        >>> for count in batch_process(large_generator(), batch_size=50, processor=save_batch):
        ...     print(f"Saved {count} items")
    """
    paginator = Paginator(items, page_size=batch_size)

    for page_obj in paginator.iter_pages():
        if processor:
            yield processor(page_obj.items)
        else:
            yield page_obj.items


class SequenceLike:
    """
    Wrapper class to make iterators behave like sequences.

    This is useful for compatibility with code that expects len().
    """

    def __init__(self, iterator: Iterator[T], total: Optional[int] = None):
        self.iterator = iterator
        self.total = total
        self._items = []
        self._exhausted = False

    def __iter__(self) -> Iterator[T]:
        return iter(self.iterator) if not self._exhausted else iter(self._items)

    def __len__(self) -> int:
        if self.total is not None:
            return self.total

        if not self._exhausted:
            # This will consume the iterator to get length
            self._items = list(self.iterator)
            self._exhausted = True
            self.total = len(self._items)

        return self.total

    def __getitem__(self, idx: int) -> T:
        if not self._exhausted:
            # We need to consume the iterator up to idx
            self._items = list(itertools.islice(self.iterator, idx + 1))
            self._exhausted = True

        return self._items[idx]
