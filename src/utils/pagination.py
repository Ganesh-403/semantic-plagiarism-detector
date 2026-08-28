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
src/utils/pagination.py
-----------------------
Reusable, framework-independent sequence pagination.

Provides a standardized dataclass for paginated responses with consistent
structure across all API endpoints and UI views that render collections.

The module exposes two layers:

``PaginationPage``
    An immutable description of one page. Callers that already know their
    page geometry build it directly or through :meth:`PaginationPage.create`.

``paginate_items``
    The slicing entry point. It *clamps* rather than raises: a page number
    past the end returns the last page, a nonsensical page size falls back to
    a usable one, and a value that is not a number at all falls back to its
    default. Pagination arguments usually arrive from a query string or a
    Streamlit widget, so refusing to render is a worse answer than rendering
    the nearest sensible page.

Recent Additions (Issue #1998):
- Added custom __repr__ for human-friendly debugging output
- Verified __eq__ works correctly via dataclass frozen=True comparison

Recent Fixes (Issue #3045):
- Restored the keyword-only ``page`` / ``page_size`` / ``max_page_size``
  contract that ``src/utils/warning_list.py`` calls, along with the
  ``_coerce_integer`` helper and the ``start_index`` / ``end_index`` fields.

Recent Additions (Issue #3215):
- Added ``CursorPaginationPage`` and ``paginate_by_cursor`` for cursor-based
  pagination. Offset pagination (``paginate_items`` / ``PaginationPage``)
  requires the database to scan and discard every skipped row, which gets
  slow once a table (e.g. the incidents table) reaches tens of thousands of
  rows. Cursor pagination instead resumes from an opaque token derived from
  the last row of the previous page, so a query can use ``WHERE (sort_key)
  > (cursor)`` instead of ``OFFSET n``.

Recent Additions (Issue #3218):
- ``PaginationPage.was_clamped`` records whether ``paginate_items`` had to
  pull an out-of-range page number back into range, so API callers can tell
  a genuine last-page response from a clamped one.
"""

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

#: Fallback page size used when a caller supplies neither a usable
#: ``page_size`` nor an explicit default.
DEFAULT_PAGE_SIZE = 10

#: Upper bound applied to ``page_size`` unless a caller overrides it. Keeps a
#: hand-edited ``?per_page=100000`` from materialising the whole table.
DEFAULT_MAX_PAGE_SIZE = 100


@dataclass(frozen=True)
class PaginationPage(Generic[T]):
    """Represents a single page of paginated results.

    This frozen dataclass provides a standardized structure for paginated
    API responses. Being frozen ensures immutability after creation,
    which is important for caching and thread safety.

    Attributes:
        items: List of items on the current page
        page: Current page number (1-indexed)
        total_pages: Total number of pages available
        total_items: Total number of items across all pages
        per_page: Number of items per page
        start_index: One-based inclusive position of the first item on this
            page within the full sequence. ``0`` when the page is empty.
        end_index: One-based inclusive position of the last item on this
            page within the full sequence. ``0`` when the page is empty.
        was_clamped: True when a helper had to adjust an out-of-range page
            number to produce this page (Issue #3218). Pages built directly,
            or sliced from an in-range request, are False.

    Recent Additions (Issue #1998):
        Custom __repr__ truncates large item lists for readability.
        __eq__ is automatically provided by @dataclass decorator.

    Recent Fixes (Issue #3045):
        ``start_index`` / ``end_index`` are back, and ``page_size`` is
        available as an alias of ``per_page`` for callers that speak the
        helper's vocabulary rather than the dataclass's.
    """

    items: list[T]
    page: int
    total_pages: int
    total_items: int
    per_page: int
    start_index: int = field(default=0)
    end_index: int = field(default=0)
    was_clamped: bool = field(default=False)

    @property
    def page_size(self) -> int:
        """Alias for :attr:`per_page`.

        ``paginate_items`` and its callers talk about a ``page_size``; the
        dataclass field has always been named ``per_page``. Exposing both
        names means neither vocabulary has to win, and no caller has to
        remember which one this object speaks.

        Returns:
            The number of items one full page holds.
        """
        return self.per_page

    def __repr__(self) -> str:
        """Return a human-friendly string representation.

        Truncates the items list display when it contains more than 3 items
        to prevent console flooding during debugging. Shows the count of
        items instead of the full list.

        The class name is read from ``type(self)`` rather than hardcoded, so
        it cannot drift from the class again and subclasses report their own
        name.

        Returns:
            Formatted string showing page info, item count, and page size.

        Examples:
            >>> page = PaginationPage(items=[1,2,3,4,5], page=1, total_pages=2, total_items=10, per_page=5)
            >>> repr(page)
            'PaginationPage(page=1/2, items=5, per_page=5)'

            >>> small_page = PaginationPage(items=[1,2], page=1, total_pages=1, total_items=2, per_page=10)
            >>> repr(small_page)
            'PaginationPage(page=1/1, items=[1, 2], per_page=10)'
        """
        # Show full items list if 3 or fewer items
        if len(self.items) <= 3:
            items_repr = repr(self.items)
        else:
            # Truncate to show count only for large lists
            items_repr = f"{len(self.items)}"

        return (
            f"{type(self).__name__}("
            f"page={self.page}/{self.total_pages}, "
            f"items={items_repr}, "
            f"per_page={self.per_page}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another PaginationPage instance.

        Two PaginationPage instances are equal if and only if all their
        fields (items, page, total_pages, total_items, per_page, start_index,
        end_index, was_clamped) are equal. This is automatically handled by
        the @dataclass decorator when frozen=True, but we document it
        explicitly for clarity.

        Args:
            other: Another object to compare against.

        Returns:
            True if all fields match, False otherwise.

        Note:
            The @dataclass decorator automatically generates __eq__ based
            on all fields. This explicit definition is for documentation
            purposes and to ensure the behavior is clear to users.
        """
        if not isinstance(other, PaginationPage):
            return False

        return (
            self.items == other.items
            and self.page == other.page
            and self.total_pages == other.total_pages
            and self.total_items == other.total_items
            and self.per_page == other.per_page
            and self.start_index == other.start_index
            and self.end_index == other.end_index
            and self.was_clamped == other.was_clamped
        )

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`.

        ``@dataclass(frozen=True)`` generates a hash over every field, and
        ``items`` is a ``list`` — so the generated hash raised
        ``TypeError: unhashable type: 'list'`` for any page that carried
        results. Hashing the items as a tuple keeps the invariant that equal
        pages hash equally while letting a page be used as a dict key or set
        member, which is what a frozen dataclass is for.

        Returns:
            A hash over the same fields ``__eq__`` compares.

        Note:
            The items themselves must be hashable. A page of dictionaries —
            the shape ``warning_list`` builds — is still unhashable, and that
            is the ordinary Python contract rather than something this class
            can paper over.
        """
        return hash(
            (
                tuple(self.items),
                self.page,
                self.total_pages,
                self.total_items,
                self.per_page,
                self.start_index,
                self.end_index,
                self.was_clamped,
            )
        )

    @classmethod
    def create(
        cls,
        items: list[T],
        page: int,
        per_page: int,
        total_items: int,
    ) -> "PaginationPage[T]":
        """Factory method to create a PaginationPage with calculated total_pages.

        Args:
            items: List of items for the current page
            page: Current page number (1-indexed)
            per_page: Number of items per page
            total_items: Total number of items across all pages

        Returns:
            PaginationPage instance with calculated total_pages

        Raises:
            ValueError: If page < 1 or per_page < 1

        Examples:
            >>> page = PaginationPage.create(items=[1,2,3], page=1, per_page=10, total_items=25)
            >>> page.total_pages
            3
        """
        if page < 1:
            raise ValueError(f"page must be >= 1, got {page}")
        if per_page < 1:
            raise ValueError(f"per_page must be >= 1, got {per_page}")

        # Calculate total pages, ensuring at least 1 page even if no items
        total_pages = max(1, (total_items + per_page - 1) // per_page)

        start_index, end_index = _bounds_for(
            page=page,
            per_page=per_page,
            item_count=len(items),
        )

        return cls(
            items=items,
            page=page,
            total_pages=total_pages,
            total_items=total_items,
            per_page=per_page,
            start_index=start_index,
            end_index=end_index,
        )

    def has_next(self) -> bool:
        """Check if there is a next page available.

        Returns:
            True if current page < total_pages, False otherwise
        """
        return self.page < self.total_pages

    @property
    def next_page(self) -> Optional[int]:
        return self.page + 1 if self.has_next() else None

    @property
    def prev_page(self) -> Optional[int]:
        return self.page - 1 if self.has_previous() else None

    def has_previous(self) -> bool:
        """Check if there is a previous page available."""
        return self.page > 1

        """Get the next page number if available.

        Returns:
            Next page number or None if on last page
        """
        return self.page + 1 if self.has_next() else None

    def previous_page(self) -> Optional[int]:
        """Get the previous page number if available.

        Returns:
            Previous page number or None if on first page
        """
        return self.page - 1 if self.has_previous() else None

    def to_dict(self) -> dict:
        """Convert to dictionary representation for JSON serialization.

        Returns:
            Dictionary with all pagination fields
        """
        return {
            "items": self.items,
            "page": self.page,
            "total_pages": self.total_pages,
            "total_items": self.total_items,
            "per_page": self.per_page,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "was_clamped": self.was_clamped,
            "has_next": self.has_next(),
            "has_previous": self.has_previous(),
            "next_page": self.next_page(),
            "previous_page": self.previous_page(),
        }


def _coerce_integer(value: object, default: int) -> int:
    """Convert a pagination value to ``int``, or fall back to *default*.

    Pagination numbers reach this module from query strings, session state and
    widget callbacks, so ``"3"``, ``3.0`` and ``None`` all turn up where an
    ``int`` is expected. Anything that will not survive ``int()`` yields the
    caller's default rather than an exception.

    Args:
        value: The raw value to convert. May be a string, float, ``None`` or
            any other object.
        default: The value returned when *value* cannot be converted.

    Returns:
        The converted integer, or *default*.

    Examples:
        >>> _coerce_integer("10", 1)
        10
        >>> _coerce_integer("abc", 1)
        1
        >>> _coerce_integer(3.9, 1)
        3
    """
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return default


def _bounds_for(*, page: int, per_page: int, item_count: int) -> tuple[int, int]:
    """Return the one-based inclusive ``(start, end)`` positions of a page.

    Args:
        page: The clamped, one-based page number.
        per_page: The clamped page size.
        item_count: How many items the page actually holds.

    Returns:
        ``(0, 0)`` for an empty page, otherwise the first and last positions
        of its items within the full sequence.
    """
    if item_count <= 0:
        return 0, 0

    start_index = (page - 1) * per_page + 1
    return start_index, start_index + item_count - 1


def paginate_items(
    items: Sequence[T],
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_page_size: Optional[int] = DEFAULT_MAX_PAGE_SIZE,
) -> PaginationPage[T]:
    """Slice *items* into one clamped page.

    Every argument is clamped rather than validated, because the values come
    from user-controlled surfaces where refusing to render is a worse outcome
    than rendering the nearest sensible page:

    * a non-numeric ``page`` or ``page_size`` falls back to its default;
    * ``page_size`` is clamped into ``[1, max_page_size]``;
    * ``page`` is clamped into ``[1, total_pages]``, so a bookmarked
      ``?page=9999`` lands on the last page instead of an empty one.

    The returned page carries ``was_clamped=True`` whenever that last rule
    fired, so API callers can distinguish "the user really asked for the
    last page" from "an out-of-range request was pulled back into range"
    (Issue #3218). A non-numeric ``page`` is a coercion to the default, not
    an out-of-range adjustment, and therefore leaves the flag False.

    Args:
        items: The full sequence to paginate. Not mutated.
        page: One-based page number requested by the caller.
        page_size: Requested number of items per page.
        max_page_size: Upper bound for ``page_size``. Pass ``None`` to lift
            the cap entirely.

    Returns:
        A :class:`PaginationPage` holding that slice, its clamped geometry,
        and the one-based ``start_index`` / ``end_index`` of the slice.

    Examples:
        >>> paginate_items([1, 2, 3, 4, 5], page=9999, page_size=2).items
        [5]
        >>> paginate_items([1, 2, 3, 4, 5], page=9999, page_size=2).page
        3
        >>> paginate_items([1, 2, 3, 4, 5], page=9999, page_size=2).was_clamped
        True
        >>> paginate_items([1, 2, 3, 4, 5], page=2, page_size=2).was_clamped
        False
        >>> paginate_items([], page=1, page_size=10).total_pages
        1
    """
    safe_page_size = _coerce_integer(page_size, DEFAULT_PAGE_SIZE)
    safe_page_size = max(1, safe_page_size)
    if max_page_size is not None:
        safe_page_size = min(safe_page_size, max(1, max_page_size))

    total_items = len(items)
    total_pages = max(1, -(-total_items // safe_page_size))

    safe_page = _coerce_integer(page, 1)
    was_clamped = safe_page < 1 or safe_page > total_pages
    safe_page = min(max(1, safe_page), total_pages)

    start = (safe_page - 1) * safe_page_size
    page_items = list(items[start : start + safe_page_size])

    start_index, end_index = _bounds_for(
        page=safe_page,
        per_page=safe_page_size,
        item_count=len(page_items),
    )

    return PaginationPage(
        items=page_items,
        page=safe_page,
        total_pages=total_pages,
        total_items=total_items,
        per_page=safe_page_size,
        start_index=start_index,
        end_index=end_index,
        was_clamped=was_clamped,
    )


def encode_cursor(value: Any) -> str:
    """Encode a sort-key value into an opaque cursor string.

    The cursor is a base64url-encoded JSON representation of *value*. It is
    deliberately opaque to callers: cursors should be treated as tokens to
    pass back verbatim, not decoded or constructed by hand.

    Args:
        value: The sort-key value to encode. Typically a scalar
            (``str``/``int``/``float``) or a small tuple/list of scalars for
            composite sort keys (e.g. ``(date_flagged, incident_id)``).

    Returns:
        An opaque, URL-safe cursor string.

    Examples:
        >>> cursor = encode_cursor(("2026-08-01T00:00:00Z", 42))
        >>> decode_cursor(cursor)
        ['2026-08-01T00:00:00Z', 42]
    """
    payload = json.dumps(value, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode_cursor(cursor: str) -> Any:
    """Decode a cursor string produced by :func:`encode_cursor`.

    Args:
        cursor: The opaque cursor string.

    Returns:
        The original sort-key value (or ``None`` if *cursor* is falsy).

    Raises:
        ValueError: If *cursor* is not a valid cursor produced by
            :func:`encode_cursor` (malformed base64 or JSON).
    """
    if not cursor:
        return None
    try:
        payload = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        return json.loads(payload)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid pagination cursor: {cursor!r}") from exc


@dataclass(frozen=True)
class CursorPaginationPage(Generic[T]):
    """Represents a single page of cursor-paginated results.

    Unlike :class:`PaginationPage`, this does not carry a total item/page
    count — computing those for a large table requires the same expensive
    ``COUNT(*)``/``OFFSET`` scan cursor pagination exists to avoid. Callers
    that need a "page 3 of 40"-style UI should use :class:`PaginationPage`
    instead; cursor pagination is for "load more" / infinite-scroll style
    interfaces over large or frequently-changing tables.

    Attributes:
        items: List of items on the current page, in the query's sort order.
        next_cursor: Opaque token to fetch the next page, or ``None`` if this
            is the last page.
        prev_cursor: Opaque token to fetch the previous page, or ``None`` if
            this is the first page.
        has_more: ``True`` if at least one more item exists after ``items``.
    """

    items: list[T]
    next_cursor: Optional[str]
    prev_cursor: Optional[str]
    has_more: bool

    def to_dict(self) -> dict:
        """Convert to dictionary representation for JSON serialization.

        Returns:
            Dictionary with all cursor-pagination fields.
        """
        return {
            "items": self.items,
            "next_cursor": self.next_cursor,
            "prev_cursor": self.prev_cursor,
            "has_more": self.has_more,
        }


def paginate_by_cursor(
    items: Sequence[T],
    *,
    cursor_key: Callable[[T], Any],
    limit: int = DEFAULT_PAGE_SIZE,
    prev_cursor: Optional[str] = None,
) -> CursorPaginationPage[T]:
    """Build a :class:`CursorPaginationPage` from an over-fetched query result.

    This is the cursor-pagination counterpart to :func:`paginate_items`. It
    does not itself query a database — callers are expected to fetch
    ``limit + 1`` rows ordered by their cursor column(s) (optionally with a
    ``WHERE (sort_key) > (decode_cursor(cursor))`` clause instead of
    ``OFFSET``), and pass that raw result in as *items*. Fetching one extra
    row lets this function determine ``has_more`` without a separate
    ``COUNT(*)`` query.

    Args:
        items: The over-fetched sequence, containing up to ``limit + 1``
            items in cursor sort order. Not mutated.
        cursor_key: Callable that extracts the sort-key value from an item,
            used to build ``next_cursor`` from the last item kept on the
            page (e.g. ``lambda incident: (incident.date_flagged,
            incident.incident_id)`` for a composite sort key).
        limit: Maximum number of items to return on this page.
        prev_cursor: The cursor that was used to fetch *this* page, echoed
            back unchanged so the caller can request the previous page.
            ``None`` on the first page.

    Returns:
        A :class:`CursorPaginationPage` holding at most ``limit`` items.

    Examples:
        >>> rows = [{"id": 1}, {"id": 2}, {"id": 3}]  # limit=2, one extra row
        >>> page = paginate_by_cursor(rows, cursor_key=lambda r: r["id"], limit=2)
        >>> page.items
        [{'id': 1}, {'id': 2}]
        >>> page.has_more
        True
        >>> decode_cursor(page.next_cursor)
        2
    """
    safe_limit = max(1, _coerce_integer(limit, DEFAULT_PAGE_SIZE))

    has_more = len(items) > safe_limit
    page_items = list(items[:safe_limit])

    next_cursor = (
        encode_cursor(cursor_key(page_items[-1])) if has_more and page_items else None
    )

    return CursorPaginationPage(
        items=page_items,
        next_cursor=next_cursor,
        prev_cursor=prev_cursor,
        has_more=has_more,
    )
