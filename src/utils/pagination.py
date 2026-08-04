"""Reusable, framework-independent sequence pagination."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar


ItemT = TypeVar("ItemT")


@dataclass(frozen=True)
class PaginationPage(Generic[ItemT]):
    """One clamped page of a larger sequence.

    ``start_index`` and ``end_index`` are human-readable, one-based
    inclusive positions. Both are zero when the input is empty.
    """

    items: list[ItemT]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    start_index: int
    end_index: int


def _coerce_integer(
    value: object,
    *,
    default: int,
) -> int:
    """Convert a pagination value to ``int`` or use its default."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def paginate_items(
    items: Sequence[ItemT],
    *,
    page: int = 1,
    page_size: int = 10,
    max_page_size: int | None = 100,
) -> PaginationPage[ItemT]:
    """Return a safe, clamped page from ``items``.

    Invalid numeric values use the defaults. Page numbers are clamped
    to the available range, and page size is constrained to at least
    one and optionally to ``max_page_size``.
    """
    if max_page_size is not None and max_page_size < 1:
        raise ValueError(
            "max_page_size must be at least 1 or None."
        )

    requested_page = _coerce_integer(
        page,
        default=1,
    )
    requested_page_size = _coerce_integer(
        page_size,
        default=10,
    )

    safe_page_size = max(1, requested_page_size)
    if max_page_size is not None:
        safe_page_size = min(
            safe_page_size,
            max_page_size,
        )

    total_items = len(items)
    total_pages = max(
        1,
        math.ceil(total_items / safe_page_size),
    )
    safe_page = min(
        max(1, requested_page),
        total_pages,
    )

    start_offset = (
        safe_page - 1
    ) * safe_page_size
    end_offset = min(
        start_offset + safe_page_size,
        total_items,
    )

    return PaginationPage(
        items=list(items[start_offset:end_offset]),
        total_items=total_items,
        page=safe_page,
        page_size=safe_page_size,
        total_pages=total_pages,
        start_index=(
            start_offset + 1
            if total_items
            else 0
        ),
        end_index=end_offset,
    )
