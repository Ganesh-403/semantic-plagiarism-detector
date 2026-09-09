"""
src/core/models/__init__.py
---------------------------
Domain model exports for the core business logic layer.
"""

from .categorization import (
    DocumentTag,
    TagCollection,
    TagSource,
    TagCategory,
)

__all__ = [
    "DocumentTag",
    "TagCollection",
    "TagSource",
    "TagCategory",
]
