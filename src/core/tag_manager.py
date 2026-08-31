import html
import logging
import re
from typing import List, Set

logger = logging.getLogger(__name__)


class TagManager:
    """
    Core utility to parse, normalize, and handle document tags.
    Ensures consistent formatting (e.g., lowercase, alphanumeric, hash-prefixed).

    This manager provides robust tagging features that allow instructors
    to group documents by assignment type (e.g. #hw1, #final), class section,
    or academic year. By doing so, similarity matrices can be dynamically filtered
    to only show relevant comparisons, drastically reducing noise in large datasets.
    """

    @staticmethod
    def normalize_tags(raw_input: str) -> list[str]:
        """
        Parses a comma-separated or space-separated string of tags into a sorted
        list of individual normalized tags.

        Tags are converted to lowercase and stripped of any non-alphanumeric
        characters (except the leading '#'). If a '#' is missing, it is
        automatically prepended. Duplicates are removed and the result is sorted
        alphabetically for consistent hashing and indexing.

        This is the canonical form. Callers that need to store the value in the
        single-column tags field should join it themselves, or use
        ``parse_tags()``; callers that operate on tags one at a time — applying
        or removing them — must use this method, because a single string cannot
        be compared against the individual entries of a document's tag list.

        Args:
            raw_input (str): The raw user input string containing tags.

        Returns:
            List[str]: Sorted, deduplicated, individually normalized tags.
                       Empty if the input is invalid or contains no valid tag.

        Example:
            >>> TagManager.normalize_tags("#hw1, FINAL,   #draft")
            ['#draft', '#final', '#hw1']
        """
        if not raw_input or not isinstance(raw_input, str):
            logger.debug(f"TagManager received empty or invalid input: {raw_input}")
            return []

        # Split by comma or space using regex to handle multiple spaces/commas gracefully
        tokens = re.split(r"[,\s]+", raw_input)

        normalized_tags: set[str] = set()

        for token in tokens:
            token = token.strip().lower()
            if not token:
                continue

            # Strip all non-alphanumeric except existing hash
            # This prevents SQL injection payloads or weird UI rendering issues
            clean_token = re.sub(r"[^a-z0-9#]", "", token)

            # A hash is only meaningful as a prefix. Leaving interior ones in
            # place produced tags like "#hw1#final" that no filter can match.
            clean_token = clean_token.lstrip("#").replace("#", "")

            # If after stripping the token is empty, skip it
            if not clean_token:
                continue

            # Skip purely numeric or non-alpha tokens (must contain at least one alphabetic character)
            if not re.sub(r"[^a-z]", "", clean_token):
                continue

            normalized_tags.add("#" + clean_token)

        final_tags = sorted(normalized_tags)
        logger.debug(f"TagManager parsed '{raw_input}' into {final_tags}")
        return final_tags

    @staticmethod
    def parse_tags(raw_input: str) -> str:
        """
        Parses a comma-separated or space-separated string of tags into a normalized,
        comma-separated string for DB storage.

        This is the storage representation of ``normalize_tags()``. Because the
        result may hold several tags, it must not be compared against a single
        entry of a document's tag list — use ``normalize_tags()`` for that.

        Args:
            raw_input (str): The raw user input string containing tags.

        Returns:
            str: A clean, sorted, comma-separated string of normalized tags.
                 Returns an empty string if the input is invalid or empty.

        Example:
            >>> TagManager.parse_tags("#hw1, FINAL,   #draft")
            '#draft,#final,#hw1'
        """
        return ",".join(TagManager.normalize_tags(raw_input))

    @staticmethod
    def extract_unique_tags(db_tags_column: list[str]) -> list[str]:
        """
        Takes a list of raw tag strings from the DB (e.g. ["#hw1,#final", "#hw1,#draft"])
        and returns a sorted list of unique individual tags across the entire corpus.

        Args:
            db_tags_column (List[str]): A list of comma-separated tag strings retrieved from the database.

        Returns:
            List[str]: A deduplicated, sorted list of all individual tags.
        """
        unique_tags = set()
        if not db_tags_column:
            return []

        for tag_str in db_tags_column:
            unique_tags.update(TagManager._split_tags(tag_str))

        return sorted(unique_tags)

    @staticmethod
    def has_matching_tag(doc_tags_str: str, filter_tag: str) -> bool:
        """
        Returns True if the filter_tag exists in the document's tag string.
        Returns True if filter_tag is empty or "All Tags" (indicating no filter is active).

        Args:
            doc_tags_str (str): The comma-separated tags associated with a document.
            filter_tag (str): The specific tag to filter by (e.g., '#hw1').

        Returns:
            bool: True if the document matches the filter criteria, False otherwise.
        """
        # If no specific filter is selected, everything matches
        if not filter_tag or filter_tag == "All Tags":
            return True

        # If a filter is selected but the document has no tags, it cannot match
        if not doc_tags_str or not isinstance(doc_tags_str, str):
            return False

        # Split document tags and check for exact inclusion
        return filter_tag in TagManager._split_tags(doc_tags_str)

    @classmethod
    def apply_tag(cls, document_ids: list[str], tag: str) -> None:
        """
        Applies one or more tags to a list of documents.
        Documents that already carry every requested tag are left untouched.

        Args:
            document_ids (List[str]): The IDs (filenames) of documents.
            tag (str): The tag or tags to apply. Accepts the same
                comma/space-separated input as ``normalize_tags()``.
        """
        from src.db.corpus_db import get_document_tags, update_document_tags

        # normalize_tags(), not parse_tags(): the input may name several tags,
        # and each has to be added to the document's list on its own. Adding the
        # joined string instead stored "#final,#hw1" as a single entry, which no
        # filter could ever match and which corrupted the column on the next
        # write, when the entry was split back apart at the commas.
        new_tags = cls.normalize_tags(tag)
        if not new_tags:
            return

        for doc_id in document_ids:
            current_tags_str = get_document_tags(doc_id)
            existing_tags = cls._split_tags(current_tags_str)

            missing_tags = [t for t in new_tags if t not in existing_tags]
            if not missing_tags:
                continue

            merged = sorted(set(existing_tags) | set(new_tags))
            update_document_tags(doc_id, ",".join(merged))

    @classmethod
    def remove_tag(cls, document_ids: list[str], tag: str) -> None:
        """
        Removes one or more tags from a list of documents.
        Documents that carry none of the requested tags are left untouched.

        Args:
            document_ids (List[str]): The IDs (filenames) of documents.
            tag (str): The tag or tags to remove. Accepts the same
                comma/space-separated input as ``normalize_tags()``.
        """
        from src.db.corpus_db import get_document_tags, update_document_tags

        doomed_tags = set(cls.normalize_tags(tag))
        if not doomed_tags:
            return

        for doc_id in document_ids:
            current_tags_str = get_document_tags(doc_id)
            existing_tags = cls._split_tags(current_tags_str)

            if not doomed_tags.intersection(existing_tags):
                continue

            remaining = sorted({t for t in existing_tags if t not in doomed_tags})
            update_document_tags(doc_id, ",".join(remaining))

    @staticmethod
    def _split_tags(tags_str: str) -> list[str]:
        """Split a stored tags column into its individual, trimmed entries."""
        if not tags_str or not isinstance(tags_str, str):
            return []
        return [t.strip() for t in tags_str.split(",") if t.strip()]

    @staticmethod
    def sanitize_tag_name(tag: str) -> str:
        """
        Sanitizes a tag name string by removing HTML tags, slashes, and whitespace.
        Limits the output length to a maximum of 30 characters.
        Rejects empty or whitespace-only tags by raising a ValueError.

        Args:
            tag (str): The raw tag name string to sanitize.

        Returns:
            str: Cleaned tag name string (up to 30 characters).

        Raises:
            ValueError: If the input tag is invalid, empty, or whitespace-only.
        """
        return sanitize_tag_name(tag)


def sanitize_tag_name(tag: str) -> str:
    """
    Sanitizes a tag name string by removing HTML tags, slashes, and whitespace.
    Limits the output length to a maximum of 30 characters.
    Rejects empty or whitespace-only tags by raising a ValueError.

    Args:
        tag (str): The raw tag name string to sanitize.

    Returns:
        str: Cleaned tag name string (up to 30 characters).

    Raises:
        ValueError: If the input tag is invalid, empty, or whitespace-only.
    """
    if tag is None or not isinstance(tag, str) or not tag.strip():
        raise ValueError("Tag name cannot be empty or whitespace-only.")

    # Unescape HTML entities first (e.g. &lt;script&gt; -> <script>)
    cleaned = html.unescape(tag)

    # Remove HTML tags (e.g. <script>, <b>, etc.)
    cleaned = re.sub(r"<[^>]*>", "", cleaned)

    # Remove slashes (/ and \) and all whitespace characters
    cleaned = re.sub(r"[/\\\s]", "", cleaned)

    if not cleaned:
        raise ValueError("Tag name cannot be empty or whitespace-only.")

    return cleaned[:30]
