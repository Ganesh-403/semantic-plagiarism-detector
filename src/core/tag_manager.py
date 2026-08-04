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
    def parse_tags(raw_input: str) -> str:
        """
        Parses a comma-separated or space-separated string of tags into a normalized,
        comma-separated string for DB storage.

        Tags are converted to lowercase and stripped of any non-alphanumeric
        characters (except the '#' prefix). If a '#' is missing, it is automatically
        prepended. Duplicate tags are removed, and the output is sorted alphabetically
        for consistent hashing and indexing.

        Args:
            raw_input (str): The raw user input string containing tags.

        Returns:
            str: A clean, sorted, comma-separated string of normalized tags.
                 Returns an empty string if the input is invalid or empty.

        Example:
            >>> TagManager.parse_tags("#hw1, FINAL,   #draft")
            '#draft,#final,#hw1'
        """
        if not raw_input or not isinstance(raw_input, str):
            logger.debug(f"TagManager received empty or invalid input: {raw_input}")
            return ""

        # Split by comma or space using regex to handle multiple spaces/commas gracefully
        tokens = re.split(r'[,\s]+', raw_input)

        normalized_tags: Set[str] = set()

        for token in tokens:
            token = token.strip().lower()
            if not token:
                continue

            # Strip all non-alphanumeric except existing hash
            # This prevents SQL injection payloads or weird UI rendering issues
            clean_token = re.sub(r'[^a-z0-9#]', '', token)

            # If after stripping the token is empty or just a hash, skip it
            if not clean_token or clean_token == '#':
                continue

            # Ensure it starts with a hash prefix
            if not clean_token.startswith('#'):
                clean_token = '#' + clean_token

            normalized_tags.add(clean_token)

        final_tags = ",".join(sorted(normalized_tags))
        logger.debug(f"TagManager parsed '{raw_input}' into '{final_tags}'")
        return final_tags

    @staticmethod
    def extract_unique_tags(db_tags_column: List[str]) -> List[str]:
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
            if tag_str and isinstance(tag_str, str):
                individual_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
                unique_tags.update(individual_tags)

        return sorted(list(unique_tags))

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
        doc_tags = [t.strip() for t in doc_tags_str.split(",") if t.strip()]
        return filter_tag in doc_tags

    @classmethod
    def apply_tag(cls, document_ids: List[str], tag: str) -> None:
        """
        Applies a specific tag to a list of documents.
        Skips documents that already have the tag.

        Args:
            document_ids (List[str]): The IDs (filenames) of documents.
            tag (str): The tag to apply.
        """
        from src.db.corpus_db import get_document_tags, update_document_tags

        normalized_tag = cls.parse_tags(tag)
        if not normalized_tag:
            return

        for doc_id in document_ids:
            current_tags_str = get_document_tags(doc_id)
            if not current_tags_str:
                new_tags_str = normalized_tag
            else:
                individual_tags = [t.strip() for t in current_tags_str.split(",") if t.strip()]
                if normalized_tag in individual_tags:
                    continue
                individual_tags.append(normalized_tag)
                new_tags_str = ",".join(sorted(set(individual_tags)))

            update_document_tags(doc_id, new_tags_str)

    @classmethod
    def remove_tag(cls, document_ids: List[str], tag: str) -> None:
        """
        Removes a specific tag from a list of documents.
        Ignores documents that don't have the tag.

        Args:
            document_ids (List[str]): The IDs (filenames) of documents.
            tag (str): The tag to remove.
        """
        from src.db.corpus_db import get_document_tags, update_document_tags

        normalized_tag = cls.parse_tags(tag)
        if not normalized_tag:
            return

        for doc_id in document_ids:
            current_tags_str = get_document_tags(doc_id)
            if not current_tags_str:
                continue
            individual_tags = [t.strip() for t in current_tags_str.split(",") if t.strip()]
            if normalized_tag not in individual_tags:
                continue

            updated_tags = [t for t in individual_tags if t != normalized_tag]
            new_tags_str = ",".join(sorted(set(updated_tags))) if updated_tags else ""
            update_document_tags(doc_id, new_tags_str)

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

    # Remove HTML tags (e.g. <script>, <b>, etc.)
    cleaned = re.sub(r'<[^>]*>', '', tag)

    # Remove slashes (/ and \) and all whitespace characters
    cleaned = re.sub(r'[/\\\s]', '', cleaned)

    if not cleaned:
        raise ValueError("Tag name cannot be empty or whitespace-only.")

    return cleaned[:30]
