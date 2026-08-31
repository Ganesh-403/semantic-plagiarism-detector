"""
src/core/citation_extractor.py
------------------------------
Automated Citation Extraction Engine.

Parses bibliography sections from extracted document text using regex and NLP
heuristics. Supports APA, IEEE and MLA formats, with a year-based fallback for
the messy reference lists real student submissions contain, so citation
laundering and shared bibliography rings can be detected.

Every entry is returned as a :class:`Citation`. Two consumers read that value
and they want different shapes, so :class:`Citation` serves both:

* :mod:`src.db.citation_graph_db` builds graph nodes from
  :meth:`Citation.get_normalized_key` and the ``authors`` / ``year`` /
  ``title`` / ``source`` attributes.
* :mod:`src.db.citation_db` stores rows keyed on the fuzzy match hash and
  reads them with mapping syntax (``cit["hash"]``, ``cit["author"]``).

Recent history (Issue #3565):
    This module had been left holding two complete implementations
    concatenated end to end -- one returning ``Citation`` objects, one
    returning dictionaries. The dictionary version won by virtue of being
    last, which broke citation graph ingestion. The two are now merged: the
    ``Citation`` contract from the first, the IEEE pattern, fuzzy hashing,
    fallback heuristic and per-document deduplication from the second.
"""

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Citation record ───────────────────────────────────────────────────────────

#: Mapping-style keys accepted by :meth:`Citation.__getitem__`, pointing at the
#: attribute that backs each one. ``author`` and ``hash`` are the spellings the
#: database layer uses; the others match the attribute names.
_ITEM_KEYS: dict[str, str] = {
    "author": "authors",
    "authors": "authors",
    "hash": "citation_hash",
    "citation_hash": "citation_hash",
    "year": "year",
    "title": "title",
    "source": "source",
    "raw_text": "raw_text",
    "format_detected": "format_detected",
}


@dataclass
class Citation:
    """A single parsed bibliography entry.

    Attributes:
        raw_text: The bibliography line the entry was parsed from.
        authors: Author list, as written.
        year: Four-digit publication year, or ``"Unknown"``.
        title: Work title, with trailing bibliography punctuation removed.
        source: Journal, publisher or container, where the format exposes one.
        format_detected: ``"APA"``, ``"IEEE"``, ``"MLA"`` or ``"HEURISTIC"``.
        citation_hash: Fuzzy match hash. Computed from the normalised author,
            year and title when not supplied, so two documents citing the same
            work with slightly different formatting produce the same value.
    """

    raw_text: str
    authors: str
    year: str
    title: str
    source: str
    format_detected: str
    citation_hash: str = field(default="")

    def __post_init__(self) -> None:
        """Derive the fuzzy match hash when the caller did not supply one."""
        if not self.citation_hash:
            self.citation_hash = generate_citation_hash(
                self.authors, self.year, self.title
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a dictionary.

        Carries both spellings the database layer uses (``author`` alongside
        ``authors``, ``hash`` alongside ``citation_hash``) so a row can be
        written straight from this mapping.
        """
        data = asdict(self)
        data["author"] = self.authors
        data["hash"] = self.citation_hash
        return data

    def __getitem__(self, key: str) -> str:
        """Support mapping access for callers written against the dict API.

        ``src.db.citation_db.add_document_citations`` reads ``cit["hash"]``,
        ``cit["author"]`` and friends. Rather than force every caller to change
        shape, the record answers to both.
        """
        try:
            attribute = _ITEM_KEYS[key]
        except KeyError:
            raise KeyError(key) from None
        return getattr(self, attribute)

    def get(self, key: str, default: Any = None) -> Any:
        """Mapping-style lookup with a default, mirroring ``dict.get``."""
        try:
            return self[key]
        except KeyError:
            return default

    def get_normalized_key(self) -> str:
        """Generate a normalized key for graph node matching.

        Combines the first author, the year, and the first five title words,
        which is coarse enough to survive formatting differences between two
        bibliographies citing the same work.
        """
        title_words = "_".join(self.title.lower().split()[:5])
        return f"{self.authors.lower().split(',')[0]}_{self.year}_{title_words}"


# ── Citation formats ──────────────────────────────────────────────────────────

# APA: Author, A. A. (Year). Title of work. Source.
_APA_PATTERN = re.compile(
    r"^(?P<authors>.+?)\s*\((?P<year>\d{4})\)\.\s*(?P<title>[^.]+)\.",
    re.MULTILINE,
)

#: Matches a four-digit year on its own, not the leading digits of a longer
#: number. The lazy ``source`` groups below rely on this to skip volume and
#: issue numbers -- ``vol. 4, no. 2, 2019`` must yield 2019, not 4.
_YEAR_GROUP = r"(?P<year>(?:19|20)\d{2})\b"

# IEEE: [1] A. Author, "Title," Journal, vol. X, no. Y, pp. Z, Year.
_IEEE_PATTERN = re.compile(
    r'^\s*\[\d+\]\s*(?P<authors>[^,]+),\s*"(?P<title>[^"]+),"'
    r"(?P<source>.*?)" + _YEAR_GROUP,
    re.MULTILINE,
)

# MLA: Author. "Title." Source, vol. X, no. Y, Year, pp. Z.
_MLA_PATTERN = re.compile(
    r'^(?P<authors>[^.]+)\.\s*"(?P<title>[^"]+)"(?P<source>.*?)' + _YEAR_GROUP,
    re.MULTILINE,
)

#: Tried in order. APA is first because its parenthesised year is the most
#: specific marker; the quoted-title formats would otherwise claim APA lines.
_FORMAT_PATTERNS = (
    ("APA", _APA_PATTERN),
    ("IEEE", _IEEE_PATTERN),
    ("MLA", _MLA_PATTERN),
)

_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

#: Punctuation that ends a field in a bibliography rather than belonging to it.
_TRAILING_PUNCTUATION = " .,;:\"'"

#: ``(format_name, authors, year, title, source)``.
ParsedCitation = tuple[str, str, str, str, str]


# ── Field helpers ─────────────────────────────────────────────────────────────


def _clean_field(value: str) -> str:
    """Strip whitespace and the punctuation bibliographies use as separators.

    MLA puts the closing period inside the quotes -- ``"A study."`` -- so the
    captured title carries punctuation that belongs to the citation format
    rather than to the work's name.
    """
    return value.strip().strip(_TRAILING_PUNCTUATION).strip()


def _clean_authors(value: str) -> str:
    """Trim an author field without touching the periods that belong to it.

    APA authors end in an initial (``Smith, J. A.``), so the general field
    cleanup would eat a character that is part of the name. Only the separator
    comma is removed here.
    """
    return value.strip().rstrip(",").strip()


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for hashing."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def generate_citation_hash(author: str, year: str, title: str) -> str:
    """Generate a deterministic SHA-256 hash for a citation, to enable matching.

    Hashes the normalized author, year and title so the same work cited by two
    documents yields the same identifier even when the formatting differs. Only
    the leading part of each field is used, which keeps minor tail differences
    (a subtitle, an extra initial) from splitting one work into two nodes.

    Args:
        author: Author field, as parsed.
        year: Four-digit year, or ``"Unknown"``.
        title: Work title, as parsed.

    Returns:
        A 64-character hexadecimal digest.
    """
    key = f"{_normalize_text(author)[:50]}|{year}|{_normalize_text(title)[:80]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _parse_strict(line: str) -> Optional[ParsedCitation]:
    """Try each known citation format against one bibliography line.

    Args:
        line: A single, already-stripped bibliography line.

    Returns:
        The parsed fields, or ``None`` when no format matched.
    """
    for format_name, pattern in _FORMAT_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue

        groups = match.groupdict()
        source = groups.get("source")
        if source is None:
            # APA exposes no source group: whatever follows the title is the
            # journal or publisher.
            source = line[match.end() :]

        return (
            format_name,
            _clean_authors(groups["authors"]),
            groups["year"].strip(),
            _clean_field(groups["title"]),
            _clean_field(source),
        )

    return None


def _parse_heuristic(line: str) -> Optional[ParsedCitation]:
    """Fall back to splitting a messy line around the first four-digit year.

    Real student bibliographies are rarely consistent enough to match a strict
    pattern. Treating the text before the year as the author and the text after
    it as the title keeps recall high; a line with no year at all is not a
    citation and is skipped.

    Args:
        line: A single, already-stripped bibliography line.

    Returns:
        The parsed fields, or ``None`` when the line carries no year.
    """
    year_match = _YEAR_PATTERN.search(line)
    if not year_match:
        return None

    before = line[: year_match.start()]
    after = line[year_match.end() :]

    return (
        "HEURISTIC",
        _clean_authors(before) or "Unknown",
        year_match.group(0),
        _clean_field(after) or line,
        "",
    )


# ── Public API ────────────────────────────────────────────────────────────────


def extract_citations(text: str) -> list[Citation]:
    """Parse the bibliography section of a document.

    Each non-empty line is matched against APA, IEEE and MLA in turn, then
    against the year-based fallback. Entries that hash identically within one
    document are collapsed, so a reference list that repeats a work does not
    inflate the citation-overlap score.

    Args:
        text: The raw text of the bibliography/references section.

    Returns:
        A list of :class:`Citation` objects, in the order they appear.
    """
    if not text or not isinstance(text, str):
        return []

    citations: list[Citation] = []
    seen_hashes: set[str] = set()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        parsed = _parse_strict(line) or _parse_heuristic(line)
        if parsed is None:
            continue

        format_name, authors, year, title, source = parsed

        citation = Citation(
            raw_text=line,
            authors=authors or "Unknown",
            year=year or "Unknown",
            title=title or line,
            source=source,
            format_detected=format_name,
        )

        if citation.citation_hash in seen_hashes:
            continue
        seen_hashes.add(citation.citation_hash)

        citations.append(citation)

    logger.info("Extracted %d unique citations from bibliography.", len(citations))
    return citations


def compute_jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute the Jaccard similarity between two sets of citation keys.

    Args:
        set_a: Citation keys from the first document.
        set_b: Citation keys from the second document.

    Returns:
        The size of the intersection over the size of the union. Two empty
        bibliographies are treated as identical and score 1.0.
    """
    if not set_a and not set_b:
        return 1.0

    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0

    return len(set_a.intersection(set_b)) / union
