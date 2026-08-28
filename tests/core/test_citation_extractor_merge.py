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
tests/core/test_citation_extractor_merge.py
-------------------------------------------
Regression tests for the merged citation extractor (Issue #3565).

src/core/citation_extractor.py had been left holding two complete
implementations concatenated end to end. The first defined the ``Citation``
dataclass and an ``extract_citations() -> List[Citation]``; the second, further
down the same file, redefined ``extract_citations()`` to return
``List[Dict[str, str]]``. Python binds the last definition, so the dictionary
version won.

``Citation`` stayed importable -- it belongs to the first half -- so nothing
failed at import time. It failed at use: ``src.db.citation_graph_db``
``ingest_citations()`` is typed ``List[Citation]`` and calls
``cite.get_normalized_key()``, which raises AttributeError on a dict, and the
``except sqlite3.Error`` around that loop does not catch it.

These tests pin the merged contract from both sides: the object API the graph
layer needs, the mapping API the citation database reads, and the parsing
capability (IEEE, the fallback heuristic, deduplication) that only existed in
the second implementation.
"""

import ast
import inspect

import pytest

import src.core.citation_extractor as citation_extractor
from src.core.citation_extractor import (
    Citation,
    compute_jaccard_similarity,
    extract_citations,
    generate_citation_hash,
)
from src.db.citation_graph_db import (
    get_document_citation_keys,
    ingest_citations,
    initialize_citation_db,
)

APA_LINE = "Smith, J. (2020). The art of plagiarism. Journal of AI, 12(3), 45-60."
IEEE_LINE = '[1] J. Smith, "A study of drift," IEEE Transactions, vol. 4, 2019.'
MLA_LINE = 'Doe, Jane. "Detecting Ghostwriting." Tech Review, 2021.'
MESSY_LINE = "Anon, some untidy reference from 2018 with no clear format"


@pytest.fixture
def temp_db(tmp_path):
    """An isolated citation graph database."""
    db_path = tmp_path / "test_cite.db"
    initialize_citation_db(db_path)
    return db_path


class TestModuleHasOneImplementation:
    """The file must not drift back into holding two of everything."""

    def test_extract_citations_is_defined_once(self):
        """A second definition would silently shadow the first again."""
        tree = ast.parse(inspect.getsource(citation_extractor))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "extract_citations"
        ]

        assert len(definitions) == 1

    def test_module_has_no_orphaned_docstring(self):
        """The second module's docstring was a discarded expression statement.

        A bare string below the first statement is invisible to pydoc, Sphinx
        and the docstring coverage gate, and is the signature of exactly this
        kind of concatenation.
        """
        tree = ast.parse(inspect.getsource(citation_extractor))
        strays = [
            node
            for node in tree.body[1:]
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]

        assert not strays

    def test_module_docstring_survives(self):
        """One docstring, and it is the module's."""
        assert citation_extractor.__doc__
        assert "Citation Extraction Engine" in citation_extractor.__doc__


class TestReturnsCitationObjects:
    """The graph layer is typed against Citation, so that is what comes back."""

    def test_extract_returns_citation_instances(self):
        """The regression itself: this returned dicts."""
        citations = extract_citations(APA_LINE)

        assert len(citations) == 1
        assert isinstance(citations[0], Citation)

    def test_citation_exposes_the_graph_attributes(self):
        """authors/year/title/source are what ingest_citations() writes."""
        citation = extract_citations(APA_LINE)[0]

        assert citation.authors == "Smith, J."
        assert citation.year == "2020"
        assert "plagiarism" in citation.title.lower()
        assert citation.source

    def test_citation_has_a_normalized_key(self):
        """get_normalized_key() is the graph node identity."""
        citation = extract_citations(APA_LINE)[0]
        key = citation.get_normalized_key()

        assert "smith" in key
        assert "2020" in key

    def test_format_detected_is_populated(self):
        """The provenance label had no equivalent in the dict output."""
        assert extract_citations(APA_LINE)[0].format_detected == "APA"
        assert extract_citations(IEEE_LINE)[0].format_detected == "IEEE"
        assert extract_citations(MLA_LINE)[0].format_detected == "MLA"
        assert extract_citations(MESSY_LINE)[0].format_detected == "HEURISTIC"


class TestMappingAccessForTheDatabaseLayer:
    """src.db.citation_db reads citations with mapping syntax."""

    def test_hash_and_author_are_readable_by_key(self):
        """add_document_citations() reads cit["hash"] and cit["author"]."""
        citation = extract_citations(APA_LINE)[0]

        assert citation["author"] == citation.authors
        assert citation["hash"] == citation.citation_hash

    def test_every_column_the_database_writes_is_reachable(self):
        """The five keys add_document_citations() indexes must all resolve."""
        citation = extract_citations(APA_LINE)[0]

        for key in ("hash", "author", "year", "title", "raw_text"):
            assert citation[key] is not None

    def test_unknown_key_raises_key_error(self):
        """Mapping access should behave like a mapping, including on misses."""
        citation = extract_citations(APA_LINE)[0]

        with pytest.raises(KeyError):
            citation["not_a_field"]

    def test_get_returns_a_default_for_a_missing_key(self):
        """dict.get semantics, so callers can probe optional fields."""
        citation = extract_citations(APA_LINE)[0]

        assert citation.get("not_a_field", "fallback") == "fallback"
        assert citation.get("year") == "2020"

    def test_to_dict_carries_both_spellings(self):
        """A row can be written straight from to_dict()."""
        data = extract_citations(APA_LINE)[0].to_dict()

        assert data["author"] == data["authors"]
        assert data["hash"] == data["citation_hash"]


class TestFuzzyHashing:
    """The hash from the second implementation is preserved and reachable."""

    def test_hash_is_a_sha256_digest(self):
        """64 hex characters, as the database column expects."""
        citation = extract_citations(APA_LINE)[0]

        assert len(citation.citation_hash) == 64
        assert all(
            character in "0123456789abcdef" for character in citation.citation_hash
        )

    def test_hash_is_deterministic(self):
        """The same entry hashes the same way on every run."""
        first = extract_citations(APA_LINE)[0]
        second = extract_citations(APA_LINE)[0]

        assert first.citation_hash == second.citation_hash

    def test_hash_survives_formatting_differences(self):
        """Fuzzy matching is the point: punctuation and case must not matter."""
        assert generate_citation_hash(
            "Smith, J.", "2020", "The Art of Plagiarism"
        ) == generate_citation_hash("smith j", "2020", "the art of plagiarism!")

    def test_different_works_hash_differently(self):
        """Distinct citations must not collapse into one graph node."""
        assert generate_citation_hash(
            "Smith, J.", "2020", "The art of plagiarism"
        ) != generate_citation_hash("Doe, A.", "2020", "The art of plagiarism")

    def test_an_explicit_hash_is_not_overwritten(self):
        """Callers reconstructing a stored row keep the hash they were given."""
        citation = Citation(
            raw_text="raw",
            authors="Author",
            year="2023",
            title="Title",
            source="Source",
            format_detected="APA",
            citation_hash="deadbeef",
        )

        assert citation.citation_hash == "deadbeef"


class TestParsingBreadth:
    """IEEE, the fallback heuristic and deduplication came from the second half."""

    def test_ieee_bracketed_reference_is_parsed(self):
        """IEEE has no equivalent in the first implementation's patterns."""
        citation = extract_citations(IEEE_LINE)[0]

        assert citation.authors == "J. Smith"
        assert citation.year == "2019"
        assert citation.title == "A study of drift"

    def test_messy_line_falls_back_to_the_year_heuristic(self):
        """High recall on bibliographies that match no strict pattern."""
        citations = extract_citations(MESSY_LINE)

        assert len(citations) == 1
        assert citations[0].year == "2018"

    def test_line_without_a_year_is_skipped(self):
        """A line with no year is not a citation."""
        assert extract_citations("Just some prose with no reference in it") == []

    def test_mixed_bibliography_parses_every_format(self):
        """One reference list can hold all three formats at once."""
        citations = extract_citations("\n".join([APA_LINE, IEEE_LINE, MLA_LINE]))

        assert len(citations) == 3
        assert {c.format_detected for c in citations} == {"APA", "IEEE", "MLA"}

    def test_duplicate_entries_are_collapsed(self):
        """A repeated reference must not inflate the overlap score."""
        citations = extract_citations("\n".join([APA_LINE, APA_LINE, IEEE_LINE]))

        assert len(citations) == 2

    def test_blank_lines_are_ignored(self):
        """Bibliographies are full of blank separators."""
        citations = extract_citations(f"\n\n{APA_LINE}\n\n   \n{IEEE_LINE}\n")

        assert len(citations) == 2

    def test_titles_lose_the_bibliography_punctuation(self):
        """MLA puts the closing period inside the quotes."""
        citation = extract_citations('Smith, John. "A study." Journal, 2020.')[0]

        assert citation.title == "A study"

    def test_apa_authors_keep_their_trailing_initial(self):
        """'Smith, J. A.' ends in a period that belongs to the name."""
        citation = extract_citations("Smith, J. A. (2020). A study. Journal.")[0]

        assert citation.authors == "Smith, J. A."

    def test_empty_and_non_string_input_returns_empty(self):
        """The guard from the second implementation is preserved."""
        assert extract_citations("") == []
        assert extract_citations(None) == []


class TestGraphIngestionRoundTrip:
    """The end-to-end path that AttributeError was breaking."""

    def test_extracted_citations_ingest_without_raising(self, temp_db):
        """extract_citations() output feeds ingest_citations() directly."""
        citations = extract_citations("\n".join([APA_LINE, IEEE_LINE, MLA_LINE]))

        inserted = ingest_citations("doc_1", citations, db_path=temp_db)

        assert inserted == 3

    def test_ingested_keys_match_the_normalized_keys(self, temp_db):
        """What goes into the graph is what get_normalized_key() produced."""
        citations = extract_citations("\n".join([APA_LINE, IEEE_LINE]))
        ingest_citations("doc_1", citations, db_path=temp_db)

        stored = get_document_citation_keys("doc_1", db_path=temp_db)

        assert stored == {citation.get_normalized_key() for citation in citations}

    def test_shared_bibliography_scores_as_overlap(self, temp_db):
        """The feature this module exists for: two documents, one reference ring."""
        shared = "\n".join([APA_LINE, IEEE_LINE])
        ingest_citations("doc_a", extract_citations(shared), db_path=temp_db)
        ingest_citations(
            "doc_b",
            extract_citations(f"{shared}\n{MLA_LINE}"),
            db_path=temp_db,
        )

        keys_a = get_document_citation_keys("doc_a", db_path=temp_db)
        keys_b = get_document_citation_keys("doc_b", db_path=temp_db)

        assert compute_jaccard_similarity(keys_a, keys_b) == pytest.approx(2 / 3)
