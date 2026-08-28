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
tests/core/test_tag_manager_bulk_operations.py
----------------------------------------------
Regression tests for applying and removing tags in bulk.

``parse_tags()`` returns the *storage* form of a tag set — one comma-joined
string that may name several tags. ``apply_tag()`` and ``remove_tag()`` treated
that string as if it were a single tag: they compared it against the individual
entries of a document's tag list and appended it whole. Tagging a selection
with "hw1, final" therefore wrote a single list entry reading ``#final,#hw1``.

Nothing matched that entry afterwards, and the next write split it back apart at
the commas, so the corruption was invisible until a filter silently returned no
documents. These tests cover the multi-tag paths on both operations, plus the
single-tag behaviour that has to keep working unchanged.
"""

import pytest

from src.core.tag_manager import TagManager


@pytest.fixture
def tag_store(mocker):
    """An in-memory stand-in for the tags column, keyed by document id."""

    class TagStore:
        def __init__(self):
            self.tags = {}
            self.writes = []

        def get(self, doc_id):
            return self.tags.get(doc_id, "")

        def update(self, doc_id, new_tags):
            self.tags[doc_id] = new_tags
            self.writes.append((doc_id, new_tags))

        def entries_for(self, doc_id):
            """The document's tags as the list the column really represents."""
            raw = self.tags.get(doc_id, "")
            return [part for part in raw.split(",") if part]

    store = TagStore()
    mocker.patch("src.db.corpus_db.get_document_tags", side_effect=store.get)
    mocker.patch("src.db.corpus_db.update_document_tags", side_effect=store.update)
    return store


class TestNormalizeTags:
    """The list form is the canonical output; the string form joins it."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("hw1, final", ["#final", "#hw1"]),
            ("#hw1, FINAL,   #draft", ["#draft", "#final", "#hw1"]),
            ("#hw1", ["#hw1"]),
            ("hw1 hw1 HW1", ["#hw1"]),
            ("", []),
            (None, []),
            ("   ", []),
            ("#", []),
            ("#123", []),
        ],
    )
    def test_normalize_tags_returns_individual_tags(self, raw, expected):
        assert TagManager.normalize_tags(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["hw1, final", "#hw1", "", "#123, #hw1", "a, b, c"],
    )
    def test_parse_tags_is_the_joined_form(self, raw):
        """The two functions must never disagree about the same input."""
        assert TagManager.parse_tags(raw) == ",".join(TagManager.normalize_tags(raw))

    def test_interior_hashes_are_stripped(self):
        """A hash is only meaningful as a prefix."""
        assert TagManager.normalize_tags("hw1#final") == ["#hw1final"]
        assert "#" not in TagManager.normalize_tags("hw1#final")[0][1:]


class TestApplyMultipleTags:
    """Applying several tags at once must add them as separate entries."""

    def test_each_tag_becomes_its_own_entry(self, tag_store):
        TagManager.apply_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.entries_for("doc1.pdf") == ["#final", "#hw1"]

    def test_no_entry_contains_a_comma(self, tag_store):
        """A list entry holding a comma is the corruption being fixed."""
        TagManager.apply_tag(["doc1.pdf"], "hw1, final, draft")

        for entry in tag_store.entries_for("doc1.pdf"):
            assert "," not in entry

    def test_applied_tags_are_individually_matchable(self, tag_store):
        TagManager.apply_tag(["doc1.pdf"], "hw1, final")
        stored = tag_store.get("doc1.pdf")

        assert TagManager.has_matching_tag(stored, "#hw1") is True
        assert TagManager.has_matching_tag(stored, "#final") is True

    def test_merges_with_existing_tags(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#draft"

        TagManager.apply_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.entries_for("doc1.pdf") == ["#draft", "#final", "#hw1"]

    def test_partially_present_tags_are_topped_up(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#hw1"

        TagManager.apply_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.entries_for("doc1.pdf") == ["#final", "#hw1"]

    def test_fully_present_tags_skip_the_write(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#final,#hw1"

        TagManager.apply_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.writes == []

    def test_applies_across_every_document(self, tag_store):
        TagManager.apply_tag(["a.pdf", "b.pdf", "c.pdf"], "hw1, final")

        for doc_id in ("a.pdf", "b.pdf", "c.pdf"):
            assert tag_store.entries_for(doc_id) == ["#final", "#hw1"]

    def test_invalid_input_writes_nothing(self, tag_store):
        TagManager.apply_tag(["doc1.pdf"], "   ")
        TagManager.apply_tag(["doc1.pdf"], "#123")
        TagManager.apply_tag(["doc1.pdf"], "")

        assert tag_store.writes == []


class TestRemoveMultipleTags:
    """Removing several tags at once must remove each of them."""

    def test_removes_every_named_tag(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#draft,#final,#hw1"

        TagManager.remove_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.entries_for("doc1.pdf") == ["#draft"]

    def test_removes_the_subset_that_is_present(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#draft,#hw1"

        TagManager.remove_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.entries_for("doc1.pdf") == ["#draft"]

    def test_removing_the_last_tag_clears_the_column(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#final,#hw1"

        TagManager.remove_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.get("doc1.pdf") == ""

    def test_document_without_any_named_tag_is_untouched(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#draft"

        TagManager.remove_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.writes == []
        assert tag_store.entries_for("doc1.pdf") == ["#draft"]

    def test_untagged_document_is_untouched(self, tag_store):
        TagManager.remove_tag(["doc1.pdf"], "hw1")

        assert tag_store.writes == []

    def test_removes_across_every_document(self, tag_store):
        tag_store.tags = {
            "a.pdf": "#final,#hw1",
            "b.pdf": "#draft,#hw1",
            "c.pdf": "#draft",
        }

        TagManager.remove_tag(["a.pdf", "b.pdf", "c.pdf"], "hw1, final")

        assert tag_store.entries_for("a.pdf") == []
        assert tag_store.entries_for("b.pdf") == ["#draft"]
        assert tag_store.entries_for("c.pdf") == ["#draft"]

    def test_invalid_input_writes_nothing(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#hw1"

        TagManager.remove_tag(["doc1.pdf"], "  ")
        TagManager.remove_tag(["doc1.pdf"], "#123")

        assert tag_store.writes == []


class TestApplyRemoveRoundTrip:
    """Applying then removing the same set must return to the start."""

    def test_multi_tag_round_trip(self, tag_store):
        tag_store.tags["doc1.pdf"] = "#draft"

        TagManager.apply_tag(["doc1.pdf"], "hw1, final")
        TagManager.remove_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.entries_for("doc1.pdf") == ["#draft"]

    def test_repeated_apply_is_stable(self, tag_store):
        TagManager.apply_tag(["doc1.pdf"], "hw1, final")
        first = tag_store.get("doc1.pdf")

        TagManager.apply_tag(["doc1.pdf"], "hw1, final")

        assert tag_store.get("doc1.pdf") == first
        assert len(tag_store.writes) == 1

    def test_single_tag_behaviour_is_unchanged(self, tag_store):
        """The original single-tag path must keep working exactly as before."""
        tag_store.tags["doc1.pdf"] = "#oldtag"

        TagManager.apply_tag(["doc1.pdf"], "#newtag")
        assert tag_store.get("doc1.pdf") == "#newtag,#oldtag"

        TagManager.remove_tag(["doc1.pdf"], "#oldtag")
        assert tag_store.get("doc1.pdf") == "#newtag"
