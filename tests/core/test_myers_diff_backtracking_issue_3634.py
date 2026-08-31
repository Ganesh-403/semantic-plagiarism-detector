"""
tests/core/test_myers_diff_backtracking_issue_3634.py
------------------------------------------------------
Regression tests for Issue #3634.

``_backtrack_myers()`` walked the trace one edit-distance level below the one
its diagonal test assumed, never landed on ``(prev_x, prev_y)`` after recording
an edit, and skipped the ``d == 0`` snake entirely. The visible symptoms were:

* ``generate_diff_blocks("A B C D", "A B")`` raised ``IndexError``;
* identical drafts scored ``0.0`` retention instead of ``1.0``;
* a block labelled ``EQUAL`` could hold two texts that were not equal.

The load-bearing property here is the round-trip: taking the EQUAL and DELETE
tokens of an edit script must rebuild v1, and taking the EQUAL and INSERT tokens
must rebuild v2. A backtrack that wanders off the path breaks that invariant
long before it breaks anything a spot-check would notice, so most of this file
asserts against it directly.
"""

import random

import pytest

from src.core.document_versioning import (
    DiffOp,
    calculate_retention_score,
    compute_myers_diff,
    generate_diff_blocks,
    tokenize_by_words,
)


def _rebuild(edits):
    """Reconstruct both versions from an edit script."""
    v1 = [t1 for op, t1, _ in edits if op in (DiffOp.EQUAL, DiffOp.DELETE)]
    v2 = [t2 for op, _, t2 in edits if op in (DiffOp.EQUAL, DiffOp.INSERT)]
    return v1, v2


def _edit_count(edits):
    """Number of non-diagonal moves — the ``D`` in Myers' O(ND)."""
    return sum(1 for op, _, _ in edits if op != DiffOp.EQUAL)


class TestRoundTrip:
    """An edit script must rebuild the two inputs it was derived from."""

    @pytest.mark.parametrize(
        "v1, v2",
        [
            ([], []),
            ([], ["a"]),
            (["a"], []),
            (["a"], ["a"]),
            (["a"], ["b"]),
            (["a", "b", "c"], ["a", "b", "c"]),
            (["a", "b", "c"], ["a", "c"]),
            (["a", "c"], ["a", "b", "c"]),
            (["a", "b", "c", "d"], ["d", "c", "b", "a"]),
            (["x"] * 6, ["x"] * 3),
            (["x"] * 3, ["x"] * 6),
            (list("abcabba"), list("cbabac")),
        ],
    )
    def test_script_rebuilds_both_versions(self, v1, v2):
        rebuilt_v1, rebuilt_v2 = _rebuild(compute_myers_diff(v1, v2))

        assert rebuilt_v1 == v1
        assert rebuilt_v2 == v2

    def test_randomised_round_trip(self):
        """Fuzz the invariant; the old backtrack failed this within a few tries."""
        rng = random.Random(20260825)

        for _ in range(2000):
            v1 = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
            v2 = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]

            rebuilt_v1, rebuilt_v2 = _rebuild(compute_myers_diff(v1, v2))

            assert rebuilt_v1 == v1, (v1, v2)
            assert rebuilt_v2 == v2, (v1, v2)

    def test_randomised_run_does_not_raise(self):
        """Deleting a suffix used to raise IndexError out of the backtrack."""
        rng = random.Random(4242)

        for _ in range(500):
            v1 = [rng.choice("ab") for _ in range(rng.randint(1, 10))]
            v2 = v1[: rng.randint(0, len(v1))]

            compute_myers_diff(v1, v2)


class TestEditScriptShape:
    """The script has to be minimal and correctly labelled, not just valid."""

    def test_identical_input_is_all_equal(self):
        edits = compute_myers_diff(["A", " ", "B"], ["A", " ", "B"])

        assert edits
        assert all(op == DiffOp.EQUAL for op, _, _ in edits)
        assert _edit_count(edits) == 0

    def test_equal_entries_carry_matching_tokens(self):
        """An EQUAL op must never pair two different tokens."""
        edits = compute_myers_diff(list("the quick fox"), list("the slow fox"))

        for op, t1, t2 in edits:
            if op == DiffOp.EQUAL:
                assert t1 == t2

    def test_insert_leaves_v1_side_empty(self):
        edits = compute_myers_diff(["a"], ["a", "b"])
        inserts = [(t1, t2) for op, t1, t2 in edits if op == DiffOp.INSERT]

        assert inserts == [("", "b")]

    def test_delete_leaves_v2_side_empty(self):
        edits = compute_myers_diff(["a", "b"], ["a"])
        deletes = [(t1, t2) for op, t1, t2 in edits if op == DiffOp.DELETE]

        assert deletes == [("b", "")]

    def test_empty_to_text_is_all_inserts(self):
        edits = compute_myers_diff([], ["A", "B"])

        assert [op for op, _, _ in edits] == [DiffOp.INSERT, DiffOp.INSERT]

    def test_text_to_empty_is_all_deletes(self):
        edits = compute_myers_diff(["A", "B"], [])

        assert [op for op, _, _ in edits] == [DiffOp.DELETE, DiffOp.DELETE]

    def test_both_empty_is_an_empty_script(self):
        assert compute_myers_diff([], []) == []

    @pytest.mark.parametrize(
        "v1, v2, expected_distance",
        [
            (["a", "b", "c"], ["a", "b", "c"], 0),
            (["a", "b", "c"], ["a", "c"], 1),
            (["a", "c"], ["a", "b", "c"], 1),
            (["a", "b"], ["b", "a"], 2),
            (["a"], ["b"], 2),
        ],
    )
    def test_edit_distance_is_minimal(self, v1, v2, expected_distance):
        assert _edit_count(compute_myers_diff(v1, v2)) == expected_distance

    def test_common_prefix_is_not_re_edited(self):
        """A shared prefix must come back as EQUAL, not as delete+insert."""
        v1 = list("common prefix then A")
        v2 = list("common prefix then B")

        edits = compute_myers_diff(v1, v2)
        leading = [op for op, _, _ in edits[: len("common prefix then ")]]

        assert set(leading) == {DiffOp.EQUAL}


class TestDiffBlocks:
    """Block grouping over a correct edit script."""

    def test_word_substitution_produces_delete_and_insert(self):
        blocks = generate_diff_blocks("The quick brown fox.", "The slow brown fox.")
        ops = [b.op for b in blocks]

        assert ops == [DiffOp.EQUAL, DiffOp.DELETE, DiffOp.INSERT, DiffOp.EQUAL]

    def test_block_texts_reassemble_the_originals(self):
        text_v1 = "The quick brown fox jumps."
        text_v2 = "The lazy brown dog jumps."

        blocks = generate_diff_blocks(text_v1, text_v2)

        assert "".join(b.text_v1 for b in blocks) == text_v1
        assert "".join(b.text_v2 for b in blocks) == text_v2

    def test_equal_blocks_hold_identical_text(self):
        """The old engine emitted an EQUAL block whose two texts differed."""
        blocks = generate_diff_blocks("The quick brown fox.", "The slow brown fox.")

        for block in blocks:
            if block.op == DiffOp.EQUAL:
                assert block.text_v1 == block.text_v2

    def test_block_offsets_match_the_text_they_carry(self):
        blocks = generate_diff_blocks("A B C D", "A B")

        for block in blocks:
            assert block.end_v1 - block.start_v1 == len(block.text_v1)
            assert block.end_v2 - block.start_v2 == len(block.text_v2)

    def test_block_offsets_are_contiguous(self):
        blocks = generate_diff_blocks("one two three", "one three four")

        cursor_v1 = cursor_v2 = 0
        for block in blocks:
            assert block.start_v1 == cursor_v1
            assert block.start_v2 == cursor_v2
            cursor_v1, cursor_v2 = block.end_v1, block.end_v2

    def test_consecutive_operations_are_grouped(self):
        """Four deleted tokens are one DELETE block, not four."""
        blocks = generate_diff_blocks("A B C D", "A B")
        deletes = [b for b in blocks if b.op == DiffOp.DELETE]

        assert len(deletes) == 1
        assert deletes[0].text_v1 == " C D"

    def test_identical_text_is_a_single_equal_block(self):
        blocks = generate_diff_blocks("Hello world", "Hello world")

        assert [b.op for b in blocks] == [DiffOp.EQUAL]
        assert blocks[0].text_v1 == blocks[0].text_v2 == "Hello world"

    def test_empty_inputs_produce_no_blocks(self):
        assert generate_diff_blocks("", "") == []

    def test_deleting_the_tail_does_not_raise(self):
        """The original IndexError reproducer."""
        blocks = generate_diff_blocks("A B C D", "A B")

        assert blocks


class TestRetentionScore:
    """Retention is what the lineage UI reports; it has to mean something."""

    def test_identical_drafts_retain_everything(self):
        blocks = generate_diff_blocks("Hello world", "Hello world")

        assert calculate_retention_score(blocks) == 1.0

    def test_longer_identical_draft_still_scores_one(self):
        text = "The rain in Spain falls mainly on the plain."

        assert calculate_retention_score(generate_diff_blocks(text, text)) == 1.0

    def test_deleting_half_retains_about_half(self):
        score = calculate_retention_score(generate_diff_blocks("A B C D", "A B"))

        assert 0.4 <= score <= 0.6

    def test_full_rewrite_retains_nothing(self):
        score = calculate_retention_score(generate_diff_blocks("aaaa", "bbbb"))

        assert score == 0.0

    def test_pure_addition_retains_everything(self):
        """Appending to a draft does not remove anything from it."""
        score = calculate_retention_score(
            generate_diff_blocks("Hello world", "Hello world and more")
        )

        assert score == 1.0

    def test_empty_block_list_scores_one(self):
        """Nothing to compare is not the same as everything deleted."""
        assert calculate_retention_score([]) == 1.0

    def test_score_is_always_in_range(self):
        rng = random.Random(99)
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]

        for _ in range(200):
            v1 = " ".join(rng.choice(words) for _ in range(rng.randint(0, 8)))
            v2 = " ".join(rng.choice(words) for _ in range(rng.randint(0, 8)))

            score = calculate_retention_score(generate_diff_blocks(v1, v2))

            assert 0.0 <= score <= 1.0


class TestTokenization:
    """Unchanged, but the diff results above depend on it."""

    def test_whitespace_is_preserved_as_tokens(self):
        assert tokenize_by_words("A  B") == ["A", "  ", "B"]

    def test_tokens_reassemble_the_input(self):
        text = "Hello,   world! 42 times."

        assert "".join(tokenize_by_words(text)) == text
