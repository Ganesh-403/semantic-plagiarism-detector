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
tests/core/test_cfg_function_body_traversal_issue_3637.py
----------------------------------------------------------
Regression tests for Issue #3637.

``visit_FunctionDef()`` created one block per top-level statement in a function
body and then stopped: it never visited the statement, and never moved
``current_block_id`` onto the block it had just created. So ``visit_For``,
``visit_While``, ``visit_If`` and ``visit_Return`` were unreachable for any code
inside a function — the only place control flow actually lives.

The consequence was not cosmetic. ``compute_cfg_hash()`` is what CFG-isomorphism
clone detection compares, and these two hashed to structurally equivalent
graphs::

    def a():                  def b():
        for i in range(10):       print(0)
            print(i)

A loop and a straight line were indistinguishable, which defeats the module's
stated purpose. The tests below are organised around that: the graph must
separate different shapes, and must still collapse the same shape under
different identifiers.
"""

import pytest

from src.core.cfg_generator import (
    BasicBlock,
    cfg_to_adjacency_list,
    compute_cfg_hash,
    generate_cfg,
)
from src.core.graph_isomorphism_engine import compare_cfgs


def _markers(blocks: dict) -> list[str]:
    """Flatten every statement marker in the graph."""
    return [stmt for block in blocks.values() for stmt in block.statements]


def _edges(blocks: dict) -> set:
    """The graph's edge set, as ``(from, to)`` pairs."""
    return {
        (block_id, successor)
        for block_id, block in blocks.items()
        for successor in block.successors
    }


LOOP_IN_FUNCTION = """
def foo():
    for i in range(10):
        print(i)
"""

FLAT_FUNCTION = """
def bar():
    print(0)
"""


class TestFunctionBodiesAreWalked:
    """The core defect: nested statements never reached their visitors."""

    def test_loop_inside_a_function_emits_its_marker(self):
        blocks = generate_cfg(LOOP_IN_FUNCTION)

        assert "FOR" in _markers(blocks)

    def test_while_inside_a_function_emits_its_marker(self):
        code = "def foo():\n    while True:\n        pass\n"

        assert "WHILE" in _markers(generate_cfg(code))

    def test_branch_inside_a_function_emits_its_marker(self):
        code = "def foo():\n    if x:\n        pass\n"

        assert "IF" in _markers(generate_cfg(code))

    def test_return_inside_a_function_emits_its_marker(self):
        code = "def foo():\n    return 1\n"

        assert "RETURN" in _markers(generate_cfg(code))

    def test_loop_body_is_walked_too(self):
        """`print(i)` used to be absent from the graph entirely."""
        blocks = generate_cfg(LOOP_IN_FUNCTION)

        assert "EXPR" in _markers(blocks)

    def test_deeply_nested_statements_are_reached(self):
        code = (
            "def foo():\n"
            "    for i in y:\n"
            "        while True:\n"
            "            if x:\n"
            "                return 1\n"
        )

        markers = _markers(generate_cfg(code))

        assert {"FOR", "WHILE", "IF", "RETURN"} <= set(markers)

    def test_nested_function_definitions_are_walked(self):
        code = "def outer():\n    def inner():\n        return 1\n    return inner\n"

        markers = _markers(generate_cfg(code))

        assert markers.count("FUNC_DEF") == 2
        assert "RETURN" in markers

    def test_methods_inside_a_class_are_walked(self):
        code = (
            "class C:\n    def method(self):\n        for i in x:\n            pass\n"
        )

        markers = _markers(generate_cfg(code))

        assert "CLASS_DEF" in markers
        assert "FUNC_DEF" in markers
        assert "FOR" in markers


class TestNormalisedMarkers:
    """One vocabulary, whatever the nesting depth."""

    def test_loop_marker_is_normalised_not_the_ast_class_name(self):
        markers = _markers(generate_cfg(LOOP_IN_FUNCTION))

        assert "FOR" in markers
        assert "For" not in markers

    @pytest.mark.parametrize(
        "code, marker",
        [
            ("def f():\n    return 1\n", "RETURN"),
            ("def f():\n    raise ValueError()\n", "RAISE"),
            ("def f():\n    with open(p) as fh:\n        pass\n", "WITH"),
            ("def f():\n    try:\n        pass\n    except E:\n        pass\n", "TRY"),
            ("def f():\n    for i in x:\n        break\n", "BREAK"),
            ("def f():\n    for i in x:\n        continue\n", "CONTINUE"),
            ("x = 1\n", "ASSIGN"),
            ("import os\n", "IMPORT"),
            ("async def f():\n    return 1\n", "FUNC_DEF"),
        ],
    )
    def test_statement_markers(self, code, marker):
        assert marker in _markers(generate_cfg(code))

    def test_markers_are_upper_case(self):
        code = "def f():\n    x += 1\n    for i in y:\n        del x\n"

        for marker in _markers(generate_cfg(code)):
            assert marker == marker.upper()


class TestGraphConnectivity:
    """The entry block used to be an isolated node."""

    def test_entry_block_exists(self):
        blocks = generate_cfg(FLAT_FUNCTION)

        assert "ENTRY" in _markers(blocks)

    def test_entry_block_is_not_orphaned(self):
        blocks = generate_cfg(FLAT_FUNCTION)
        adjacency = cfg_to_adjacency_list(blocks)

        assert adjacency[1] != []

    def test_every_block_except_entry_is_reachable(self):
        code = (
            "def foo():\n"
            "    if x:\n"
            "        return 1\n"
            "    for i in y:\n"
            "        print(i)\n"
            "    return 0\n"
        )
        blocks = generate_cfg(code)

        reachable = {successor for _, successor in _edges(blocks)} | {1}

        assert reachable == set(blocks)

    def test_successors_reference_real_blocks(self):
        blocks = generate_cfg(LOOP_IN_FUNCTION)

        for block in blocks.values():
            for successor in block.successors:
                assert successor in blocks

    def test_function_body_hangs_off_the_definition_block(self):
        blocks = generate_cfg(FLAT_FUNCTION)
        adjacency = cfg_to_adjacency_list(blocks)

        # ENTRY -> FUNC_DEF -> body
        assert adjacency[1] == [2]
        assert adjacency[2] == [3]


class TestControlFlowEdges:
    """Loops and branches must produce the edges they imply."""

    def test_loop_has_a_back_edge_to_its_header(self):
        blocks = generate_cfg(LOOP_IN_FUNCTION)

        header = next(i for i, b in blocks.items() if "FOR" in b.statements)
        body = next(i for i, b in blocks.items() if "EXPR" in b.statements)

        assert (header, body) in _edges(blocks)
        assert (body, header) in _edges(blocks)

    def test_while_loop_has_a_back_edge(self):
        blocks = generate_cfg("def f():\n    while x:\n        y()\n")

        header = next(i for i, b in blocks.items() if "WHILE" in b.statements)

        assert any(target == header for _, target in _edges(blocks))

    def test_branch_fans_out_and_merges(self):
        blocks = generate_cfg(
            "def f():\n    if x:\n        a()\n    else:\n        b()\n"
        )

        header = next(i for i, b in blocks.items() if "IF" in b.statements)
        join = next(i for i, b in blocks.items() if "JOIN" in b.statements)
        successors = blocks[header].successors

        assert len(successors) == 2
        assert all((s, join) in _edges(blocks) for s in successors)

    def test_branch_without_else_still_merges(self):
        blocks = generate_cfg("def f():\n    if x:\n        a()\n    b()\n")

        header = next(i for i, b in blocks.items() if "IF" in b.statements)
        join = next(i for i, b in blocks.items() if "JOIN" in b.statements)

        assert (header, join) in _edges(blocks)

    def test_continue_jumps_back_to_the_enclosing_loop(self):
        blocks = generate_cfg("def f():\n    for i in x:\n        continue\n")

        header = next(i for i, b in blocks.items() if "FOR" in b.statements)
        cont = next(i for i, b in blocks.items() if "CONTINUE" in b.statements)

        assert (cont, header) in _edges(blocks)

    def test_continue_targets_the_innermost_loop(self):
        code = (
            "def f():\n"
            "    for i in x:\n"
            "        for j in y:\n"
            "            continue\n"
        )
        blocks = generate_cfg(code)

        headers = [i for i, b in blocks.items() if "FOR" in b.statements]
        inner = max(headers)
        cont = next(i for i, b in blocks.items() if "CONTINUE" in b.statements)

        assert (cont, inner) in _edges(blocks)

    def test_except_handler_is_reachable_from_the_try_block(self):
        code = "def f():\n    try:\n        a()\n    except E:\n        b()\n"
        blocks = generate_cfg(code)

        header = next(i for i, b in blocks.items() if "TRY" in b.statements)
        handler = next(i for i, b in blocks.items() if "EXCEPT" in b.statements)

        assert (header, handler) in _edges(blocks)

    def test_statement_after_a_loop_follows_the_header(self):
        code = "def f():\n    for i in x:\n        a()\n    b()\n"
        blocks = generate_cfg(code)

        header = next(i for i, b in blocks.items() if "FOR" in b.statements)

        # The header reaches both the loop body and whatever comes next.
        assert len(blocks[header].successors) == 2


class TestCloneDetection:
    """What the graph is for."""

    def test_a_loop_and_a_straight_line_are_not_clones(self):
        """The headline regression."""
        result = compare_cfgs(
            generate_cfg(LOOP_IN_FUNCTION), generate_cfg(FLAT_FUNCTION)
        )

        assert result["is_exact_clone"] is False
        assert result["structural_similarity"] < 1.0

    def test_renaming_identifiers_still_yields_a_clone(self):
        original = "def foo():\n    for i in range(10):\n        print(i)\n"
        renamed = "def bar():\n    for counter in range(10):\n        print(counter)\n"

        result = compare_cfgs(generate_cfg(original), generate_cfg(renamed))

        assert result["is_exact_clone"] is True
        assert result["structural_similarity"] == 1.0

    def test_changing_literals_still_yields_a_clone(self):
        original = "def f():\n    x = 1\n    return x\n"
        altered = "def f():\n    y = 99999\n    return y\n"

        assert compute_cfg_hash(generate_cfg(original)) == compute_cfg_hash(
            generate_cfg(altered)
        )

    def test_a_while_loop_is_not_a_for_loop(self):
        for_code = "def f():\n    for i in x:\n        a()\n"
        while_code = "def f():\n    while x:\n        a()\n"

        assert compute_cfg_hash(generate_cfg(for_code)) != compute_cfg_hash(
            generate_cfg(while_code)
        )

    def test_an_extra_branch_changes_the_hash(self):
        plain = "def f():\n    a()\n    return 1\n"
        branched = "def f():\n    if x:\n        a()\n    return 1\n"

        assert compute_cfg_hash(generate_cfg(plain)) != compute_cfg_hash(
            generate_cfg(branched)
        )

    def test_nesting_depth_changes_the_hash(self):
        shallow = (
            "def f():\n    for i in x:\n        a()\n    for j in y:\n        b()\n"
        )
        nested = "def f():\n    for i in x:\n        for j in y:\n            b()\n"

        assert compute_cfg_hash(generate_cfg(shallow)) != compute_cfg_hash(
            generate_cfg(nested)
        )


class TestDeterminism:
    """Hashes are only comparable if generation is reproducible."""

    def test_identical_source_produces_identical_hashes(self):
        code = "def foo():\n    if x:\n        return 1\n    return 0\n"

        assert compute_cfg_hash(generate_cfg(code)) == compute_cfg_hash(
            generate_cfg(code)
        )

    def test_repeated_generation_produces_identical_graphs(self):
        first = generate_cfg(LOOP_IN_FUNCTION)
        second = generate_cfg(LOOP_IN_FUNCTION)

        assert cfg_to_adjacency_list(first) == cfg_to_adjacency_list(second)
        assert _markers(first) == _markers(second)

    def test_block_ids_are_contiguous_from_one(self):
        blocks = generate_cfg(LOOP_IN_FUNCTION)

        assert sorted(blocks) == list(range(1, len(blocks) + 1))


class TestDegenerateInput:
    """Unchanged behaviour that must stay unchanged."""

    def test_syntax_error_returns_an_empty_graph(self):
        assert generate_cfg("def foo(:") == {}

    def test_empty_source_still_has_an_entry_block(self):
        blocks = generate_cfg("")

        assert _markers(blocks) == ["ENTRY"]

    def test_empty_graph_hashes_to_an_empty_string(self):
        assert compute_cfg_hash({}) == ""

    def test_comment_only_source_is_handled(self):
        blocks = generate_cfg("# nothing here\n")

        assert _markers(blocks) == ["ENTRY"]

    def test_module_level_statements_are_walked(self):
        blocks = generate_cfg("x = 1\nfor i in x:\n    print(i)\n")

        assert {"ASSIGN", "FOR", "EXPR"} <= set(_markers(blocks))

    def test_empty_block_signature(self):
        assert BasicBlock(1).get_signature() == "EMPTY"
