"""
src/core/cfg_generator.py
-------------------------
Control Flow Graph (CFG) Generator for Source Code Plagiarism Detection.

Parses Python source code into normalized Control Flow Graphs where nodes
represent basic blocks (sequences of statements without branches) and edges
represent control flow (jumps, branches, loops). This allows detection of
algorithmic cloning even when variable names or syntax are obfuscated.

Every statement becomes a block carrying an UPPER_SNAKE_CASE marker for its
type, and compound statements add the edges their control flow implies: a loop
gets a back edge to its header, a branch fans out to its arms and merges at a
join block, ``continue`` jumps back to the enclosing loop header. Because the
markers are derived from statement types rather than source text, two functions
with the same shape hash identically however their identifiers are spelled.
"""

import ast
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Set, Optional

logger = logging.getLogger(__name__)


class BasicBlock:
    """Represents a basic block in the Control Flow Graph."""

    def __init__(self, block_id: int):
        self.id = block_id
        self.statements: List[str] = []  # Normalized statement types
        self.successors: List[int] = []  # IDs of successor blocks

    def add_statement(self, stmt_type: str) -> None:
        """Add a normalized statement type to the block."""
        self.statements.append(stmt_type)

    def add_successor(self, block_id: int) -> None:
        """Add a successor block ID."""
        if block_id not in self.successors:
            self.successors.append(block_id)

    def get_signature(self) -> str:
        """Generate a structural signature for the block."""
        return "_".join(self.statements) if self.statements else "EMPTY"


# Normalised markers for the statements that carry control flow. Everything
# else falls back to _marker(), which upper-snake-cases the AST class name, so
# the vocabulary is uniform wherever a statement appears.
_STATEMENT_MARKERS: dict[type, str] = {
    ast.FunctionDef: "FUNC_DEF",
    ast.AsyncFunctionDef: "FUNC_DEF",
    ast.ClassDef: "CLASS_DEF",
    ast.For: "FOR",
    ast.AsyncFor: "FOR",
    ast.While: "WHILE",
    ast.If: "IF",
    ast.Try: "TRY",
    ast.With: "WITH",
    ast.AsyncWith: "WITH",
    ast.Return: "RETURN",
    ast.Break: "BREAK",
    ast.Continue: "CONTINUE",
    ast.Raise: "RAISE",
}


def _marker(node: ast.AST) -> str:
    """Normalise an AST node type into an UPPER_SNAKE_CASE marker."""
    explicit = _STATEMENT_MARKERS.get(type(node))
    if explicit is not None:
        return explicit

    name = type(node).__name__
    out: List[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0 and not name[index - 1].isupper():
            out.append("_")
        out.append(char.upper())
    return "".join(out)


class CFGGenerator(ast.NodeVisitor):
    """AST Visitor that generates a Control Flow Graph from Python source code.

    Each statement becomes its own basic block, and compound statements wire up
    the edges their control flow implies: a loop gets a back edge to its header,
    a branch fans out to its arms and merges at a join block, ``continue`` jumps
    back to the enclosing loop header.

    ``visit()`` is only ever dispatched on statements, from :meth:`visit_body`.
    Expressions are deliberately not descended into — a CFG describes the shape
    of the control flow, and folding every sub-expression into a block signature
    would drown that shape in noise.
    """

    def __init__(self):
        self.blocks: Dict[int, BasicBlock] = {}
        self.current_block_id = 0
        self.block_counter = 0
        # Enclosing loop headers, innermost last, for `continue` back edges.
        self._loop_headers: List[int] = []

    def _new_block(self) -> int:
        """Create a new basic block and return its ID."""
        self.block_counter += 1
        self.blocks[self.block_counter] = BasicBlock(self.block_counter)
        return self.block_counter

    def _link(self, from_id: int, to_id: int) -> None:
        """Add an edge between two blocks."""
        if from_id in self.blocks and to_id in self.blocks:
            self.blocks[from_id].add_successor(to_id)

    def _emit(self, node: ast.AST) -> None:
        """Record a statement's normalised marker on the current block."""
        if self.current_block_id in self.blocks:
            self.blocks[self.current_block_id].add_statement(_marker(node))

    def _new_labelled_block(self, label: str) -> int:
        """Create a block carrying a single synthetic marker."""
        block_id = self._new_block()
        self.blocks[block_id].add_statement(label)
        return block_id

    def visit_body(self, body: List[ast.stmt], entry_id: int) -> int:
        """Chain a statement list into consecutive blocks.

        Args:
            body: The statements to walk.
            entry_id: The block control flows from into the first statement.

        Returns:
            The ID of the block control flows out of. For a compound statement
            that is its own exit point (a loop header, a branch join), not
            necessarily the last block created.
        """
        previous = entry_id
        for stmt in body:
            block_id = self._new_block()
            self._link(previous, block_id)
            self.current_block_id = block_id
            self.visit(stmt)
            # A compound statement moves current_block_id to its exit.
            previous = self.current_block_id
        return previous

    # -- Definitions ------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and walk their bodies.

        The body is walked with the definition's own block as the entry, so the
        specialised visitors below run for nested code. Control flow then
        resumes from the definition block: statements after a `def` are reached
        by falling past it, not by running through it.
        """
        self._emit(node)
        definition_block = self.current_block_id

        self.visit_body(node.body, definition_block)

        self.current_block_id = definition_block

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions and walk their bodies."""
        self._emit(node)
        definition_block = self.current_block_id

        self.visit_body(node.body, definition_block)

        self.current_block_id = definition_block

    # -- Loops ------------------------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        """Visit For loops, wiring the body's back edge to the header."""
        self._emit(node)
        header = self.current_block_id

        self._loop_headers.append(header)
        try:
            body_exit = self.visit_body(node.body, header)
        finally:
            self._loop_headers.pop()
        self._link(body_exit, header)

        # The loop exits through its header, or through an else clause.
        self.current_block_id = (
            self.visit_body(node.orelse, header) if node.orelse else header
        )

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        """Visit While loops, wiring the body's back edge to the header."""
        self._emit(node)
        header = self.current_block_id

        self._loop_headers.append(header)
        try:
            body_exit = self.visit_body(node.body, header)
        finally:
            self._loop_headers.pop()
        self._link(body_exit, header)

        self.current_block_id = (
            self.visit_body(node.orelse, header) if node.orelse else header
        )

    def visit_Continue(self, node: ast.Continue) -> None:
        """Visit continue statements and jump back to the loop header."""
        self._emit(node)
        if self._loop_headers:
            self._link(self.current_block_id, self._loop_headers[-1])

    # -- Branches ---------------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        """Visit If statements, splitting into arms and merging at a join."""
        self._emit(node)
        header = self.current_block_id

        then_exit = self.visit_body(node.body, header)
        # No else clause means the false arm falls straight through the header.
        else_exit = self.visit_body(node.orelse, header) if node.orelse else header

        join = self._new_labelled_block("JOIN")
        self._link(then_exit, join)
        self._link(else_exit, join)

        self.current_block_id = join

    def visit_Try(self, node: ast.Try) -> None:
        """Visit Try statements: body, handlers, else and finally."""
        self._emit(node)
        header = self.current_block_id

        exits = [self.visit_body(node.body, header)]

        for handler in node.handlers:
            handler_block = self._new_labelled_block("EXCEPT")
            # A handler is reachable from anywhere in the protected body.
            self._link(header, handler_block)
            exits.append(self.visit_body(handler.body, handler_block))

        if node.orelse:
            exits.append(self.visit_body(node.orelse, exits[0]))

        join = self._new_labelled_block("JOIN")
        for exit_id in exits:
            self._link(exit_id, join)

        # finally runs on every path out of the statement.
        self.current_block_id = (
            self.visit_body(node.finalbody, join) if node.finalbody else join
        )

    def visit_With(self, node: ast.With) -> None:
        """Visit With statements and walk the managed body."""
        self._emit(node)
        header = self.current_block_id

        self.current_block_id = self.visit_body(node.body, header)

    visit_AsyncWith = visit_With

    # -- Everything else --------------------------------------------------

    def generic_visit(self, node: ast.AST) -> None:
        """Record a simple statement and stop.

        Deliberately does not descend: visit() is dispatched on statements only,
        and a statement with no dedicated visitor above has no control flow of
        its own to model.
        """
        self._emit(node)


def generate_cfg(source_code: str) -> Dict[int, BasicBlock]:
    """Generate a Control Flow Graph from Python source code.

    Args:
        source_code: Python source code string.

    Returns:
        Dictionary mapping block IDs to BasicBlock objects.
    """
    try:
        tree = ast.parse(source_code)
        generator = CFGGenerator()

        # Create initial entry block
        entry_block = generator._new_block()
        generator.blocks[entry_block].add_statement("ENTRY")
        generator.current_block_id = entry_block

        # Walk the module body from the entry block so it is wired into the
        # graph rather than left as an isolated node.
        generator.visit_body(tree.body, entry_block)

        logger.info("Generated CFG with %d basic blocks.", len(generator.blocks))
        return generator.blocks

    except SyntaxError as e:
        logger.error("Failed to parse source code for CFG: %s", e)
        return {}
    except Exception as e:
        logger.error("Unexpected error generating CFG: %s", e)
        return {}


def cfg_to_adjacency_list(blocks: Dict[int, BasicBlock]) -> Dict[int, List[int]]:
    """Convert CFG blocks to a simple adjacency list representation."""
    adj_list = {}
    for block_id, block in blocks.items():
        adj_list[block_id] = block.successors
    return adj_list


def compute_cfg_hash(blocks: Dict[int, BasicBlock]) -> str:
    """Compute a structural hash of the CFG.

    Generates a deterministic hash based on block signatures and edge structure,
    ignoring variable names and specific literal values.
    """
    if not blocks:
        return ""

    # Build a canonical string representation of the CFG
    # Sort blocks by ID to ensure determinism
    canonical_parts = []
    for block_id in sorted(blocks.keys()):
        block = blocks[block_id]
        sig = block.get_signature()
        successors = sorted(block.successors)
        canonical_parts.append(f"B{block_id}({sig})->{successors}")

    canonical_str = "|".join(canonical_parts)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
