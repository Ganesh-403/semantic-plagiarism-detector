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
src/core/code_ast_parser.py
---------------------------
Abstract Syntax Tree (AST) Parser and Normalizer for Code Plagiarism Detection.

Parses source code into normalized ASTs, stripping comments, docstrings,
and standardizing variable names. This allows the system to detect structural
code plagiarism regardless of superficial obfuscation like variable renaming,
whitespace changes, or comment injection.
"""

import ast
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ASTNormalizer(ast.NodeTransformer):
    """AST NodeTransformer that normalizes variable names and strips docstrings.

    This transformer traverses the AST and renames all local variables,
    function arguments, and imports to standardized names (e.g., var_1, var_2).
    This ensures that two structurally identical programs with different variable
    names produce the same normalized AST.
    """

    def __init__(self):
        self.var_map: dict[str, str] = {}
        self.var_counter = 0

    def _get_normalized_name(self, name: str) -> str:
        """Map an original variable name to a standardized name."""
        if name not in self.var_map:
            self.var_counter += 1
            self.var_map[name] = f"var_{self.var_counter}"
        return self.var_map[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize variable names (e.g., x -> var_1)."""
        node.id = self._get_normalized_name(node.id)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Normalize function argument names."""
        node.arg = self._get_normalized_name(node.arg)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize function names and strip docstrings."""
        node.name = self._get_normalized_name(node.name)

        # Strip docstrings (first expression if it's a string constant)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)

        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Normalize class names and strip docstrings."""
        node.name = self._get_normalized_name(node.name)

        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)

        self.generic_visit(node)
        return node

    def visit_Import(self, node: ast.Import) -> Optional[ast.AST]:
        """Strip import statements to focus on core logic structure."""
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        """Strip from-import statements."""
        return None


def parse_and_normalize_code(source_code: str) -> Optional[ast.AST]:
    """Parse Python source code and return a normalized AST.

    Args:
        source_code: The raw Python source code string.

    Returns:
        A normalized ast.AST object, or None if parsing fails.
    """
    try:
        # Parse the source code into an AST
        tree = ast.parse(source_code)

        # Apply the normalizer transformer
        normalizer = ASTNormalizer()
        normalized_tree = normalizer.visit(tree)

        # Fix missing locations and parent pointers
        ast.fix_missing_locations(normalized_tree)

        logger.info("Successfully parsed and normalized code AST.")
        return normalized_tree

    except SyntaxError as e:
        logger.error("Failed to parse source code: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error during AST normalization: %s", e)
        return None


def ast_to_node_sequence(tree: ast.AST) -> list[str]:
    """Convert an AST into a sequence of node type names.

    This creates a structural fingerprint of the code that can be compared
    using sequence alignment algorithms (like Levenshtein distance).

    Args:
        tree: The normalized AST.

    Returns:
        A list of strings representing the node types in traversal order.
    """
    sequence = []
    for node in ast.walk(tree):
        sequence.append(type(node).__name__)
    return sequence
