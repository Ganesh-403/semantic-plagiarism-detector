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

import ast
import hashlib
import json


class ASTHashingEngine(ast.NodeTransformer):
    """
    Traverses and normalizes Abstract Syntax Trees to eliminate identifier name variation profiles.
    Converts structural blocks into consistent, deterministic semantic tokens.
    """

    def __init__(self):
        self.structural_tokens = []
        self.variable_map = {}
        self.var_counter = 0

    def _get_normalized_name(self, original_name: str) -> str:
        """Maps varying identifier names to sequential structural tokens."""
        if original_name not in self.variable_map:
            self.var_counter += 1
            self.variable_map[original_name] = f"var_{self.var_counter}"
        return self.variable_map[original_name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Normalizes local variable lookup handles."""
        node.id = self._get_normalized_name(node.id)
        self.structural_tokens.append(f"Name:{node.id}")
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Normalizes incoming parameter variable declarations."""
        node.arg = self._get_normalized_name(node.arg)
        self.structural_tokens.append(f"Arg:{node.arg}")
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Logs structural block entries while neutralizing function names."""
        self.structural_tokens.append("FunctionDefStart")
        # Do not normalize to keep tracking crisp, but record node type abstraction
        normalized_body = self.generic_visit(node)
        self.structural_tokens.append("FunctionDefEnd")
        return normalized_body

    def generate_fingerprint(self, source_code: str) -> dict:
        """
        Compiles source strings into normalized token vectors and computes SHA-256 hashes.
        Gracefully handles syntax parsing crashes caused by corrupted submissions.
        """
        try:
            parsed_tree = ast.parse(source_code)
            self.structural_tokens = []
            self.variable_map = {}
            self.var_counter = 0

            # Execute modification traversal loops
            self.visit(parsed_tree)

            # Serialize token streams into deterministic signature strings
            token_string = ",".join(self.structural_tokens)
            ast_sha256 = hashlib.sha256(token_string.encode("utf-8")).hexdigest()

            return {
                "success": True,
                "ast_hash": ast_sha256,
                "tokens": self.structural_tokens,
                "error": None,
            }
        except SyntaxError as parse_error:
            return {
                "success": False,
                "ast_hash": "0" * 64,
                "tokens": [],
                "error": f"Syntax parsing exception at line {parse_error.lineno}: {parse_error.msg}",
            }

    @staticmethod
    def calculate_token_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
        """
        Calculates containment similarities between structural token streams.
        Returns a percentage value indicating structural match density.
        """
        if not tokens_a or not tokens_b:
            return 0.00

        set_a, set_b = set(tokens_a), set(tokens_b)
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        return round((intersection / union) * 100, 2) if union > 0 else 0.00
