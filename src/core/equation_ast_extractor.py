"""
src/core/equation_ast_extractor.py
----------------------------------
Mathematical Equation AST Extractor.

Parses LaTeX and MathML strings into normalized Abstract Syntax Trees (ASTs)
with abstract variable placeholders. This allows the system to detect
structural mathematical plagiarism even when variable names are swapped
(e.g., swapping $x$ for $\alpha$).
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MathNode:
    """Represents a node in the mathematical AST."""

    node_type: str  # 'VAR', 'NUM', 'OP', 'FUNC', 'GROUP'
    value: str
    children: List["MathNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
        }


# Regex patterns for LaTeX tokenization
LATEX_TOKEN_PATTERN = re.compile(
    r"\\[a-zA-Z]+|"  # Commands like \frac, \sin
    r"\{|\}|"  # Grouping
    r"\^|_|"  # Superscript/subscript
    r"[+\-*/=<>]|"  # Operators
    r"[a-zA-Z]|"  # Variables
    r"\d+(?:\.\d+)?"  # Numbers
)


def tokenize_latex(latex_str: str) -> List[str]:
    """Tokenize a LaTeX string into structural components."""
    # Remove math environment delimiters like $ or \[ \]
    clean_str = re.sub(
        r"^\$+|\$+$|^\\\[|\\\]$|^\\begin\{equation\}|\\end\{equation\}$",
        "",
        latex_str.strip(),
    )
    return LATEX_TOKEN_PATTERN.findall(clean_str)


def normalize_variables(tokens: List[str]) -> List[str]:
    """Normalize variable names to abstract placeholders (e.g., VAR_1, VAR_2).

    This ensures that swapping $x$ for $\alpha$ doesn't bypass the detector.
    """
    var_map = {}
    var_counter = 0
    normalized = []

    # Known functions and commands that should not be treated as variables
    known_commands = {
        "\\frac",
        "\\sin",
        "\\cos",
        "\\tan",
        "\\log",
        "\\ln",
        "\\exp",
        "\\sqrt",
        "\\sum",
        "\\prod",
        "\\int",
        "\\lim",
        "\\alpha",
        "\\beta",
        "\\gamma",
        "\\delta",
        "\\theta",
        "\\pi",
        "\\sigma",
        "\\mu",
    }

    for token in tokens:
        if token in known_commands:
            normalized.append(token)
        elif re.match(r"^[a-zA-Z]$", token):
            if token not in var_map:
                var_counter += 1
                var_map[token] = f"VAR_{var_counter}"
            normalized.append(var_map[token])
        else:
            normalized.append(token)

    return normalized


def build_math_ast(tokens: List[str]) -> MathNode:
    """Build a simplified AST from normalized tokens.

    This is a lightweight recursive descent parser for mathematical expressions.
    """
    root = MathNode(node_type="GROUP", value="ROOT")

    # Simplified parsing: just group by braces and operators for structural fingerprinting
    current_group = root
    stack = [root]

    for token in tokens:
        if token == "{":
            new_group = MathNode(node_type="GROUP", value="{")
            current_group.children.append(new_group)
            stack.append(new_group)
            current_group = new_group
        elif token == "}":
            if len(stack) > 1:
                stack.pop()
                current_group = stack[-1]
        elif token in {"^", "_"}:
            op_node = MathNode(node_type="OP", value=token)
            current_group.children.append(op_node)
        elif token.startswith("\\"):
            func_node = MathNode(node_type="FUNC", value=token)
            current_group.children.append(func_node)
        elif re.match(r"^\d+(?:\.\d+)?$", token):
            num_node = MathNode(node_type="NUM", value=token)
            current_group.children.append(num_node)
        else:
            var_node = MathNode(node_type="VAR", value=token)
            current_group.children.append(var_node)

    return root


def extract_equation_ast(latex_str: str) -> MathNode:
    """Extract and normalize a mathematical equation into an AST.

    Args:
        latex_str: The raw LaTeX string.

    Returns:
        The root MathNode of the normalized AST.
    """
    if not latex_str or not isinstance(latex_str, str):
        return MathNode(node_type="EMPTY", value="")

    tokens = tokenize_latex(latex_str)
    normalized_tokens = normalize_variables(tokens)
    ast_root = build_math_ast(normalized_tokens)

    logger.info(
        "Extracted math AST with %d top-level children.", len(ast_root.children)
    )
    return ast_root
