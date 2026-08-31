"""
src/core/docstring_extractor.py
-------------------------------
Code Comment and Docstring Extractor.

Parses source code to separate executable logic from comments and docstrings.
This allows the system to detect mismatches between code behavior and
comment descriptions, or copied documentation with rewritten code.
"""

import ast
import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """Represents a block of code and its associated comments."""

    code_text: str
    comment_text: str
    line_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_text": self.code_text,
            "comment_text": self.comment_text,
            "line_number": self.line_number,
        }


def extract_python_blocks(source_code: str) -> List[CodeBlock]:
    """Extract code blocks and their associated docstrings/comments from Python source.

    Uses the ast module to identify function/class definitions and their docstrings,
    and regex to extract inline comments.

    Args:
        source_code: The Python source code string.

    Returns:
        List of CodeBlock objects.
    """
    blocks = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.warning("Failed to parse Python code for comments: %s", e)
        return blocks

    # Extract docstrings from functions and classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node) or ""

            # Extract the code body (simplified: just get the source lines if possible)
            # Since ast doesn't give raw source easily, we'll use the node name as proxy
            code_text = (
                f"def {node.name}(...)"
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else f"class {node.name}"
            )

            blocks.append(
                CodeBlock(
                    code_text=code_text, comment_text=docstring, line_number=node.lineno
                )
            )

    # Extract inline comments using regex
    lines = source_code.split("\n")
    for i, line in enumerate(lines, 1):
        match = re.search(r"#(.*)$", line)
        if match:
            comment = match.group(1).strip()
            code_part = line[: match.start()].strip()
            if code_part or comment:
                blocks.append(
                    CodeBlock(code_text=code_part, comment_text=comment, line_number=i)
                )

    logger.info("Extracted %d code/comment blocks.", len(blocks))
    return blocks


def extract_generic_blocks(
    source_code: str, comment_prefix: str = "//"
) -> List[CodeBlock]:
    """Extract code blocks and comments for C-style languages (Java, C++, JS).

    Args:
        source_code: The source code string.
        comment_prefix: The single-line comment prefix (e.g., '//', '--').

    Returns:
        List of CodeBlock objects.
    """
    blocks = []
    lines = source_code.split("\n")

    # Simple state machine for multi-line comments /* ... */
    in_multiline = False
    multiline_comment = []
    multiline_start = 0

    for i, line in enumerate(lines, 1):
        if "/*" in line and "*/" in line:
            # Single line multi-line comment
            match = re.search(r"/\*(.*?)\*/", line)
            if match:
                blocks.append(
                    CodeBlock(
                        code_text=line[: match.start()].strip()
                        + line[match.end() :].strip(),
                        comment_text=match.group(1).strip(),
                        line_number=i,
                    )
                )
            continue

        if "/*" in line:
            in_multiline = True
            multiline_start = i
            multiline_comment = [line.split("/*", 1)[1]]
            continue

        if "*/" in line:
            in_multiline = False
            multiline_comment.append(line.split("*/", 1)[0])
            blocks.append(
                CodeBlock(
                    code_text=line.split("*/", 1)[1].strip(),
                    comment_text=" ".join(multiline_comment).strip(),
                    line_number=multiline_start,
                )
            )
            continue

        if in_multiline:
            multiline_comment.append(line)
            continue

        # Single line comments
        if comment_prefix in line:
            parts = line.split(comment_prefix, 1)
            blocks.append(
                CodeBlock(
                    code_text=parts[0].strip(),
                    comment_text=parts[1].strip(),
                    line_number=i,
                )
            )
        else:
            if line.strip():
                blocks.append(
                    CodeBlock(code_text=line.strip(), comment_text="", line_number=i)
                )

    return blocks
