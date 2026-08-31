"""
src/core/pseudocode_parser.py
-----------------------------
Pseudocode and Natural Language Algorithmic Step Parser.

Normalizes natural language algorithmic descriptions and pseudocode into
structured logical blocks (loops, conditionals, assignments) to enable
cross-modal alignment with source code implementations.
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LogicalBlock:
    """Represents a normalized logical block extracted from text or pseudocode."""

    block_type: str  # 'loop', 'conditional', 'assignment', 'io', 'control'
    content: str
    normalized_tokens: List[str] = field(default_factory=list)


# Regex patterns for common algorithmic constructs
LOOP_PATTERNS = [
    re.compile(r"\b(for|while|loop|iterate|repeat)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+each\b", re.IGNORECASE),
    re.compile(r"\bfor\s+\w+\s+in\b", re.IGNORECASE),
]

CONDITIONAL_PATTERNS = [
    re.compile(r"\b(if|else|else\s+if|elif|unless|when|switch|case)\b", re.IGNORECASE),
    re.compile(r"\bcheck\s+if\b", re.IGNORECASE),
    re.compile(r"\bverify\s+that\b", re.IGNORECASE),
]

ASSIGNMENT_PATTERNS = [
    re.compile(r"\b(set|assign|initialize|let|var|let\s+\w+\s*=)\b", re.IGNORECASE),
    re.compile(r"\b\w+\s*=\s*\w+", re.IGNORECASE),  # Simple assignment heuristic
]

IO_PATTERNS = [
    re.compile(r"\b(read|write|print|output|input|return|yield|log)\b", re.IGNORECASE)
]


def normalize_algorithmic_text(text: str) -> str:
    """Clean and normalize algorithmic text for parsing.

    Removes excessive whitespace, standardizes punctuation, and lowercases
    the text to simplify regex matching.
    """
    if not text:
        return ""
    # Remove markdown/code block formatting if present
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"```", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_logical_blocks(text: str) -> List[LogicalBlock]:
    """Parse natural language or pseudocode into structured logical blocks.

    Splits the text into sentences or lines and classifies each into
    a logical block type (loop, conditional, assignment, io, control).

    Args:
        text: The natural language or pseudocode text.

    Returns:
        List of LogicalBlock objects.
    """
    text = normalize_algorithmic_text(text)
    if not text:
        return []

    # Split by sentence terminators or newlines
    lines = re.split(r"(?<=[.!?])\s+|\n", text)
    blocks = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        block_type = "control"  # Default fallback

        # Check patterns in order of specificity
        if any(p.search(line) for p in LOOP_PATTERNS):
            block_type = "loop"
        elif any(p.search(line) for p in CONDITIONAL_PATTERNS):
            block_type = "conditional"
        elif any(p.search(line) for p in ASSIGNMENT_PATTERNS):
            block_type = "assignment"
        elif any(p.search(line) for p in IO_PATTERNS):
            block_type = "io"

        # Extract normalized tokens (alphanumeric only)
        tokens = re.findall(r"\b\w+\b", line.lower())

        blocks.append(
            LogicalBlock(block_type=block_type, content=line, normalized_tokens=tokens)
        )

    logger.info("Extracted %d logical blocks from text.", len(blocks))
    return blocks


def compute_block_signature(blocks: List[LogicalBlock]) -> List[str]:
    """Generate a structural signature for a sequence of logical blocks.

    Creates a sequence of block types (e.g., ['loop', 'conditional', 'assignment'])
    to represent the high-level algorithmic flow.

    Args:
        blocks: List of LogicalBlock objects.

    Returns:
        List of block type strings.
    """
    return [b.block_type for b in blocks]
