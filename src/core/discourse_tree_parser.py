"""
src/core/discourse_tree_parser.py
---------------------------------
Hierarchical Discourse Tree Parser.

Parses text into hierarchical rhetorical blocks (claims, evidence,
rebuttals, conclusions) using NLP heuristics to detect structural idea theft.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiscourseNode:
    """Represents a node in the discourse tree."""

    node_type: str  # 'CLAIM', 'EVIDENCE', 'REBUTTAL', 'CONCLUSION', 'INTRODUCTION'
    text: str
    children: List["DiscourseNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "children": [c.to_dict() for c in self.children],
        }


# Regex patterns for rhetorical markers
CLAIM_PATTERNS = [
    re.compile(
        r"\b(we argue|we claim|this paper shows|we propose|the thesis is)\b",
        re.IGNORECASE,
    )
]

EVIDENCE_PATTERNS = [
    re.compile(
        r"\b(for example|for instance|studies show|research indicates|data suggests|according to)\b",
        re.IGNORECASE,
    )
]

REBUTTAL_PATTERNS = [
    re.compile(
        r"\b(however|on the other hand|conversely|nevertheless|despite this|critics argue)\b",
        re.IGNORECASE,
    )
]

CONCLUSION_PATTERNS = [
    re.compile(
        r"\b(in conclusion|to summarize|ultimately|therefore|thus|in summary)\b",
        re.IGNORECASE,
    )
]

INTRODUCTION_PATTERNS = [
    re.compile(
        r"\b(in this paper|this study|the purpose of|background)\b", re.IGNORECASE
    )
]


def classify_paragraph(paragraph: str) -> str:
    """Classify a paragraph into a rhetorical block type.

    Args:
        paragraph: The paragraph text.

    Returns:
        The rhetorical node type string.
    """
    if any(p.search(paragraph) for p in CONCLUSION_PATTERNS):
        return "CONCLUSION"
    if any(p.search(paragraph) for p in REBUTTAL_PATTERNS):
        return "REBUTTAL"
    if any(p.search(paragraph) for p in EVIDENCE_PATTERNS):
        return "EVIDENCE"
    if any(p.search(paragraph) for p in INTRODUCTION_PATTERNS):
        return "INTRODUCTION"
    if any(p.search(paragraph) for p in CLAIM_PATTERNS):
        return "CLAIM"

    # Default to CLAIM if no specific marker is found
    return "CLAIM"


def parse_discourse_tree(text: str) -> DiscourseNode:
    """Parse text into a hierarchical discourse tree.

    Splits the text into paragraphs and classifies each into a rhetorical
    node, building a flat or shallow tree structure representing the
    argumentative flow.

    Args:
        text: The full document text.

    Returns:
        The root DiscourseNode representing the document structure.
    """
    if not text or not isinstance(text, str):
        return DiscourseNode(node_type="EMPTY", text="")

    # Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    root = DiscourseNode(node_type="DOCUMENT", text="")

    for para in paragraphs:
        node_type = classify_paragraph(para)
        child_node = DiscourseNode(node_type=node_type, text=para)
        root.children.append(child_node)

    logger.info("Parsed discourse tree with %d rhetorical blocks.", len(root.children))
    return root
