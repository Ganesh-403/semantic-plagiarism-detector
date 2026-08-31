"""
src/core/layout_tree_extractor.py
---------------------------------
Document Structural Layout and Formatting Tree Extractor.

Parses document formatting (headings, bold, italic) into a normalized tree
structure to detect structural cloning, even when the text is heavily paraphrased.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LayoutNode:
    """Represents a node in the document layout tree."""

    tag: str  # e.g., 'H1', 'H2', 'B', 'I', 'P'
    children: List["LayoutNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tree to a dictionary representation."""
        return {
            "tag": self.tag,
            "children": [child.to_dict() for child in self.children],
        }


def parse_html_layout(html_content: str) -> LayoutNode:
    """Parse HTML content into a normalized layout tree.

    Extracts structural tags (H1-H6, B, I, P) and ignores content.
    This creates a pure structural fingerprint of the document.

    Args:
        html_content: Raw HTML string.

    Returns:
        Root LayoutNode representing the document structure.
    """
    root = LayoutNode(tag="ROOT")

    # Regex to find structural tags
    # We only care about the tag type, not the attributes or content
    tag_pattern = re.compile(r"<(h[1-6]|b|i|p|ul|ol|li)[^>]*>", re.IGNORECASE)

    # Stack to track current nesting
    stack = [root]

    for match in tag_pattern.finditer(html_content):
        tag = match.group(1).upper()

        # Simple heuristic: if it's a closing tag, pop the stack
        # (This is a simplified parser; real HTML parsing requires a proper parser)
        # For this implementation, we just build a flat-ish tree of structural markers
        node = LayoutNode(tag=tag)
        stack[-1].children.append(node)

        # Push block-level elements to stack (simplified)
        if tag in ["UL", "OL"]:
            stack.append(node)

    # In a real implementation, we would handle closing tags to pop the stack.
    # For structural fingerprinting, the sequence and nesting of block elements is key.

    logger.info("Extracted layout tree with %d root children.", len(root.children))
    return root


def parse_markdown_layout(md_content: str) -> LayoutNode:
    """Parse Markdown content into a normalized layout tree.

    Extracts headings (#, ##, ###) and bold/italic markers.
    """
    root = LayoutNode(tag="ROOT")
    lines = md_content.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("######"):
            root.children.append(LayoutNode(tag="H6"))
        elif line.startswith("#####"):
            root.children.append(LayoutNode(tag="H5"))
        elif line.startswith("####"):
            root.children.append(LayoutNode(tag="H4"))
        elif line.startswith("###"):
            root.children.append(LayoutNode(tag="H3"))
        elif line.startswith("##"):
            root.children.append(LayoutNode(tag="H2"))
        elif line.startswith("#"):
            root.children.append(LayoutNode(tag="H1"))
        elif line.startswith("- ") or line.startswith("* "):
            root.children.append(LayoutNode(tag="LI"))
        elif "**" in line or "__" in line:
            root.children.append(LayoutNode(tag="B"))
        elif "*" in line or "_" in line:
            root.children.append(LayoutNode(tag="I"))
        else:
            root.children.append(LayoutNode(tag="P"))

    return root
