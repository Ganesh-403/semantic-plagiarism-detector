"""
src/core/document_versioning.py
-------------------------------
Sequence alignment and diff engine for tracking document evolution.

Implements a simplified Myers diff algorithm optimized for large document
drafts. This allows the system to track the lineage of documents uploaded
by the same user and visualize textual evolution (additions, deletions,
and patchwriting) between draft versions.
"""

import logging
from typing import List, Tuple, Dict, Any, NamedTuple
from enum import Enum

logger = logging.getLogger(__name__)


class DiffOp(str, Enum):
    """Enumeration of diff operations."""
    EQUAL = "equal"
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


class DiffBlock(NamedTuple):
    """Represents a contiguous block of identical diff operations."""
    op: DiffOp
    start_v1: int
    end_v1: int
    start_v2: int
    end_v2: int
    text_v1: str
    text_v2: str


def tokenize_by_words(text: str) -> list[str]:
    """Tokenize text into words, preserving whitespace as distinct tokens.
    
    This ensures that changes in spacing are captured in the diff,
    which is important for detecting patchwriting and formatting changes.
    """
    import re
    # Split on word boundaries but keep the delimiters
    tokens = re.findall(r'\w+|\s+|[^\w\s]', text)
    return tokens


def compute_myers_diff(tokens_v1: list[str], tokens_v2: list[str]) -> list[tuple[DiffOp, str, str]]:
    """Compute the shortest edit script between two token lists using Myers' algorithm.
    
    This is a simplified implementation of the Myers diff algorithm that
    operates in O(ND) time, where N is the sum of the lengths and D is
    the size of the minimum edit script.
    
    Args:
        tokens_v1: Token list from the older version.
        tokens_v2: Token list from the newer version.
        
    Returns:
        A list of tuples: (Operation, token_from_v1, token_from_v2).
        For INSERT, token_from_v1 is empty. For DELETE, token_from_v2 is empty.
    """
    n = len(tokens_v1)
    m = len(tokens_v2)
    max_d = n + m
    
    # V array stores the furthest reaching x for a given diagonal k
    # We use a dictionary to handle negative indices easily
    v = {1: 0}
    trace = []
    
    # Forward pass to find the shortest edit script
    for d in range(max_d + 1):
        trace.append(v.copy())
        for k in range(-d, d + 1, 2):
            # Determine whether to go down (insert) or right (delete)
            if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
                x = v.get(k + 1, -1)
            else:
                x = v.get(k - 1, -1) + 1
                
            y = x - k
            
            # Extend along the diagonal (equal tokens)
            while x < n and y < m and tokens_v1[x] == tokens_v2[y]:
                x += 1
                y += 1
                
            v[k] = x
            
            # Check if we've reached the end
            if x >= n and y >= m:
                # Backtrack to build the edit script
                return _backtrack_myers(trace, tokens_v1, tokens_v2, d)
                
    # Fallback if max_d is reached (should not happen for valid inputs)
    return _backtrack_myers(trace, tokens_v1, tokens_v2, max_d)


def _backtrack_myers(
    trace: list[dict[int, int]], 
    tokens_v1: list[str], 
    tokens_v2: list[str],
    d: int
) -> list[tuple[DiffOp, str, str]]:
    """Backtrack through the Myers trace to build the edit script.

    ``trace[i]`` is the V-state *entering* forward step ``i``, so the state that
    belongs with the ``k == -i`` / ``k != i`` diagonal test is ``trace[i]`` —
    reading ``trace[i - 1]`` there reconstructs the path from the wrong
    edit-distance level and can walk ``x`` or ``y`` below zero.

    Each iteration walks the snake back to ``(prev_x, prev_y)``, records the one
    non-diagonal move that reached this diagonal, and then *lands* on
    ``(prev_x, prev_y)``. The leading snake is emitted after the loop, which is
    also the entire script when the inputs are identical and ``d == 0``.

    Args:
        trace: One V-state snapshot per forward step, oldest first.
        tokens_v1: Token list from the older version.
        tokens_v2: Token list from the newer version.
        d: The edit distance the forward pass settled on.

    Returns:
        The edit script in chronological order.
    """
    x = len(tokens_v1)
    y = len(tokens_v2)
    edits: list[tuple[DiffOp, str, str]] = []

    for i in range(min(d, len(trace) - 1), 0, -1):
        v = trace[i]
        k = x - y

        # Which neighbouring diagonal did this position come from? Mirrors the
        # choice the forward pass made when it filled v[k] at step i.
        if k == -i or (k != i and v.get(k - 1, 0) < v.get(k + 1, 0)):
            prev_k = k + 1
        else:
            prev_k = k - 1

        prev_x = v.get(prev_k, 0)
        prev_y = prev_x - prev_k

        # Unwind the diagonal (equal) run that led into this position.
        while x > prev_x and y > prev_y:
            x -= 1
            y -= 1
            edits.append((DiffOp.EQUAL, tokens_v1[x], tokens_v2[y]))

        # Then the single insert or delete that crossed onto this diagonal.
        if x == prev_x:
            y -= 1
            edits.append((DiffOp.INSERT, "", tokens_v2[y]))
        else:
            x -= 1
            edits.append((DiffOp.DELETE, tokens_v1[x], ""))

        # Land exactly on the predecessor before the next iteration reads k.
        x, y = prev_x, prev_y

    # The snake running back to the origin. When d == 0 the loop above never
    # ran and this is the whole script, which is what identical inputs need.
    while x > 0 and y > 0:
        x -= 1
        y -= 1
        edits.append((DiffOp.EQUAL, tokens_v1[x], tokens_v2[y]))

    # Anything still left on one side is a pure prefix insert or delete. Only
    # reachable via the max_d fallback, but cheap insurance against a truncated
    # trace producing a script that does not rebuild the inputs.
    while x > 0:
        x -= 1
        edits.append((DiffOp.DELETE, tokens_v1[x], ""))
    while y > 0:
        y -= 1
        edits.append((DiffOp.INSERT, "", tokens_v2[y]))

    # Reverse to get chronological order
    edits.reverse()
    return edits


def generate_diff_blocks(
    text_v1: str, 
    text_v2: str
) -> list[DiffBlock]:
    """Generate high-level diff blocks from two text strings.
    
    Groups consecutive identical operations into DiffBlock objects
    for easier visualization and analysis.
    
    Args:
        text_v1: The older version of the text.
        text_v2: The newer version of the text.
        
    Returns:
        A list of DiffBlock objects representing the changes.
    """
    tokens_v1 = tokenize_by_words(text_v1)
    tokens_v2 = tokenize_by_words(text_v2)
    
    edits = compute_myers_diff(tokens_v1, tokens_v2)
    
    blocks = []
    current_op = None
    start_v1, start_v2 = 0, 0
    current_text_v1, current_text_v2 = [], []
    
    idx_v1, idx_v2 = 0, 0
    
    for op, t1, t2 in edits:
        if op != current_op:
            if current_op is not None:
                blocks.append(DiffBlock(
                    op=current_op,
                    start_v1=start_v1,
                    end_v1=idx_v1,
                    start_v2=start_v2,
                    end_v2=idx_v2,
                    text_v1="".join(current_text_v1),
                    text_v2="".join(current_text_v2)
                ))
            current_op = op
            start_v1, start_v2 = idx_v1, idx_v2
            current_text_v1, current_text_v2 = [], []
            
        if t1:
            current_text_v1.append(t1)
            idx_v1 += len(t1)
        if t2:
            current_text_v2.append(t2)
            idx_v2 += len(t2)
            
    # Append the final block
    if current_op is not None:
        blocks.append(DiffBlock(
            op=current_op,
            start_v1=start_v1,
            end_v1=idx_v1,
            start_v2=start_v2,
            end_v2=idx_v2,
            text_v1="".join(current_text_v1),
            text_v2="".join(current_text_v2)
        ))
        
    return blocks


def calculate_retention_score(blocks: list[DiffBlock]) -> float:
    """Calculate the percentage of text retained from v1 to v2.

    Args:
        blocks: The list of DiffBlock objects.

    Returns:
        A float between 0.0 and 1.0 representing retention. A v1 with nothing
        in it — no blocks at all, or blocks that are all insertions — scores
        1.0: there was nothing to lose, which is not the same as having lost
        everything.
    """
    total_v1_len = sum(b.end_v1 - b.start_v1 for b in blocks if b.op != DiffOp.INSERT)
    equal_len = sum(b.end_v1 - b.start_v1 for b in blocks if b.op == DiffOp.EQUAL)

    if total_v1_len == 0:
        return 1.0

    return round(equal_len / total_v1_len, 4)

import difflib

class DocumentDiffEngine:
    @staticmethod
    def compute_word_diff(parent_text: str, child_text: str) -> list[dict]:
        """
        Executes a sequence alignment loop tracking word-by-word alterations.
        Returns a list of token dictionaries containing the token text and structural action state.
        """
        parent_words = parent_text.split()
        child_words = child_text.split()
        
        matcher = difflib.SequenceMatcher(None, parent_words, child_words)
        diff_tokens = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for word in parent_words[i1:i2]:
                    diff_tokens.append({"text": word, "action": "unchanged"})
            elif tag == 'delete':
                for word in parent_words[i1:i2]:
                    diff_tokens.append({"text": word, "action": "deleted"})
            elif tag == 'insert':
                for word in child_words[j1:j2]:
                    diff_tokens.append({"text": word, "action": "added"})
            elif tag == 'replace':
                for word in parent_words[i1:i2]:
                    diff_tokens.append({"text": word, "action": "deleted"})
                for word in child_words[j1:j2]:
                    diff_tokens.append({"text": word, "action": "added"})
                    
        return diff_tokens

    @staticmethod
    def calculate_retention_metrics(diff_tokens: list[dict]) -> dict:
        """Computes summary statistics regarding text evolution between versions."""
        total = len(diff_tokens)
        if total == 0:
            return {"retention_rate": 100.0, "addition_rate": 0.0, "deletion_rate": 0.0}
            
        unchanged = sum(1 for t in diff_tokens if t["action"] == "unchanged")
        added = sum(1 for t in diff_tokens if t["action"] == "added")
        deleted = sum(1 for t in diff_tokens if t["action"] == "deleted")
        
        return {
            "retention_rate": round((unchanged / (unchanged + deleted if (unchanged + deleted) > 0 else 1)) * 100, 2),
            "addition_rate": round((added / total) * 100, 2),
            "deletion_rate": round((deleted / total) * 100, 2)
        }
