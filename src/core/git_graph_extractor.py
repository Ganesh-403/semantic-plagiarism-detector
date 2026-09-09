"""
src/core/git_graph_extractor.py
-------------------------------
Git Commit Graph and Forensics Extractor.

Parses exported Git commit logs to extract the commit Directed Acyclic Graph (DAG),
author timestamps, timezone entropy, and code churn metrics to detect covert collaboration.
"""

import re
import math
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GitCommit:
    """Represents a single Git commit."""

    commit_hash: str
    author: str
    timestamp: int
    timezone_offset: str
    additions: int = 0
    deletions: int = 0


@dataclass
class GitGraph:
    """Represents the commit DAG and forensics metrics."""

    commits: List[GitCommit] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"commit_count": len(self.commits), "edges": self.edges}


def parse_git_log(log_content: str) -> GitGraph:
    """Parse `git log --stat` into commits, churn, and reverse-log adjacency.

    Without parent hashes, edges describe log order rather than merge topology.
    """
    commits = []
    normalized = log_content.replace("\r\n", "\n")
    headers = list(re.finditer(
        r"^commit\s+([a-fA-F0-9]+)\s*\n(?:Merge:[^\n]*\n)?Author:\s*([^\n]+)\nDate:\s*([^\n]+)",
        normalized, re.MULTILINE,
    ))
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(normalized)
        body = normalized[header.end():end]
        date_text = header.group(3).strip()
        try:
            dt = datetime.strptime(date_text, "%a %b %d %H:%M:%S %Y %z")
        except ValueError:
            try:
                dt = datetime.fromisoformat(date_text)
                if dt.tzinfo is None:
                    raise ValueError("Missing timezone")
            except ValueError:
                logger.warning("Skipping commit with invalid date: %s", header.group(1))
                continue
        additions = deletions = 0
        for stat in re.finditer(r"\|\s+(\d+)\s+([+-]+)", body):
            changes = int(stat.group(1))
            symbols = stat.group(2)
            added = round(changes * symbols.count("+") / len(symbols))
            additions += added
            deletions += changes - added
        # The summary provides exact counts; graphical --stat bars are scaled.
        summary = re.search(r"^\s*\d+ files? changed[^\n]*", body, re.MULTILINE)
        if summary:
            added = re.search(r"(\d+) insertions?\(\+\)", summary.group(0))
            deleted = re.search(r"(\d+) deletions?\(-\)", summary.group(0))
            additions = int(added.group(1)) if added else 0
            deletions = int(deleted.group(1)) if deleted else 0
        commits.append(GitCommit(
            commit_hash=header.group(1), author=header.group(2).strip(),
            timestamp=int(dt.timestamp()), timezone_offset=dt.strftime("%z"),
            additions=additions, deletions=deletions,
        ))

    # Build simple linear DAG edges (parent -> child)
    edges = []
    for i in range(len(commits) - 1):
        edges.append((commits[i + 1].commit_hash, commits[i].commit_hash))

    return GitGraph(commits=commits, edges=edges)


def compute_timezone_entropy(graph: GitGraph) -> Dict[str, float]:
    """Compute Shannon entropy of timezone offsets and author distribution."""
    if not graph.commits:
        return {"tz_entropy": 0.0, "author_entropy": 0.0}

    tz_counts = Counter(c.timezone_offset for c in graph.commits)
    author_counts = Counter(c.author for c in graph.commits)

    def entropy(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum(
            (c / total) * math.log2(c / total) for c in counts.values() if c > 0
        )

    return {
        "tz_entropy": round(entropy(tz_counts), 4),
        "author_entropy": round(entropy(author_counts), 4),
    }
