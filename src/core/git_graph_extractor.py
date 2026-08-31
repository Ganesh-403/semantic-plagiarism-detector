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
    """Parse standard `git log --stat` output to extract commits and churn."""
    commits = []

    # Regex for commit header: commit <hash> \n Author: <name> \n Date: <date>
    commit_pattern = re.compile(
        r"commit\s+([a-f0-9]+)\nAuthor:\s+(.*?)\nDate:\s+(.*?)\n", re.MULTILINE
    )

    # Regex for stat lines: <file> | <changes>
    stat_pattern = re.compile(r"\|\s+(\d+)\s+([+-]*)")

    current_commit = None
    lines = log_content.split("\n")

    for line in lines:
        header_match = commit_pattern.match(line)
        if header_match:
            if current_commit:
                commits.append(current_commit)

            hash_val = header_match.group(1)
            author = header_match.group(2).strip()
            date_str = header_match.group(3).strip()

            # Parse date: "Thu Jan 1 12:00:00 2024 +0000"
            try:
                # Simplified parsing for timestamp and timezone
                parts = date_str.rsplit(" ", 1)
                tz_offset = parts[1] if len(parts) > 1 else "+0000"
                dt = datetime.strptime(parts[0].strip(), "%a %b %d %H:%M:%S %Y")
                timestamp = int(dt.timestamp())
            except ValueError:
                timestamp = 0
                tz_offset = "+0000"

            current_commit = GitCommit(
                commit_hash=hash_val,
                author=author,
                timestamp=timestamp,
                timezone_offset=tz_offset,
            )
            continue

        if current_commit:
            stat_match = stat_pattern.search(line)
            if stat_match:
                changes = int(stat_match.group(1))
                symbols = stat_match.group(2)
                adds = symbols.count("+")
                dels = symbols.count("-")

                # Proportional estimation based on total changes
                if adds + dels > 0:
                    current_commit.additions += int(changes * (adds / (adds + dels)))
                    current_commit.deletions += int(changes * (dels / (adds + dels)))

    if current_commit:
        commits.append(current_commit)

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
