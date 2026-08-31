"""
src/core/knowledge_graph_extractor.py
-------------------------------------
Knowledge Graph Extraction Engine for Conceptual Plagiarism Detection.

Extracts Subject-Predicate-Object (SPO) triples from text using lightweight
NLP heuristics and regex-based dependency parsing proxies. This allows the
system to build a conceptual graph of ideas and arguments, enabling the
detection of idea theft even when vocabulary and syntax are rewritten.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """Represents a Subject-Predicate-Object triple."""

    subject: str
    predicate: str
    object: str

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject.lower(), self.predicate.lower(), self.object.lower())


# Common verbs/relations to act as predicates in our lightweight extraction
RELATION_VERBS = {
    "is",
    "are",
    "was",
    "were",
    "causes",
    "leads",
    "results",
    "implies",
    "requires",
    "depends",
    "produces",
    "generates",
    "contains",
    "includes",
}


def extract_spo_triples(text: str) -> List[Triple]:
    """Extract Subject-Predicate-Object triples from text using regex heuristics.

    This is a lightweight proxy for full dependency parsing. It identifies
    sentences, splits them around relation verbs, and extracts the noun
    phrases before and after the verb as subject and object.

    Args:
        text: The input document text.

    Returns:
        List of extracted Triple objects.
    """
    if not text or not isinstance(text, str):
        return []

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    triples = []

    # Regex to find relation verbs
    verb_pattern = re.compile(r"\b(" + "|".join(RELATION_VERBS) + r")\b", re.IGNORECASE)

    for sentence in sentences:
        # Clean punctuation for splitting
        clean_sent = re.sub(r"[^\w\s]", "", sentence).strip()
        words = clean_sent.split()

        for match in verb_pattern.finditer(clean_sent):
            verb = match.group(0).lower()
            verb_idx = match.start()

            # Extract subject (words before verb)
            subject_words = clean_sent[:verb_idx].strip().split()
            subject = (
                " ".join(subject_words[-3:]) if subject_words else ""
            )  # Take last 3 words as subject proxy

            # Extract object (words after verb)
            object_text = clean_sent[match.end() :].strip()
            object_words = object_text.split()
            object_str = (
                " ".join(object_words[:3]) if object_words else ""
            )  # Take first 3 words as object proxy

            if subject and object_str:
                triples.append(
                    Triple(
                        subject=subject.strip(),
                        predicate=verb,
                        object=object_str.strip(),
                    )
                )

    logger.info("Extracted %d SPO triples from text.", len(triples))
    return triples


def build_knowledge_graph(triples: List[Triple]) -> Dict[str, Any]:
    """Build a conceptual knowledge graph from extracted triples.

    Args:
        triples: List of Triple objects.

    Returns:
        Dictionary representing the graph (nodes and edges).
    """
    nodes = set()
    edges = set()

    for t in triples:
        nodes.add(t.subject.lower())
        nodes.add(t.object.lower())
        edges.add(t.to_tuple())

    return {
        "nodes": list(nodes),
        "edges": list(edges),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
