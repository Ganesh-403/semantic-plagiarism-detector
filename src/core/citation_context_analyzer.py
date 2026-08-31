"""
src/core/citation_context_analyzer.py
-------------------------------------
Citation Context Extraction Engine.

Extracts the textual context surrounding citations (e.g., [1] or (Author, 2020))
to detect "citation bluffing" where a student cites a source but the surrounding
text does not actually reflect the cited paper's content.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Regex patterns for common citation formats
# Matches numeric citations like [1], [1, 2], [1-3]
NUMERIC_CITATION_PATTERN = re.compile(r"\[(\d+(?:[, \-\d]*)\d*)\]")

# Matches author-year citations like (Smith, 2020) or (Smith & Jones, 2020)
AUTHOR_YEAR_PATTERN = re.compile(
    r"\(([A-Za-z]+(?:\s(?:&|and)\s[A-Za-z]+)?,?\s*\d{4}[a-z]?)\)"
)

# Pattern to split text into sentences for context extraction
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def extract_citation_contexts(
    text: str, context_window: int = 1
) -> List[Dict[str, Any]]:
    """Extract the textual context surrounding each citation in the document.

    Args:
        text: The full document text.
        context_window: Number of sentences before and after the citation
                        to include in the context.

    Returns:
        List of dictionaries containing the citation ID, raw citation string,
        and the extracted context text.
    """
    if not text or not isinstance(text, str):
        return []

    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    contexts = []

    for i, sentence in enumerate(sentences):
        # Check for numeric citations
        for match in NUMERIC_CITATION_PATTERN.finditer(sentence):
            citation_id = match.group(1)
            raw_citation = match.group(0)

            # Extract context window
            start_idx = max(0, i - context_window)
            end_idx = min(len(sentences), i + context_window + 1)
            context_text = " ".join(sentences[start_idx:end_idx])

            contexts.append(
                {
                    "citation_id": citation_id,
                    "raw_citation": raw_citation,
                    "context_text": context_text,
                    "sentence_index": i,
                }
            )

        # Check for author-year citations
        for match in AUTHOR_YEAR_PATTERN.finditer(sentence):
            citation_id = match.group(1)
            raw_citation = match.group(0)

            start_idx = max(0, i - context_window)
            end_idx = min(len(sentences), i + context_window + 1)
            context_text = " ".join(sentences[start_idx:end_idx])

            contexts.append(
                {
                    "citation_id": citation_id,
                    "raw_citation": raw_citation,
                    "context_text": context_text,
                    "sentence_index": i,
                }
            )

    logger.info("Extracted %d citation contexts from document.", len(contexts))
    return contexts


def map_citations_to_references(
    contexts: List[Dict[str, Any]], references: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Map extracted citation contexts to their corresponding reference abstracts.

    Args:
        contexts: List of extracted citation contexts.
        references: Dictionary mapping citation IDs to reference abstracts/text.

    Returns:
        List of contexts enriched with the reference abstract.
    """
    mapped_contexts = []
    for ctx in contexts:
        cit_id = ctx["citation_id"]
        # Handle numeric ranges like "1-3" by just taking the first number for mapping
        clean_id = re.split(r"[,\-]", cit_id)[0].strip()

        abstract = references.get(clean_id, "")

        mapped_contexts.append({**ctx, "reference_abstract": abstract})

    return mapped_contexts
