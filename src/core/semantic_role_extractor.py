"""
src/core/semantic_role_extractor.py
-----------------------------------
Semantic Role Labeling (SRL) Engine.

Extracts Agent-Action-Patient semantic role triples from text using
lightweight NLP heuristics and regex-based dependency parsing proxies.
This allows the system to detect deep semantic paraphrasing where
students transform sentences (e.g., active to passive voice) while
preserving the exact semantic roles.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class SemanticRole:
    """Represents a semantic role within a sentence."""

    role_type: str  # 'AGENT', 'ACTION', 'PATIENT', 'ADJUNCT'
    text: str
    normalized_tokens: List[str] = field(default_factory=list)


@dataclass
class SemanticTriple:
    """Represents an Agent-Action-Patient triple."""

    agent: Optional[SemanticRole] = None
    action: Optional[SemanticRole] = None
    patient: Optional[SemanticRole] = None
    raw_sentence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent.text if self.agent else None,
            "action": self.action.text if self.action else None,
            "patient": self.patient.text if self.patient else None,
            "raw_sentence": self.raw_sentence,
        }


# Regex patterns for identifying passive voice and active voice structures
PASSIVE_VOICE_PATTERN = re.compile(
    r"\b(\w+(?:\s+\w+)*)\s+(?:is|are|was|were|be|been|being)\s+(\w+ed|\w+en|\w+t)\s+(?:by\s+)(\w+(?:\s+\w+)*)",
    re.IGNORECASE,
)

ACTIVE_VOICE_PATTERN = re.compile(
    r"\b(\w+(?:\s+\w+)*)\s+(\w+(?:ed|s|es|ing|en|t)?)\s+(\w+(?:\s+\w+)*)", re.IGNORECASE
)


def _normalize_tokens(text: str) -> List[str]:
    """Extract and lowercase alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def extract_semantic_roles_from_sentence(sentence: str) -> SemanticTriple:
    """Extract Agent-Action-Patient roles from a single sentence.

    Uses regex heuristics to identify passive and active voice structures.
    If passive voice is detected, the object of the preposition 'by' is
    assigned as the Agent, the participle as the Action, and the subject
    as the Patient.

    Args:
        sentence: The input sentence string.

    Returns:
        A SemanticTriple object representing the extracted roles.
    """
    sentence = sentence.strip()
    if not sentence:
        return SemanticTriple(raw_sentence=sentence)

    triple = SemanticTriple(raw_sentence=sentence)

    # Check for passive voice first
    passive_match = PASSIVE_VOICE_PATTERN.search(sentence)
    if passive_match:
        patient_text = passive_match.group(1).strip()
        action_text = passive_match.group(2).strip()
        agent_text = passive_match.group(3).strip()

        triple.patient = SemanticRole(
            "PATIENT", patient_text, _normalize_tokens(patient_text)
        )
        triple.action = SemanticRole(
            "ACTION", action_text, _normalize_tokens(action_text)
        )
        triple.agent = SemanticRole("AGENT", agent_text, _normalize_tokens(agent_text))
        return triple

    # Fallback to active voice heuristic
    active_match = ACTIVE_VOICE_PATTERN.search(sentence)
    if active_match:
        agent_text = active_match.group(1).strip()
        action_text = active_match.group(2).strip()
        patient_text = active_match.group(3).strip()

        triple.agent = SemanticRole("AGENT", agent_text, _normalize_tokens(agent_text))
        triple.action = SemanticRole(
            "ACTION", action_text, _normalize_tokens(action_text)
        )
        triple.patient = SemanticRole(
            "PATIENT", patient_text, _normalize_tokens(patient_text)
        )

    return triple


def extract_document_semantic_roles(text: str) -> List[SemanticTriple]:
    """Extract semantic roles from an entire document.

    Splits the document into sentences and extracts semantic triples
    from each sentence.

    Args:
        text: The full document text.

    Returns:
        List of SemanticTriple objects.
    """
    if not text or not isinstance(text, str):
        return []

    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    triples = []

    for sent in sentences:
        triple = extract_semantic_roles_from_sentence(sent)
        if triple.agent or triple.action or triple.patient:
            triples.append(triple)

    logger.info("Extracted %d semantic triples from document.", len(triples))
    return triples
