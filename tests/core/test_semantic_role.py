"""
tests/core/test_semantic_role.py
--------------------------------
Unit tests for Semantic Role Labeling and Argument Structure Alignment.
"""

import pytest
from src.core.semantic_role_extractor import (
    extract_semantic_roles_from_sentence,
    extract_document_semantic_roles,
)
from src.core.argument_structure_aligner import compute_role_sequence_similarity


class TestSemanticRoleExtractor:
    def test_extract_active_voice(self):
        triple = extract_semantic_roles_from_sentence("The cat chased the mouse.")
        assert triple.agent is not None
        assert "cat" in triple.agent.text.lower()
        assert triple.action is not None
        assert "chased" in triple.action.text.lower()
        assert triple.patient is not None
        assert "mouse" in triple.patient.text.lower()

    def test_extract_passive_voice(self):
        triple = extract_semantic_roles_from_sentence(
            "The mouse was chased by the cat."
        )
        assert triple.agent is not None
        assert "cat" in triple.agent.text.lower()
        assert triple.patient is not None
        assert "mouse" in triple.patient.text.lower()

    def test_extract_document_roles(self):
        text = "The dog barked at the mailman. The mailman was frightened by the dog."
        triples = extract_document_semantic_roles(text)
        assert len(triples) >= 1


class TestArgumentStructureAligner:
    def test_compute_similarity_identical_structure(self):
        triples_a = extract_document_semantic_roles("The cat chased the mouse.")
        triples_b = extract_document_semantic_roles("The dog chased the rabbit.")
        result = compute_role_sequence_similarity(triples_a, triples_b)
        assert result["structural_similarity"] == 1.0
        assert result["is_deep_paraphrase"] is True

    def test_compute_similarity_different_structure(self):
        triples_a = extract_document_semantic_roles("The cat chased the mouse.")
        triples_b = extract_document_semantic_roles("The mouse ran away.")
        result = compute_role_sequence_similarity(triples_a, triples_b)
        assert result["structural_similarity"] < 1.0
