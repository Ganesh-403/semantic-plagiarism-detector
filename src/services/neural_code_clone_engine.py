"""Neural Code Clone Service Engine.

Provides AST tokenization, neural semantic embedding calculations, clone type classification,
and obfuscation detection algorithms for multi-language source code.
"""

import math
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.models.neural_code_clone_model import (
    CodeAstEmbedding,
    CodeCloneMatch,
    CodeCloneScanReport,
)


class NeuralCodeCloneEngine:
    """Core analytics engine for detecting syntactic and semantic code clones."""

    @staticmethod
    def calculate_token_cosine_similarity(
        vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculates cosine similarity between two vector embeddings."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        return round(dot_product / (magnitude1 * magnitude2), 4)

    @staticmethod
    def classify_clone_type(
        ast_score: float, token_score: float, semantic_score: float
    ) -> Tuple[str, bool]:
        """Classifies code clone into Type 1-4 based on structural and semantic thresholds."""
        obfuscation = False

        if ast_score >= 0.98 and token_score >= 0.98:
            clone_type = "Type-1 (Exact Match)"
        elif ast_score >= 0.85 and token_score >= 0.80:
            clone_type = "Type-2 (Renamed Identifiers)"
        elif ast_score >= 0.70 and semantic_score >= 0.75:
            clone_type = "Type-3 (AST Structural Alteration)"
        elif semantic_score >= 0.80 and ast_score < 0.60:
            clone_type = "Type-4 (Semantic Functional Equivalent)"
            obfuscation = True
        else:
            clone_type = "Type-3 (Partial Structural Overlap)"

        return clone_type, obfuscation

    @classmethod
    def analyze_code_pair(
        cls,
        source_id: str,
        target_id: str,
        source_ast: CodeAstEmbedding,
        target_ast: CodeAstEmbedding,
    ) -> CodeCloneMatch:
        """Compares two AST code embeddings and generates a clone match record."""
        ast_sim = cls.calculate_token_cosine_similarity(
            source_ast.vector_embedding, target_ast.vector_embedding
        )

        # Token overlap estimation based on token count ratios
        min_tokens = min(source_ast.ast_token_count, target_ast.ast_token_count)
        max_tokens = max(source_ast.ast_token_count, target_ast.ast_token_count)
        token_sim = round(min_tokens / max_tokens, 4) if max_tokens > 0 else 0.0

        # Simulated transformer neural semantic score
        semantic_sim = round((ast_sim * 0.6) + (token_sim * 0.4), 4)
        overall_score = round(
            (ast_sim * 0.4) + (token_sim * 0.3) + (semantic_sim * 0.3), 4
        )

        clone_type, obfuscation = cls.classify_clone_type(
            ast_sim, token_sim, semantic_sim
        )

        return CodeCloneMatch(
            clone_id=f"CLONE-{uuid.uuid4().hex[:8].upper()}",
            source_file_id=source_id,
            target_file_id=target_id,
            ast_similarity_score=ast_sim,
            token_overlap_score=token_sim,
            neural_semantic_similarity=semantic_sim,
            overall_clone_score=overall_score,
            clone_type=clone_type,
            obfuscation_detected=obfuscation,
            detected_at=datetime.utcnow(),
        )
