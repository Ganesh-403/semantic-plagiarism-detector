"""
Comprehensive Unit Tests for Embedding Vector L2-Normalization
Issue: #4015
"""

import math
import pytest
from src.utils.embedding_utils import l2_normalize


class TestL2NormalizationBasic:
    def test_normalizes_simple_vector(self):
        """Should normalize a simple vector."""
        result = l2_normalize([3, 4])
        norm = math.sqrt(sum(val * val for val in result))
        assert norm == pytest.approx(1.0)

    def test_normalized_length_is_one(self):
        """Should ensure the length of the vector is 1 after normalization."""
        vector = [1, 2, 3]
        result = l2_normalize(vector)
        norm = math.sqrt(sum(val * val for val in result))
        assert norm == pytest.approx(1.0)

    def test_direction_preserved(self):
        """Should preserve the direction of the vector."""
        vector = [10, 20]
        result = l2_normalize(vector)
        assert result[0] == pytest.approx(10 / math.sqrt(500))
        assert result[1] == pytest.approx(20 / math.sqrt(500))

    def test_negative_values(self):
        """Should handle negative values."""
        result = l2_normalize([-3, -4])
        norm = math.sqrt(sum(val * val for val in result))
        assert norm == pytest.approx(1.0)


class TestL2NormalizationEdgeCases:
    def test_zero_vector(self):
        """Should handle a zero vector (return unchanged)."""
        result = l2_normalize([0, 0, 0])
        assert result == [0, 0, 0]

    def test_single_element_vector(self):
        """Should handle a single element vector."""
        result = l2_normalize([5])
        assert result == [1.0]

    def test_empty_vector(self):
        """Should handle an empty vector."""
        with pytest.raises(ZeroDivisionError):
            l2_normalize([])

    def test_float_values(self):
        """Should handle float values."""
        vector = [0.5, 0.5]
        result = l2_normalize(vector)
        norm = math.sqrt(sum(val * val for val in result))
        assert norm == pytest.approx(1.0)


class TestL2NormalizationAiModels:
    def test_high_dimensional_vector(self):
        """Should handle a large 512-dimensional vector."""
        vector = [1.0] * 512
        result = l2_normalize(vector)
        norm = math.sqrt(sum(val * val for val in result))
        assert norm == pytest.approx(1.0)

    def test_mixed_dimensions(self):
        """Should handle vectors with different dimensions."""
        assert len(l2_normalize([1, 2, 3])) == 3
        assert len(l2_normalize([1, 2])) == 2


class TestL2NormalizationNumericalStability:
    def test_large_numbers(self):
        """Should handle very large numbers without overflow."""
        vector = [1e6, 2e6]
        result = l2_normalize(vector)
        norm = math.sqrt(sum(val * val for val in result))
        assert norm == pytest.approx(1.0)

    def test_small_numbers(self):
        """Should handle very small numbers."""
        vector = [1e-6, 2e-6]
        result = l2_normalize(vector)
        norm = math.sqrt(sum(val * val for val in result))
        assert norm == pytest.approx(1.0)