"""
tests/core/test_tabular_plagiarism.py
-------------------------------------
Unit tests for Tabular Data and CSV Structural Plagiarism Detection.
"""

import pytest
from src.core.tabular_data_extractor import extract_table_fingerprint, _infer_dtype
from src.core.table_structure_aligner import (
    compute_distribution_similarity,
    compute_table_similarity,
)


class TestTabularDataExtractor:
    def test_infer_dtype_numeric(self):
        assert _infer_dtype(["1.5", "2.0", "3.14", ""]) == "numeric"

    def test_infer_dtype_categorical(self):
        assert _infer_dtype(["A", "B", "A", "B", "A", "B"]) == "categorical"

    def test_extract_table_fingerprint(self):
        csv = "id,value\n1,10.5\n2,20.5\n3,30.5"
        fp = extract_table_fingerprint(csv)
        assert fp.row_count == 3
        assert fp.column_count == 2
        assert fp.columns[1].dtype == "numeric"
        assert fp.columns[1].mean == 20.5


class TestTableStructureAligner:
    def test_compute_distribution_similarity_identical(self):
        from src.core.tabular_data_extractor import ColumnSchema

        col_a = ColumnSchema(
            "A", "numeric", 3, 0, mean=10.0, std_dev=2.0, min_val=8.0, max_val=12.0
        )
        col_b = ColumnSchema(
            "B", "numeric", 3, 0, mean=10.0, std_dev=2.0, min_val=8.0, max_val=12.0
        )
        sim = compute_distribution_similarity(col_a, col_b)
        assert sim == 1.0

    def test_compute_distribution_similarity_scaled(self):
        from src.core.tabular_data_extractor import ColumnSchema

        # Linear transformation: y = 2x
        col_a = ColumnSchema(
            "A", "numeric", 3, 0, mean=10.0, std_dev=2.0, min_val=8.0, max_val=12.0
        )
        col_b = ColumnSchema(
            "B", "numeric", 3, 0, mean=20.0, std_dev=4.0, min_val=16.0, max_val=24.0
        )
        sim = compute_distribution_similarity(col_a, col_b)
        assert (
            sim > 0.8
        )  # Should be highly similar due to invariant CV and normalized range

    def test_compute_table_similarity_cloned(self):
        csv_a = "id,val\n1,10\n2,20"
        csv_b = "idx,value\n1,100\n2,200"  # Scaled by 10
        fp_a = extract_table_fingerprint(csv_a)
        fp_b = extract_table_fingerprint(csv_b)
        result = compute_table_similarity(fp_a, fp_b)
        assert result["is_cloned_dataset"] is True
