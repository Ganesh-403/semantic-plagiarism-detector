import json

import numpy as np
import pandas as pd

from src.utils.json_export import export_similarity_matrix_to_json


def test_export_similarity_matrix_to_json_valid():
    """Verify that a valid similarity matrix is converted to a clean JSON array of unique pairs."""
    data = [[1.0, 0.85, 0.45], [0.85, 1.0, 0.92], [0.45, 0.92, 1.0]]
    df = pd.DataFrame(
        data, index=["docA", "docB", "docC"], columns=["docA", "docB", "docC"]
    )

    json_str = export_similarity_matrix_to_json(df)
    result = json.loads(json_str)

    # 3 unique pairs: (docA, docB), (docA, docC), (docB, docC)
    assert len(result) == 3

    assert result[0] == {
        "document_1": "docA",
        "document_2": "docB",
        "similarity_score": 0.85,
    }
    assert result[1] == {
        "document_1": "docA",
        "document_2": "docC",
        "similarity_score": 0.45,
    }
    assert result[2] == {
        "document_1": "docB",
        "document_2": "docC",
        "similarity_score": 0.92,
    }


def test_export_similarity_matrix_to_json_empty():
    """Verify that empty or None DataFrame returns empty JSON array."""
    assert export_similarity_matrix_to_json(None) == "[]"
    assert export_similarity_matrix_to_json(pd.DataFrame()) == "[]"


def test_export_similarity_matrix_to_json_single_document():
    """Verify that a single document similarity matrix returns an empty array (no pairs)."""
    df = pd.DataFrame([[1.0]], index=["docA"], columns=["docA"])
    assert export_similarity_matrix_to_json(df) == "[]"


def test_export_similarity_matrix_to_json_nan_handling():
    """Verify that NaN similarity scores are gracefully set to 0.0 in JSON export."""
    data = [[1.0, np.nan], [np.nan, 1.0]]
    df = pd.DataFrame(data, index=["docA", "docB"], columns=["docA", "docB"])

    json_str = export_similarity_matrix_to_json(df)
    result = json.loads(json_str)

    assert len(result) == 1
    assert result[0] == {
        "document_1": "docA",
        "document_2": "docB",
        "similarity_score": 0.0,
    }


def test_export_similarity_matrix_to_json_unicode_filenames():
    """Verify that Unicode (UTF-8) characters in filenames are preserved and not escaped."""
    data = [[1.0, 0.75], [0.75, 1.0]]
    df = pd.DataFrame(
        data, index=["📄_doc.txt", "doc_üñ.txt"], columns=["📄_doc.txt", "doc_üñ.txt"]
    )

    json_str = export_similarity_matrix_to_json(df)

    # Check that Unicode characters are in their raw representation (not escaped \uXXXX)
    assert "📄_doc.txt" in json_str
    assert "doc_üñ.txt" in json_str

    result = json.loads(json_str)
    assert len(result) == 1
    assert result[0]["document_1"] == "📄_doc.txt"
    assert result[0]["document_2"] == "doc_üñ.txt"


def test_export_similarity_matrix_to_json_is_pretty_printed():
    """Verify that JSON output uses two-space indentation."""
    data = [[1.0, 0.85], [0.85, 1.0]]
    df = pd.DataFrame(data, index=["docA", "docB"], columns=["docA", "docB"])

    json_str = export_similarity_matrix_to_json(df)

    lines = json_str.splitlines()

    assert lines[0] == "["
    assert lines[1].startswith("  {")
    assert lines[-1] == "]"
