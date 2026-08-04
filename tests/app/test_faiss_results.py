import numpy as np
import pandas as pd
from app.components.faiss_results import faiss_results_dataframe, RESULT_COLUMNS


class MockRecord:
    def __init__(self, doc_name: str, chunk_index: int, chunk_text: str):
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.chunk_text = chunk_text


def test_faiss_results_dataframe_empty():
    """Verify empty input returns empty DataFrame with correct columns."""
    df = faiss_results_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 0


def test_faiss_results_dataframe_valid():
    """Verify valid results are correctly formatted, sorted, and ranked."""
    results = [
        (MockRecord("doc_a.pdf", 0, "This is chunk 1 text."), 0.75),
        ({"doc_name": "doc_b.pdf", "chunk_index": 1, "chunk_text": "This is chunk 2 text."}, 0.92),
        (MockRecord("doc_c.pdf", 2, "This is chunk 3 text."), 0.60),
    ]

    df = faiss_results_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 3

    # Ranking & Sorting (highest score first)
    assert df.loc[0, "Rank"] == 1
    assert df.loc[0, "Target Document"] == "doc_b.pdf"
    assert df.loc[0, "Similarity Score"] == 0.92

    assert df.loc[1, "Rank"] == 2
    assert df.loc[1, "Target Document"] == "doc_a.pdf"
    assert df.loc[1, "Similarity Score"] == 0.75

    assert df.loc[2, "Rank"] == 3
    assert df.loc[2, "Target Document"] == "doc_c.pdf"
    assert df.loc[2, "Similarity Score"] == 0.60


def test_faiss_results_dataframe_similarity_filters():
    """Verify min_similarity and max_similarity filtering works."""
    results = [
        (MockRecord("doc_a.pdf", 0, "text1"), 0.50),
        (MockRecord("doc_b.pdf", 1, "text2"), 0.70),
        (MockRecord("doc_c.pdf", 2, "text3"), 0.90),
    ]

    # Min similarity filter
    df_min = faiss_results_dataframe(results, min_similarity=0.70)
    assert len(df_min) == 2
    assert "doc_a.pdf" not in df_min["Target Document"].values

    # Max similarity filter
    df_max = faiss_results_dataframe(results, max_similarity=0.80)
    assert len(df_max) == 2
    assert "doc_c.pdf" not in df_max["Target Document"].values

    # Both filters
    df_both = faiss_results_dataframe(results, min_similarity=0.60, max_similarity=0.80)
    assert len(df_both) == 1
    assert df_both.iloc[0]["Target Document"] == "doc_b.pdf"


def test_faiss_results_dataframe_nan_and_inf_scores():
    """Verify handling of NaN and Inf scores does not crash and works gracefully."""
    results = [
        (MockRecord("doc_a.pdf", 0, "text1"), float("nan")),
        (MockRecord("doc_b.pdf", 1, "text2"), float("inf")),
        (MockRecord("doc_c.pdf", 2, "text3"), 0.85),
    ]

    df = faiss_results_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    # inf score should sort first, then 0.85, then nan should be sorted at the end
    assert df.iloc[0]["Target Document"] == "doc_b.pdf"
    assert df.iloc[0]["Similarity Score"] == float("inf")

    assert df.iloc[1]["Target Document"] == "doc_c.pdf"
    assert df.iloc[1]["Similarity Score"] == 0.85

    assert df.iloc[2]["Target Document"] == "doc_a.pdf"
    assert np.isnan(df.iloc[2]["Similarity Score"])


def test_faiss_results_dataframe_missing_attributes():
    """Verify default values are used when fields are missing from records."""
    # Record has no doc_name or chunk_index attributes
    results = [
        (object(), 0.80),
    ]

    df = faiss_results_dataframe(results)
    assert len(df) == 1
    assert df.iloc[0]["Target Document"] == "Unknown document"
    assert df.iloc[0]["Chunk"] == 1  # chunk_index (default 0) + 1
