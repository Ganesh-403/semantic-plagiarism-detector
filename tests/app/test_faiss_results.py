from dataclasses import dataclass

import pandas as pd

from app.components.faiss_results import (RESULT_COLUMNS,
                                          faiss_results_dataframe)


@dataclass
class Record:
    doc_name: str
    chunk_index: int
    chunk_text: str


def test_faiss_results_dataframe_has_sortable_columns():
    dataframe = faiss_results_dataframe(
        [
            (Record("beta.pdf", 2, "Lower match."), 0.72),
            (Record("alpha.pdf", 0, "Strongest match."), 0.96),
        ]
    )

    assert list(dataframe.columns) == RESULT_COLUMNS
    assert dataframe["Similarity Score"].dtype.kind == "f"
    assert dataframe["Chunk"].dtype.kind in {"i", "u"}
    assert dataframe["Rank"].dtype.kind in {"i", "u"}


def test_results_default_to_similarity_descending():
    dataframe = faiss_results_dataframe(
        [
            (Record("low.pdf", 0, "Low"), 0.25),
            (Record("high.pdf", 1, "High"), 0.95),
            (Record("middle.pdf", 2, "Middle"), 0.60),
        ]
    )

    assert dataframe["Target Document"].tolist() == [
        "high.pdf",
        "middle.pdf",
        "low.pdf",
    ]
    assert dataframe["Rank"].tolist() == [1, 2, 3]


def test_target_document_breaks_equal_score_ties():
    dataframe = faiss_results_dataframe(
        [
            (Record("zeta.pdf", 1, "Z"), 0.8),
            (Record("alpha.pdf", 2, "A"), 0.8),
        ]
    )

    assert dataframe["Target Document"].tolist() == [
        "alpha.pdf",
        "zeta.pdf",
    ]


def test_chunk_numbers_are_one_based():
    dataframe = faiss_results_dataframe(
        [(Record("report.pdf", 0, "Text"), 0.9)]
    )
    assert dataframe.loc[0, "Chunk"] == 1


def test_mapping_records_are_supported():
    dataframe = faiss_results_dataframe(
        [
            (
                {
                    "doc_name": "mapping.pdf",
                    "chunk_index": 4,
                    "chunk_text": "Mapped record",
                },
                0.88,
            )
        ]
    )

    assert dataframe.loc[0, "Target Document"] == "mapping.pdf"
    assert dataframe.loc[0, "Chunk"] == 5
    assert dataframe.loc[0, "Matching Text"] == "Mapped record"


def test_empty_results_return_empty_table():
    dataframe = faiss_results_dataframe([])
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.empty
    assert list(dataframe.columns) == RESULT_COLUMNS


def test_faiss_results_filtering_by_similarity_range():
    records = [
        (Record("doc1.pdf", 0, "Match 1"), 0.50),
        (Record("doc2.pdf", 1, "Match 2"), 0.85),
        (Record("doc3.pdf", 2, "Match 3"), 0.95),
    ]

    # No filter (default)
    df_all = faiss_results_dataframe(records)
    assert len(df_all) == 3

    # Min filter
    df_min = faiss_results_dataframe(records, min_similarity=0.80)
    assert len(df_min) == 2
    assert "doc1.pdf" not in df_min["Target Document"].tolist()

    # Max filter
    df_max = faiss_results_dataframe(records, max_similarity=0.90)
    assert len(df_max) == 2
    assert "doc3.pdf" not in df_max["Target Document"].tolist()

    # Min and Max filter range
    df_range = faiss_results_dataframe(records, min_similarity=0.60, max_similarity=0.90)
    assert len(df_range) == 1
    assert df_range.iloc[0]["Target Document"] == "doc2.pdf"
