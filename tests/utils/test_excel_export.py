import inspect
import io
from datetime import datetime, timezone

import openpyxl
import pandas as pd

from src.utils.bulk_export import export_incidents_xlsx_stream
from src.utils.excel_export import (
    build_similarity_workbook,
    export_similarity_matrix_to_excel,
    generate_csv_matrix_stream,
)


def test_generate_csv_matrix_stream():
    # Setup test DataFrame
    data = {
        "DocA.txt": [1.0, 0.85, 0.12],
        "DocB.txt": [0.85, 1.0, 0.45],
        "DocC.txt": [0.12, 0.45, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    # Test 1: Return type is a Generator
    stream = generate_csv_matrix_stream(df)
    assert inspect.isgenerator(stream)

    # Test 2: Verify chunk output
    chunks = list(stream)
    assert len(chunks) == len(df) + 1  # 1 header row + 3 data rows

    # Verify header line
    assert chunks[0].strip() == "Document,DocA.txt,DocB.txt,DocC.txt"

    # Verify data lines
    assert chunks[1].strip() == "DocA.txt,1.0,0.85,0.12"
    assert chunks[2].strip() == "DocB.txt,0.85,1.0,0.45"
    assert chunks[3].strip() == "DocC.txt,0.12,0.45,1.0"

    # Test 3: Verify complete CSV reconstruction matches Expected CSV output
    full_csv = "".join(chunks)
    reconstructed_df = pd.read_csv(io.StringIO(full_csv), index_col=0)
    pd.testing.assert_frame_equal(df, reconstructed_df, check_names=False)


def test_build_similarity_workbook_metadata_properties():
    """Verify build_similarity_workbook populates document title, creator, and created timestamp (#3438)."""
    df = pd.DataFrame({"Doc1.txt": [1.0]}, index=["Doc1.txt"])
    before = datetime.now(timezone.utc)
    wb = build_similarity_workbook(df)
    after = datetime.now(timezone.utc)

    assert wb.properties.title == "Semantic Plagiarism Similarity Report"
    assert wb.properties.creator == "Semantic Plagiarism Detector"
    assert wb.properties.created is not None
    assert isinstance(wb.properties.created, datetime)
    assert before <= wb.properties.created <= after


def test_export_similarity_matrix_to_excel_persists_metadata():
    """Verify export_similarity_matrix_to_excel persists metadata in the saved XLSX file (#3438)."""
    df = pd.DataFrame(
        {"DocA.txt": [1.0, 0.8], "DocB.txt": [0.8, 1.0]},
        index=["DocA.txt", "DocB.txt"],
    )
    xlsx_bytes = export_similarity_matrix_to_excel(df)
    assert isinstance(xlsx_bytes, bytes)
    assert len(xlsx_bytes) > 0

    # Load back with openpyxl to inspect file properties
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    assert wb.properties.title == "Semantic Plagiarism Similarity Report"
    assert wb.properties.creator == "Semantic Plagiarism Detector"
    assert wb.properties.created is not None


def test_export_incidents_xlsx_stream_persists_metadata():
    """Verify export_incidents_xlsx_stream sets title, creator, and created metadata (#3438)."""
    incidents = [
        {
            "incident_id": "INC-001",
            "document_a": "Essay1.docx",
            "document_b": "Essay2.docx",
            "similarity_score": 0.88,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": "2026-08-25",
        }
    ]
    xlsx_bytes = export_incidents_xlsx_stream(incidents)
    assert isinstance(xlsx_bytes, bytes)

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    assert wb.properties.title == "Semantic Plagiarism Similarity Report"
    assert wb.properties.creator == "Semantic Plagiarism Detector"
    assert wb.properties.created is not None


def test_flagged_pairs_sheet_lists_pairs_above_threshold():
    """Verify the Flagged Pairs worksheet contains pairs at or above threshold (#3433)."""
    data = {
        "DocA.txt": [1.0, 0.80, 0.30],
        "DocB.txt": [0.80, 1.0, 0.20],
        "DocC.txt": [0.30, 0.20, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    wb = build_similarity_workbook(df, threshold=0.59)
    ws_flagged = wb["Flagged Pairs"]

    assert ws_flagged.cell(row=1, column=1).value == "Document A"
    assert ws_flagged.cell(row=1, column=2).value == "Document B"
    assert ws_flagged.cell(row=1, column=3).value == "Similarity Score"
    assert ws_flagged.cell(row=1, column=4).value == "Severity"

    pairs = set()
    for row in ws_flagged.iter_rows(min_row=2, max_col=2, values_only=True):
        pairs.add(tuple(row))
    assert ("DocA.txt", "DocB.txt") in pairs
    assert ("DocB.txt", "DocA.txt") in pairs
    assert ("DocA.txt", "DocC.txt") not in pairs
    assert ("DocC.txt", "DocA.txt") not in pairs
    assert ("DocB.txt", "DocC.txt") not in pairs
    assert ("DocC.txt", "DocB.txt") not in pairs


def test_flagged_pairs_sorted_by_similarity_descending():
    """Verify flagged pairs are sorted highest similarity first (#3433)."""
    data = {
        "DocA.txt": [1.0, 0.95, 0.70],
        "DocB.txt": [0.95, 1.0, 0.65],
        "DocC.txt": [0.70, 0.65, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    wb = build_similarity_workbook(df, threshold=0.59)
    ws_flagged = wb["Flagged Pairs"]

    scores = []
    for row in ws_flagged.iter_rows(min_row=2, min_col=3, max_col=3, values_only=True):
        scores.append(row[0])

    assert scores == sorted(scores, reverse=True)


def test_flagged_pairs_severity_labels():
    """Verify severity labels are correctly assigned (#3433)."""
    data = {
        "DocA.txt": [1.0, 0.80, 0.92],
        "DocB.txt": [0.80, 1.0, 0.60],
        "DocC.txt": [0.92, 0.60, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    wb = build_similarity_workbook(df, threshold=0.59)
    ws_flagged = wb["Flagged Pairs"]

    severities = {}
    for row in ws_flagged.iter_rows(min_row=2, max_col=4, values_only=True):
        key = (row[0], row[1])
        severities[key] = row[3]

    assert severities[("DocA.txt", "DocB.txt")] == "Medium"
    assert severities[("DocB.txt", "DocA.txt")] == "Medium"
    assert severities[("DocA.txt", "DocC.txt")] == "High"
    assert severities[("DocC.txt", "DocA.txt")] == "High"
    assert severities[("DocB.txt", "DocC.txt")] == "Low"
    assert severities[("DocC.txt", "DocB.txt")] == "Low"


def test_no_flagged_pairs_sheet_when_none_above_threshold():
    """Verify no Flagged Pairs sheet is created when all scores are below threshold (#3433)."""
    data = {
        "DocA.txt": [1.0, 0.30],
        "DocB.txt": [0.30, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt"])

    wb = build_similarity_workbook(df, threshold=0.59)
    sheet_names = wb.sheetnames
    assert "Similarity Matrix" in sheet_names
    assert "Flagged Pairs" not in sheet_names


def test_flagged_pairs_diagonal_excluded():
    """Verify self-pairs (diagonal) are excluded from Flagged Pairs (#3433)."""
    data = {
        "DocA.txt": [1.0, 0.85],
        "DocB.txt": [0.85, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt"])

    wb = build_similarity_workbook(df, threshold=0.59)
    ws_flagged = wb["Flagged Pairs"]

    for row in ws_flagged.iter_rows(min_row=2, max_col=2, values_only=True):
        assert row[0] != row[1], f"Self-pair detected: {row}"

