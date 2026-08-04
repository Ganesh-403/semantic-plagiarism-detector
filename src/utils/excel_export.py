"""
src/utils/excel_export.py
-------------------------
Utility for exporting similarity matrices into styled Excel (.xlsx) workbooks
with conditional formatting matching the application's heatmap logic.
Supports both in-memory generation and managed temporary disk-file creation with automatic exit cleanup.
Also provides streaming CSV generation for memory-efficient exports of large datasets.
"""

import atexit
import csv
import io
import os
import tempfile
from typing import Generator

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.comments import Comment


def _create_managed_temp_file(suffix: str = ".xlsx", prefix: str = "temp_") -> str:
    """Helper to create a temporary file that is automatically deleted on exit."""
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)

    def _cleanup():
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    atexit.register(_cleanup)
    return temp_path


def _truncate_title(title: str, max_length: int = 60) -> str:
    """
    Truncate a title to max_length characters, appending '...' if truncated.

    Args:
        title: The title to truncate
        max_length: Maximum length before truncation (default: 60)

    Returns:
        Truncated title with '...' suffix if original was longer
    """
    if len(title) <= max_length:
        return title
    return title[: max_length - 3] + "..."


def build_similarity_workbook(
    df: pd.DataFrame, threshold: float = 0.59
) -> Workbook:
    """Helper function that builds and styles the openpyxl Workbook."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Similarity Matrix"

    # Write headers and index labels with truncated titles, preserving full names in comments
    ws.cell(row=1, column=1, value="Document")
    for col_idx, col_name in enumerate(df.columns, start=2):
        truncated_name = _truncate_title(col_name)
        cell = ws.cell(row=1, column=col_idx, value=truncated_name)
        # Add full title as comment if truncated
        if len(col_name) > 60:
            cell.comment = Comment(col_name, "Excel Export")

    for row_idx, (index_label, row) in enumerate(df.iterrows(), start=2):
        truncated_label = _truncate_title(index_label)
        cell = ws.cell(row=row_idx, column=1, value=truncated_label)
        # Add full title as comment if truncated
        if len(index_label) > 60:
            cell.comment = Comment(index_label, "Excel Export")

        for col_idx, val in enumerate(row, start=2):
            cell = ws.cell(row=row_idx, column=col_idx, value=float(val))
            cell.number_format = "0.0%"
            cell.alignment = Alignment(horizontal="right")

    # Header styling
    header_fill = PatternFill(
        start_color="1F2937", end_color="1F2937", fill_type="solid"
    )
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in ws.iter_rows(min_row=2, min_col=1, max_col=1):
        for cell in row:
            cell.fill = header_fill
            cell.font = header_font

    # Apply Conditional Formatting (3-Color Scale)
    max_row = len(df) + 1
    max_col = len(df.columns) + 1

    if max_row > 1 and max_col > 1:
        start_cell = "B2"
        end_col_letter = ws.cell(row=max_row, column=max_col).column_letter
        end_cell = f"{end_col_letter}{max_row}"
        matrix_range = f"{start_cell}:{end_cell}"

        color_scale = ColorScaleRule(
            start_type="num",
            start_value=0.0,
            start_color="FFFFFF",  # White (0%)
            mid_type="num",
            mid_value=threshold,
            mid_color="FEF08A",  # Yellow (At threshold)
            end_type="num",
            end_value=1.0,
            end_color="EF4444",  # Red (100%)
        )
        ws.conditional_formatting.add(matrix_range, color_scale)

    # Auto-adjust column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    return wb


def export_similarity_matrix_to_excel(
    df: pd.DataFrame, threshold: float = 0.59
) -> bytes:
    """Exports a similarity matrix DataFrame into an in-memory Excel file (.xlsx) with formatting."""
    wb = build_similarity_workbook(df, threshold=threshold)
    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


def export_similarity_matrix_to_temp_file(
    df: pd.DataFrame, threshold: float = 0.59
) -> str:
    """
    Exports the similarity matrix to a temporary .xlsx file on disk.
    The created file is automatically registered for cleanup on application exit via atexit.

    Returns:
        str: Absolute path to the created temporary Excel file.
    """
    wb = build_similarity_workbook(df, threshold=threshold)
    temp_path = _create_managed_temp_file(suffix=".xlsx", prefix="similarity_matrix_")
    wb.save(temp_path)
    return temp_path


def generate_csv_matrix_stream(matrix_df: pd.DataFrame) -> Generator[str, None, None]:
    """
    Yields CSV formatted lines line-by-line from a similarity matrix DataFrame.

    Memory-efficient generator for exporting large result sets (>10,000 document pairs)
    without materializing the entire formatted output string or Excel workbook in memory.

    Args:
        matrix_df (pd.DataFrame): Similarity matrix DataFrame with document labels as index and columns.

    Yields:
        str: CSV formatted string row (including newline character).
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    # Yield header row
    header = ["Document"] + list(matrix_df.columns)
    writer.writerow(header)
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    # Yield data rows line by line
    for index, row in matrix_df.iterrows():
        writer.writerow([index] + row.tolist())
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
