"""
src/utils/bulk_export.py
-------------------------
Bulk export utilities for generating ZIP archives, CSV streams, JSON payloads,
and formatted reports from plagiarism detection results.

Provides functions to:
- Stream incident data into multiple formats (CSV, JSON, XLSX) via a unified dispatcher
- Generate multi-format ZIP archives with per-pair plagiarism reports
- Normalize CSV column headers to standardized formats
- Sanitize cell values to prevent formula injection

Recent Additions (Issue #2008):
- Added `ExportFormat` enum to centralize and validate supported export formats.
- Replaced magic strings ("csv", "json", "xlsx", "pdf") with strict enum typing.
- Added `export_incidents_to_format()` dispatcher function that routes data
  to the appropriate serializer based on the `ExportFormat` enum.

Previous Additions (Issue #1253):
- Added `normalize_csv_headers` function for standardized snake_case/Title Case formatting.
"""

import csv
import io
import json
import logging
import os
import re
import zipfile
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.similarity import find_most_similar_chunks
from src.utils.pdf_report import generate_plagiarism_report

logger = logging.getLogger(__name__)

# Standard column headers for the incident CSV export
_CSV_HEADERS = [
    "Incident ID",
    "Doc A",
    "Doc B",
    "Similarity",
    "Severity",
    "Status",
    "Date",
]

# Regular expression pattern matching characters that are invalid
# or problematic in CSV column headers. These include special symbols,
# punctuation marks, whitespace, and other non-alphanumeric characters
# (except underscores and hyphens which are preserved).
_INVALID_HEADER_CHARS_PATTERN = re.compile(r"[^\w\s\-]")

# Pattern to match multiple consecutive underscores for collapsing
_MULTIPLE_UNDERSCORES_PATTERN = re.compile(r"_{2,}")

# Pattern to match leading and trailing underscores or whitespace
_LEADING_TRAILING_PATTERN = re.compile(r"^[\s_]+|[\s_]+$")


class ExportFormat(str, Enum):
    """Enumeration of supported export formats for bulk data extraction.

    Using an Enum instead of raw strings provides IDE autocomplete,
    prevents typos (e.g., "csv" vs "CSV" vs "Csv"), and allows for
    strict type checking in function signatures.

    Members:
        CSV: Comma-Separated Values format.
        JSON: JavaScript Object Notation format.
        XLSX: Microsoft Excel Open XML Spreadsheet format.
        PDF: Portable Document Format for visual reports.
    """

    CSV = "csv"
    JSON = "json"
    XLSX = "xlsx"
    PDF = "pdf"

    @classmethod
    def _missing_(cls, value: Any) -> Optional["ExportFormat"]:
        """Handle case-insensitive lookup and raise ValueError for invalid formats.

        This override allows users to pass strings like "CSV" or "Json" and
        have them automatically resolved to the correct enum member. If the
        value is completely unrecognized, a descriptive ValueError is raised.

        Args:
            value: The raw value to look up.

        Returns:
            The matching ExportFormat member, or None if not found.

        Raises:
            ValueError: If the value does not match any known format.
        """
        if isinstance(value, str):
            # Attempt case-insensitive match
            lower_value = value.lower().strip()
            for member in cls:
                if member.value == lower_value:
                    return member

            # If we reach here, the string was not recognized
            valid_options = ", ".join([f"'{m.value}'" for m in cls])
            raise ValueError(
                f"Invalid export format: '{value}'. "
                f"Supported formats are: {valid_options}."
            )

        # For non-string types, let the default Enum behavior handle it (returns None)
        return None

    def get_mime_type(self) -> str:
        """Return the standard MIME type for this export format."""
        mime_map = {
            ExportFormat.CSV: "text/csv",
            ExportFormat.JSON: "application/json",
            ExportFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ExportFormat.PDF: "application/pdf",
        }
        return mime_map.get(self, "application/octet-stream")

    def get_file_extension(self) -> str:
        """Return the standard file extension for this export format."""
        return f".{self.value}"


def sanitize_csv_cell_value(val: Any) -> str:
    """Sanitize a CSV cell value to prevent CSV formula injection (Issue #1744).

    Prepends a single quote `'` if the string representation of val begins with
    '=', '+', '-', or '@'.

    Args:
        val: Any cell value (string, numeric, None, etc.)

    Returns:
        Sanitized string representation safe from formula injection.
    """
    if val is None:
        return ""
    str_val = str(val)
    if str_val and str_val[0] in ("=", "+", "-", "@"):
        return f"'{str_val}"
    return str_val


def normalize_csv_headers(headers: list[str]) -> list[str]:
    """Normalize CSV column headers to a standardized, clean format.

    Processes each header string by stripping leading/trailing whitespace,
    replacing invalid symbols and spaces with underscores, collapsing
    consecutive underscores, and removing leading/trailing underscores.
    The result is a list of standardized snake_case-style headers that are
    safe for use in CSV files, databases, and data processing pipelines.

    The normalization rules applied to each header:
    1. Strip leading and trailing whitespace
    2. Replace spaces and invalid symbols (anything not alphanumeric,
       underscore, or hyphen) with underscores
    3. Collapse multiple consecutive underscores into a single underscore
    4. Strip leading and trailing underscores from the result
    5. If the result is empty after normalization, use 'column_N' where
       N is the 0-based index of the header

    Args:
        headers: A list of header strings to normalize. May contain
            whitespace, special symbols, punctuation, or mixed case.

    Returns:
        A new list of normalized header strings. Each header is
        stripped of whitespace, has invalid symbols replaced with
        underscores, and consecutive underscores collapsed.

    Examples:
        >>> normalize_csv_headers(["  Incident ID ", "Doc A!", "similarity_score"])
        ['Incident_ID', 'Doc_A', 'similarity_score']

        >>> normalize_csv_headers(["First Name", "Last Name!!!", "  Age  "])
        ['First_Name', 'Last_Name', 'Age']

        >>> normalize_csv_headers(["", "   ", "valid_header"])
        ['column_0', 'column_1', 'valid_header']

        >>> normalize_csv_headers(["Hello World", "Test@#$%Header", "foo__bar"])
        ['Hello_World', 'Test_Header', 'foo_bar']
    """
    if not headers or not isinstance(headers, list):
        logger.debug(
            "normalize_csv_headers: empty or invalid input, returning empty list"
        )
        return []

    normalized: list[str] = []

    for index, header in enumerate(headers):
        # Handle non-string headers by converting to string first
        if header is None:
            cleaned = ""
        else:
            if not isinstance(header, str):
                header = str(header)
            # Step 1: Strip leading and trailing whitespace
            cleaned = header.strip()

        # Step 2: Replace invalid characters with underscores
        cleaned = _INVALID_HEADER_CHARS_PATTERN.sub("_", cleaned)

        # Step 3: Replace spaces with underscores
        cleaned = cleaned.replace(" ", "_")

        # Step 4: Collapse multiple consecutive underscores into one
        cleaned = _MULTIPLE_UNDERSCORES_PATTERN.sub("_", cleaned)

        # Step 5: Strip leading and trailing underscores
        cleaned = _LEADING_TRAILING_PATTERN.sub("", cleaned)

        # Step 6: If the header is empty after cleaning, use a fallback name
        if not cleaned:
            cleaned = f"column_{index}"
            logger.debug(
                "normalize_csv_headers: header at index %d was empty after "
                "normalization, using fallback name '%s'.",
                index,
                cleaned,
            )

        normalized.append(cleaned)

    logger.debug(
        "normalize_csv_headers: normalized %d header(s): %s",
        len(normalized),
        normalized,
    )

    return normalized


def export_incidents_csv_stream(
    incidents_list: List[Dict],
    delimiter: str = ",",
    quoting_style: int = csv.QUOTE_MINIMAL,
) -> bytes:
    """Stream a list of incident dicts into a CSV-formatted byte stream
    encoded with **utf-8-sig** (UTF-8 with BOM) for Excel compatibility.

    The function writes the following columns in order:

    * **Incident ID** – ``incident_id`` field (default: empty string)
    * **Doc A**       – ``document_a`` field
    * **Doc B**       – ``document_b`` field
    * **Similarity**  – ``similarity_score`` formatted as a percentage (e.g. ``95.00%``)
    * **Severity**    – ``severity_rank`` field
    * **Status**      – ``review_status`` field
    * **Date**        – ``date_flagged`` field

    Parameters
    ----------
    incidents_list:
        A list of incident dictionaries, as returned by
        :func:`~src.db.incidents.get_all_incidents`.
    delimiter:
        Single-character field delimiter passed through to
        :class:`csv.DictWriter`. Defaults to ``","``. Use ``";"`` or
        ``"\\t"`` for locales (e.g. many European Excel configurations)
        that expect semicolon- or tab-delimited CSV files.
    quoting_style:
        The quoting mode passed to :class:`csv.DictWriter`. Defaults to
        ``csv.QUOTE_MINIMAL``.

    Returns
    -------
    bytes
        UTF-8-SIG encoded CSV byte stream (includes UTF-8 BOM) ready for
        direct use with Streamlit download buttons or file writing, ensuring
        Excel on Windows opens the file with correct character encoding.

    Examples
    --------
    >>> csv_bytes = export_incidents_csv_stream(incidents)
    >>> assert csv_bytes.startswith(b"\\xef\\xbb\\xbf")  # UTF-8 BOM

    >>> csv_bytes = export_incidents_csv_stream(incidents, delimiter=";")
    >>> assert b";" in csv_bytes
    """
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        delimiter = ","

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=_CSV_HEADERS,
        extrasaction="ignore",
        lineterminator="\r\n",
        delimiter=delimiter,
        quoting=quoting_style,
    )
    writer.writeheader()

    for incident in incidents_list:
        raw_score = incident.get("similarity_score", 0.0)
        try:
            similarity_str = f"{float(raw_score):.2%}"
        except (TypeError, ValueError):
            similarity_str = str(raw_score)

        writer.writerow(
            {
                "Incident ID": sanitize_csv_cell_value(incident.get("incident_id", "")),
                "Doc A": sanitize_csv_cell_value(incident.get("document_a", "")),
                "Doc B": sanitize_csv_cell_value(incident.get("document_b", "")),
                "Similarity": sanitize_csv_cell_value(similarity_str),
                "Severity": sanitize_csv_cell_value(incident.get("severity_rank", "")),
                "Status": sanitize_csv_cell_value(incident.get("review_status", "")),
                "Date": sanitize_csv_cell_value(incident.get("date_flagged", "")),
            }
        )

    csv_text = output.getvalue()
    return csv_text.encode("utf-8-sig")


def export_incidents_json_stream(incidents_list: List[Dict]) -> bytes:
    """Serialize a list of incident dicts into a JSON-formatted byte stream.

    Args:
        incidents_list: List of incident dictionaries.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    # Use default=str to handle datetime objects gracefully
    json_str = json.dumps(incidents_list, indent=2, default=str, ensure_ascii=False)
    return json_str.encode("utf-8")


def export_incidents_xlsx_stream(incidents_list: List[Dict]) -> bytes:
    """Convert a list of incident dicts into an Excel XLSX byte stream.

    Requires pandas and openpyxl. If openpyxl is missing, falls back to CSV.

    Args:
        incidents_list: List of incident dictionaries.

    Returns:
        Raw bytes of the XLSX file.
    """
    try:
        df = pd.DataFrame(incidents_list)

        # Rename columns for better readability in Excel
        column_mapping = {
            "incident_id": "Incident ID",
            "document_a": "Document A",
            "document_b": "Document B",
            "similarity_score": "Similarity Score",
            "severity_rank": "Severity",
            "review_status": "Status",
            "date_flagged": "Date Flagged",
        }
        df = df.rename(columns=column_mapping)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Plagiarism Incidents")

        return output.getvalue()
    except ImportError:
        logger.warning("openpyxl not installed. Falling back to CSV for XLSX request.")
        return export_incidents_csv_stream(incidents_list)


def export_incidents_to_format(
    incidents_list: List[Dict],
    format: ExportFormat | str = ExportFormat.CSV,
) -> bytes:
    """Dispatcher function that routes incident data to the appropriate serializer.

    This is the primary entry point for exporting incidents. It accepts either
    an `ExportFormat` enum member or a raw string (which is coerced to the enum).

    Args:
        incidents_list: List of incident dictionaries to export.
        format: The desired export format. Can be an `ExportFormat` enum member
                or a string like "csv", "json", "xlsx".

    Returns:
        Raw bytes of the serialized data.

    Raises:
        ValueError: If the provided format string is invalid.
        TypeError: If the format argument is not a string or ExportFormat.
    """
    # Coerce string to Enum (raises ValueError if invalid via _missing_)
    if isinstance(format, str):
        format_enum = ExportFormat(format)
    elif isinstance(format, ExportFormat):
        format_enum = format
    else:
        raise TypeError(
            f"format must be an ExportFormat enum or string, got {type(format).__name__}"
        )

    logger.info(
        "Exporting %d incidents to %s format.",
        len(incidents_list),
        format_enum.value.upper(),
    )

    if format_enum == ExportFormat.CSV:
        return export_incidents_csv_stream(incidents_list)
    elif format_enum == ExportFormat.JSON:
        return export_incidents_json_stream(incidents_list)
    elif format_enum == ExportFormat.XLSX:
        return export_incidents_xlsx_stream(incidents_list)
    elif format_enum == ExportFormat.PDF:
        # PDF export typically requires a different data structure (flags, not just incidents)
        # For this dispatcher, we log a warning and fallback to JSON
        logger.warning(
            "PDF export via incident dispatcher is not fully supported. Returning JSON."
        )
        return export_incidents_json_stream(incidents_list)
    else:
        # Unreachable due to Enum validation, but kept for safety
        raise ValueError(f"Unhandled export format: {format_enum}")


def sanitize_export_filename(filename: str, default_ext: str = ".csv") -> str:
    """Strip illegal OS/filesystem characters from the filename and ensure it ends with default_ext."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
    if not sanitized.endswith(default_ext):
        sanitized += default_ext
    return sanitized


def export_incidents_csv(
    incidents_list: List[Dict],
    delimiter: str = ",",
    quoting_style: int = csv.QUOTE_MINIMAL,
    filename: Optional[str] = None,
) -> bytes | Tuple[bytes, str]:
    """Export a list of incident dicts to a CSV-formatted byte stream.

    Validates that the delimiter is a single character string, falling back to a
    comma if an invalid delimiter is supplied.
    """
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        delimiter = ","

    csv_bytes = export_incidents_csv_stream(
        incidents_list, delimiter=delimiter, quoting_style=quoting_style
    )

    if filename is not None:
        return csv_bytes, sanitize_export_filename(filename)
    return csv_bytes


def stream_incidents_csv_chunks(
    query_func: Callable,
    batch_size: int = 1000,
    delimiter: str = ",",
    quoting_style: int = csv.QUOTE_MINIMAL,
) -> Generator[str, None, None]:
    """Stream incidents in chunks to a CSV-formatted string generator.

    This avoids loading all incidents into memory at once by fetching them in batches.
    The first yielded string includes the CSV headers.

    Parameters
    ----------
    query_func:
        Callable accepting ``limit`` and ``offset`` keyword arguments that
        returns a batch of incident dicts.
    batch_size:
        Number of incidents to fetch per batch.
    delimiter:
        Single-character field delimiter. Defaults to ``","``.
    quoting_style:
        The quoting mode passed to :class:`csv.DictWriter`.
    """
    # Yield the header first
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=_CSV_HEADERS,
        extrasaction="ignore",
        lineterminator="\r\n",
        delimiter=delimiter,
        quoting=quoting_style,
    )
    writer.writeheader()
    yield output.getvalue()

    offset = 0
    while True:
        batch = query_func(limit=batch_size, offset=offset)
        if not batch:
            break

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=_CSV_HEADERS,
            extrasaction="ignore",
            lineterminator="\r\n",
            delimiter=delimiter,
            quoting=quoting_style,
        )

        for incident in batch:
            raw_score = incident.get("similarity_score", 0.0)
            try:
                similarity_str = f"{float(raw_score):.2%}"
            except (TypeError, ValueError):
                similarity_str = str(raw_score)

            writer.writerow(
                {
                    "Incident ID": sanitize_csv_cell_value(
                        incident.get("incident_id", "")
                    ),
                    "Doc A": sanitize_csv_cell_value(incident.get("document_a", "")),
                    "Doc B": sanitize_csv_cell_value(incident.get("document_b", "")),
                    "Similarity": sanitize_csv_cell_value(similarity_str),
                    "Severity": sanitize_csv_cell_value(
                        incident.get("severity_rank", "")
                    ),
                    "Status": sanitize_csv_cell_value(
                        incident.get("review_status", "")
                    ),
                    "Date": sanitize_csv_cell_value(incident.get("date_flagged", "")),
                }
            )

        yield output.getvalue()

        offset += batch_size
        if len(batch) < batch_size:
            break


def _sanitise_filename(name: str) -> str:
    """Strip non-alphanumeric characters (except ``-``, ``_``) for safe filenames."""
    return (
        "".join(c for c in name if c.isalnum() or c in ("-", "_")).rstrip() or "unnamed"
    )


def generate_bulk_reports_zip(
    flags: List[Dict],
    *,
    chunked_docs: Optional[Dict[str, List[str]]] = None,
    embeddings: Optional[Dict[str, "np.ndarray"]] = None,
    include_pdf: bool = True,
    include_csv: bool = True,
    include_json: bool = True,
) -> bytes:
    """Generate a ZIP file containing selected artefacts for flagged document pairs.

    Parameters
    ----------
    flags:
        List of flag dicts returned by :func:`~src.core.similarity.flag_plagiarism`.
    chunked_docs:
        Optional mapping of document name → list of text chunks.
    embeddings:
        Optional mapping of document name → NumPy embedding array.
    include_pdf:
        Whether to generate per‑pair PDF reports.
    include_csv:
        Whether to include a summary CSV of all flagged pairs.
    include_json:
        Whether to include a metadata JSON file describing the export.

    Returns
    -------
    bytes
        In‑memory ZIP file contents.
    """
    memory_file = io.BytesIO()
    csv_rows = []

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, flag in enumerate(flags):
            doc_a = flag.get("doc_a", f"doc_A_{idx}")
            doc_b = flag.get("doc_b", f"doc_B_{idx}")
            score = float(flag.get("similarity", 0.0))
            threshold = float(flag.get("threshold_at_time_of_flag", 0.5))

            csv_rows.append(
                {
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "similarity_score": score,
                    "threshold_at_time_of_flag": threshold,
                }
            )

            top_pairs = []
            if (
                chunked_docs
                and embeddings
                and doc_a in chunked_docs
                and doc_b in chunked_docs
            ):
                try:
                    emb_a = embeddings[doc_a]
                    emb_b = embeddings[doc_b]
                    top_pairs = find_most_similar_chunks(
                        chunked_docs[doc_a],
                        chunked_docs[doc_b],
                        emb_a,
                        emb_b,
                        top_k=3,
                        threshold=threshold,
                    )
                except Exception as exc:
                    logger.debug(
                        "Could not compute chunk pairs for %s ↔ %s: %s",
                        doc_a,
                        doc_b,
                        exc,
                    )

            if include_pdf:
                try:
                    pdf_buffer = generate_plagiarism_report(
                        doc_a=doc_a,
                        doc_b=doc_b,
                        overall_similarity=score,
                        threshold=threshold,
                        top_pairs=top_pairs,
                        report_title=f"Plagiarism Report: {doc_a} vs {doc_b}",
                    )
                    safe_a = _sanitise_filename(doc_a)
                    safe_b = _sanitise_filename(doc_b)
                    pdf_filename = f"report_{safe_a}_{safe_b}.pdf"
                    zf.writestr(pdf_filename, pdf_buffer.getvalue())
                except Exception as exc:
                    logger.error(
                        "Failed to generate PDF for %s ↔ %s: %s", doc_a, doc_b, exc
                    )

            # Fallback JSON per‑pair if PDF generation fails
            safe_a = _sanitise_filename(doc_a)
            safe_b = _sanitise_filename(doc_b)
            fallback = {
                "generated_at": datetime.now().isoformat(),
                "document_a": doc_a,
                "document_b": doc_b,
                "similarity_score": score,
                "threshold": threshold,
                "note": "PDF generation failed; JSON fallback provided.",
            }
            zf.writestr(
                f"report_{safe_a}_{safe_b}.json", json.dumps(fallback, indent=2)
            )

        # Optional CSV summary
        if include_csv:
            try:
                df = pd.DataFrame(csv_rows)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                zf.writestr("summary.csv", csv_bytes)
            except Exception as exc:
                logger.warning("Failed to generate CSV summary: %s", exc)

        # Optional JSON metadata
        if include_json:
            try:
                metadata = {
                    "generated_at": datetime.now().isoformat(),
                    "flags": flags,
                }
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            except Exception as exc:
                logger.warning("Failed to generate JSON metadata: %s", exc)

    return memory_file.getvalue()


def create_batch_incident_zip_archive(
    incidents: list[dict],
    delimiter: str = ",",
    quoting_style: int = csv.QUOTE_MINIMAL,
) -> bytes:
    """Generate in-memory ZIP byte buffer containing incidents_summary.csv, metadata.json, and PDF reports.

    Args:
        incidents: A list of incident dictionaries.
        delimiter: Field delimiter used for ``incidents_summary.csv``.
        quoting_style: The quoting mode passed to CSV export.

    Returns:
        bytes: The in-memory ZIP file content.
    """
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Generate and write incidents_summary.csv
        try:
            csv_bytes = export_incidents_csv_stream(
                incidents, delimiter=delimiter, quoting_style=quoting_style
            )
            zf.writestr("incidents_summary.csv", csv_bytes)
        except Exception as exc:
            logger.error(
                "Failed to generate incidents_summary.csv in bulk export zip: %s", exc
            )

        # 2. Generate and write metadata.json
        try:
            metadata = {
                "generated_at": datetime.now().isoformat(),
                "total_incidents": len(incidents),
                "incidents": incidents,
            }
            metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
            zf.writestr("metadata.json", metadata_bytes)
        except Exception as exc:
            logger.error("Failed to generate metadata.json in bulk export zip: %s", exc)

        # 3. Generate and write PDF report for each incident
        for idx, incident in enumerate(incidents):
            doc_a = incident.get("document_a") or incident.get("doc_a", "")
            doc_b = incident.get("document_b") or incident.get("doc_b", "")

            if not doc_a or not doc_b:
                logger.warning(
                    "Skipping PDF generation for incident at index %d: missing doc_a or doc_b",
                    idx,
                )
                continue

            raw_score = incident.get("similarity_score") or incident.get(
                "similarity", 0.0
            )
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0

            raw_threshold = incident.get("threshold_at_time_of_flag") or incident.get(
                "threshold", 0.59
            )
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError):
                threshold = 0.59

            incident_id = incident.get("incident_id")
            if not incident_id:
                # generate a fallback incident ID
                try:
                    from src.db.incidents import build_incident_id

                    incident_id = build_incident_id(doc_a, doc_b)
                except Exception:
                    incident_id = f"unknown_{idx}"

            safe_a = _sanitise_filename(doc_a)
            safe_b = _sanitise_filename(doc_b)
            pdf_filename = f"report_{incident_id}_{safe_a}_{safe_b}.pdf"

            try:
                pdf_buffer = generate_plagiarism_report(
                    doc_a=doc_a,
                    doc_b=doc_b,
                    overall_similarity=score,
                    threshold=threshold,
                    top_pairs=[],
                    report_title=f"Plagiarism Report: {doc_a} vs {doc_b}",
                )
                zf.writestr(pdf_filename, pdf_buffer.getvalue())
            except Exception as exc:
                logger.error(
                    "Failed to generate PDF for incident %s (%s ↔ %s): %s",
                    incident_id,
                    doc_a,
                    doc_b,
                    exc,
                )

    return memory_file.getvalue()


def create_documents_bulk_zip_archive(filenames: list[str]) -> bytes:
    """Create a downloadable .zip archive containing text content and metadata manifest
    for the specified document filenames from the corpus database.

    Args:
        filenames: List of document filenames to include in the ZIP archive.

    Returns:
        ZIP archive file bytes ready for download.
    """
    from src.db.corpus_db import _connect, get_all_documents, get_document_word_counts
    from src.utils.filename import sanitize_filename

    buffer = io.BytesIO()
    raw_docs = get_all_documents(include_deleted=True)
    all_docs = {}
    for d in raw_docs:
        fn = getattr(d, "filename", None) or (
            d.get("filename") if isinstance(d, dict) else None
        )
        if fn:
            all_docs[fn] = d

    word_counts = get_document_word_counts()
    manifest_rows = []

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for filename in filenames:
            doc_obj = all_docs.get(filename)
            doc_meta = {}
            if doc_obj:
                if hasattr(doc_obj, "model_dump"):
                    doc_meta = doc_obj.model_dump()
                elif hasattr(doc_obj, "__dict__"):
                    doc_meta = doc_obj.__dict__
                elif isinstance(doc_obj, dict):
                    doc_meta = doc_obj

            rows = []
            text_content = ""
            try:
                with _connect() as conn:
                    rows = conn.execute(
                        "SELECT chunk_text FROM chunks WHERE filename = ? ORDER BY chunk_index",
                        (filename,),
                    ).fetchall()
                    text_content = "\n\n".join(r[0] for r in rows if r[0])
            except Exception as exc:
                logger.error(
                    f"Failed to fetch content for document '{filename}': {exc}"
                )

            clean_name = sanitize_filename(filename, fallback="document")
            if not os.path.splitext(clean_name)[1]:
                clean_name += ".txt"

            zip_file.writestr(clean_name, text_content.encode("utf-8"))

            manifest_rows.append(
                {
                    "filename": filename,
                    "exported_as": clean_name,
                    "student_name": doc_meta.get("student_name") or "N/A",
                    "assignment_title": doc_meta.get("assignment_title") or "N/A",
                    "class_section": doc_meta.get("class_section") or "N/A",
                    "word_count": word_counts.get(filename, 0),
                    "chunk_count": len(rows),
                    "upload_date": doc_meta.get("upload_date") or "N/A",
                }
            )

        if manifest_rows:
            manifest_df = pd.DataFrame(manifest_rows)
            manifest_csv = manifest_df.to_csv(index=False).encode("utf-8-sig")
            zip_file.writestr("export_manifest.csv", manifest_csv)

    buffer.seek(0)
    return buffer.getvalue()
