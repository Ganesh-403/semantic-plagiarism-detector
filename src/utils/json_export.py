from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def get_export_timestamp() -> str:
    """Generate an ISO 8601 formatted UTC timestamp string with Z suffix.

    Returns:
        ISO 8601 UTC timestamp string, e.g. "2026-07-31T07:25:00Z".
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_serializer_fallback(obj: Any) -> Any:
    """Custom JSON serializer for NumPy data types, pandas Timestamps, and datetime objects.

    Passed as the ``default=`` callback to :func:`json.dumps` so that
    otherwise non-serializable objects (e.g. ``numpy.int64``, ``numpy.float64``,
    ``datetime``) don't raise an unhandled ``TypeError`` when exporting.

    Args:
        obj: Object instance to serialize.

    Returns:
        JSON-serializable Python native type representation.
    """
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return 0.0 if math.isnan(float(obj)) or math.isinf(float(obj)) else round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def export_similarity_matrix_to_json(
    df: Optional[Union[pd.DataFrame, Any]],
    include_metadata: bool = False,
    indent: Optional[int] = 2,
) -> str:
    """Serializes a similarity matrix DataFrame into a clean JSON string.

    Format (without metadata):
    [
      {
        "document_1": "doc_a",
        "document_2": "doc_b",
        "similarity_score": 0.92
      }
    ]

    Format (with metadata):
    {
      "metadata": {
        "exported_at": "2026-07-31T07:25:00Z",
        "pair_count": 3
      },
      "pairs": [...]
    }

    Includes only unique document pairs (upper triangle of the symmetric matrix, j > i).
    Handles empty, single-document, NaN, and Unicode filename values.

    Args:
        df: Symmetric similarity DataFrame (doc × doc) or None.
        include_metadata: Whether to wrap output in a metadata object with timestamp.
        indent: JSON indentation level (use None for minified JSON).

    Returns:
        JSON formatted string representation of the unique similarity pairs.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        if include_metadata:
            payload = {
                "metadata": {
                    "exported_at": get_export_timestamp(),
                    "pair_count": 0,
                    "documents_count": 0,
                },
                "pairs": [],
            }
            return json.dumps(payload, indent=indent, ensure_ascii=False)
        return "[]"

    doc_names: List[str] = [str(col) for col in df.columns]
    n: int = len(doc_names)
    pairs: List[Dict[str, Union[str, float]]] = []

    for i in range(n):
        for j in range(i + 1, n):
            score = df.iloc[i, j]
            if pd.isna(score) or (isinstance(score, (float, np.floating)) and math.isnan(float(score))):
                score_val = 0.0
            else:
                score_val = round(float(score), 4)

            pairs.append(
                {
                    "document_1": doc_names[i],
                    "document_2": doc_names[j],
                    "similarity_score": score_val,
                }
            )

    if include_metadata:
        output_data = {
            "metadata": {
                "exported_at": get_export_timestamp(),
                "pair_count": len(pairs),
                "documents_count": n,
            },
            "pairs": pairs,
        }
        return json.dumps(output_data, indent=indent, ensure_ascii=False, default=json_serializer_fallback)

    return json.dumps(pairs, indent=indent, ensure_ascii=False, default=json_serializer_fallback)


def export_to_json(
    data: Any,
    include_metadata: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    indent: Optional[int] = 2,
) -> str:
    """Export arbitrary report data structures into a clean JSON string with an exported_at timestamp.

    Issue #1034 & #1250: Adds an `exported_at` ISO 8601 UTC timestamp and supports configurable indent.

    Args:
        data: Core report data (dict, list, DataFrame, or primitive).
        include_metadata: Whether to wrap output with a metadata root object containing exported_at.
        metadata: Optional additional metadata key-values to merge into the root metadata block.
        indent: JSON indentation level (default=2, use None for minified single-line JSON).

    Returns:
        JSON string with UTC timestamp in metadata.exported_at.
    """
    exported_at_timestamp = get_export_timestamp()

    if isinstance(data, pd.DataFrame):
        processed_data = json.loads(
            export_similarity_matrix_to_json(data, include_metadata=False, indent=indent)
        )
    else:
        processed_data = data

    if not include_metadata:
        return json.dumps(
            processed_data,
            indent=indent,
            ensure_ascii=False,
            default=json_serializer_fallback,
        )

    root_metadata: Dict[str, Any] = {
        "exported_at": exported_at_timestamp,
    }

    if isinstance(data, list):
        root_metadata["total_records"] = len(data)
    elif isinstance(data, dict):
        root_metadata["keys_count"] = len(data)
    elif isinstance(data, pd.DataFrame):
        root_metadata["total_records"] = len(processed_data)

    if metadata and isinstance(metadata, dict):
        for k, v in metadata.items():
            if k != "exported_at":
                root_metadata[k] = v

    payload = {
        "metadata": root_metadata,
        "data": processed_data,
    }

    return json.dumps(
        payload,
        indent=indent,
        ensure_ascii=False,
        default=json_serializer_fallback,
    )


def export_report_to_json(
    report_dict: Dict[str, Any],
    custom_metadata: Optional[Dict[str, Any]] = None,
    indent: Optional[int] = 2,
) -> str:
    """Export a comprehensive plagiarism inspection report dictionary to JSON.

    Args:
        report_dict: Main inspection report payload.
        custom_metadata: Optional metadata parameters.
        indent: JSON indentation level (use None for minified JSON).

    Returns:
        JSON string containing metadata.exported_at timestamp root.
    """
    merged_metadata = custom_metadata.copy() if custom_metadata else {}
    if "report_type" not in merged_metadata:
        merged_metadata["report_type"] = "plagiarism_analysis"

    return export_to_json(
        data=report_dict,
        include_metadata=True,
        metadata=merged_metadata,
        indent=indent,
    )


def export_incidents_to_json(
    incidents: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    indent: Optional[int] = 2,
) -> str:
    """Export a list of incident log records into formatted JSON with timestamp.

    Args:
        incidents: List of incident dictionaries.
        session_id: Optional active session identifier.
        indent: JSON indentation level (use None for minified JSON).

    Returns:
        JSON report string with exported_at root metadata.
    """
    metadata = {
        "report_type": "incident_log",
        "total_incidents": len(incidents) if isinstance(incidents, list) else 0,
    }
    if session_id:
        metadata["session_id"] = session_id

    return export_to_json(
        data=incidents,
        include_metadata=True,
        metadata=metadata,
        indent=indent,
    )


def parse_export_json(json_str: str) -> Dict[str, Any]:
    """Parse a serialized JSON report string and validate structure.

    Args:
        json_str: Raw JSON string input.

    Returns:
        Parsed Python dictionary or list structure.
    """
    if not json_str or not isinstance(json_str, str):
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def validate_json_export_schema(export_dict: Any) -> bool:
    """Verify that an exported report dictionary complies with the metadata schema.

    Args:
        export_dict: Python dictionary representation of exported JSON report.

    Returns:
        True if valid schema containing metadata.exported_at, False otherwise.
    """
    if not isinstance(export_dict, dict):
        return False
    if "metadata" not in export_dict or not isinstance(export_dict["metadata"], dict):
        return False
    metadata = export_dict["metadata"]
    return "exported_at" in metadata and isinstance(metadata["exported_at"], str)


def generate_export_checksum(json_str: str) -> str:
    """Compute SHA-256 checksum hex digest of a JSON string for data integrity verification.

    Args:
        json_str: Formatted JSON text string.

    Returns:
        64-character SHA-256 hex string.
    """
    import hashlib
    if not json_str:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def export_batch_reports_to_json(
    reports: List[Dict[str, Any]],
    batch_id: Optional[str] = None,
    indent: Optional[int] = 2,
) -> str:
    """Export multiple analysis reports into a unified batch JSON document.

    Args:
        reports: Collection of individual report objects.
        batch_id: Optional unique batch reference ID.
        indent: JSON indentation level (use None for minified JSON).

    Returns:
        JSON report string with batch metadata and exported_at timestamp.
    """
    metadata: Dict[str, Any] = {
        "report_type": "batch_plagiarism_analysis",
        "batch_size": len(reports) if isinstance(reports, list) else 0,
    }
    if batch_id:
        metadata["batch_id"] = batch_id

    return export_to_json(
        data=reports,
        include_metadata=True,
        metadata=metadata,
        indent=indent,
    )


def export_filtered_similarity_matrix_to_json(
    df: Optional[pd.DataFrame],
    min_similarity_threshold: float = 0.5,
    include_metadata: bool = True,
    indent: Optional[int] = 2,
) -> str:
    """Export similarity matrix pairs that meet or exceed a minimum similarity threshold score.

    Args:
        df: Input similarity matrix DataFrame.
        min_similarity_threshold: Minimum similarity score cutoff (0.0 to 1.0).
        include_metadata: Whether to include root metadata header.
        indent: JSON indentation level (use None for minified JSON).

    Returns:
        JSON string of filtered similarity pairs.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return export_similarity_matrix_to_json(
            None, include_metadata=include_metadata, indent=indent
        )

    doc_names: List[str] = [str(col) for col in df.columns]
    n: int = len(doc_names)
    filtered_pairs: List[Dict[str, Union[str, float]]] = []

    for i in range(n):
        for j in range(i + 1, n):
            score = df.iloc[i, j]
            if not pd.isna(score) and float(score) >= min_similarity_threshold:
                filtered_pairs.append(
                    {
                        "document_1": doc_names[i],
                        "document_2": doc_names[j],
                        "similarity_score": round(float(score), 4),
                    }
                )

    metadata = {
        "min_similarity_threshold": min_similarity_threshold,
        "filtered_pairs_count": len(filtered_pairs),
    }

    return export_to_json(
        data=filtered_pairs,
        include_metadata=include_metadata,
        metadata=metadata,
        indent=indent,
    )


def build_export_schema_definition() -> Dict[str, Any]:
    """Return JSON Schema representation for validating exported metadata reports.

    Returns:
        JSON Schema dictionary.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PlagiarismDetectorExportReport",
        "type": "object",
        "required": ["metadata", "data"],
        "properties": {
            "metadata": {
                "type": "object",
                "required": ["exported_at"],
                "properties": {
                    "exported_at": {
                        "type": "string",
                        "format": "date-time",
                        "description": "ISO 8601 UTC timestamp when report was generated",
                    },
                    "version": {"type": "string"},
                    "report_type": {"type": "string"},
                },
            },
            "data": {
                "description": "Main report payload body",
            },
        },
    }
