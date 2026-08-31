"""
src/core/tabular_data_extractor.py
----------------------------------
Tabular Data and CSV Schema Extractor.

Parses CSV and tabular data files to extract column schemas, data types,
and numerical distributions. This allows the system to detect cloned
datasets even when column headers are renamed or numerical values are
linearly transformed (e.g., scaled or shifted).
"""

import csv
import io
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ColumnSchema:
    """Represents the schema and distribution of a single column."""

    name: str
    dtype: str  # 'numeric', 'categorical', 'text'
    unique_count: int
    null_count: int
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "null_count": self.null_count,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }


@dataclass
class TableFingerprint:
    """Represents the structural fingerprint of a tabular dataset."""

    row_count: int
    column_count: int
    columns: List[ColumnSchema] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [c.to_dict() for c in self.columns],
        }


def _infer_dtype(values: List[str]) -> str:
    """Infer the data type of a column based on its values."""
    numeric_count = 0
    for v in values:
        if not v:
            continue
        try:
            float(v)
            numeric_count += 1
        except ValueError:
            pass

    non_empty = [v for v in values if v]
    if not non_empty:
        return "text"

    if numeric_count / len(non_empty) > 0.8:
        return "numeric"

    unique_ratio = len(set(non_empty)) / len(non_empty)
    if unique_ratio < 0.5:
        return "categorical"

    return "text"


def _compute_numeric_stats(values: List[str]) -> Dict[str, float]:
    """Compute mean, std_dev, min, and max for a list of numeric strings."""
    nums = []
    for v in values:
        if not v:
            continue
        try:
            nums.append(float(v))
        except ValueError:
            pass

    if not nums:
        return {}

    n = len(nums)
    mean = sum(nums) / n
    variance = sum((x - mean) ** 2 for x in nums) / n if n > 0 else 0.0
    std_dev = math.sqrt(variance)

    return {
        "mean": round(mean, 4),
        "std_dev": round(std_dev, 4),
        "min_val": round(min(nums), 4),
        "max_val": round(max(nums), 4),
    }


def extract_table_fingerprint(csv_content: str) -> TableFingerprint:
    """Parse CSV content and extract a structural fingerprint.

    Args:
        csv_content: Raw CSV string.

    Returns:
        A TableFingerprint object containing schema and distribution metrics.
    """
    if not csv_content or not isinstance(csv_content, str):
        return TableFingerprint(row_count=0, column_count=0)

    reader = csv.reader(io.StringIO(csv_content))
    try:
        headers = next(reader)
    except StopIteration:
        return TableFingerprint(row_count=0, column_count=0)

    column_data = {h: [] for h in headers}
    row_count = 0

    for row in reader:
        row_count += 1
        for i, val in enumerate(row):
            if i < len(headers):
                column_data[headers[i]].append(val.strip())

    columns = []
    for header, values in column_data.items():
        dtype = _infer_dtype(values)
        unique_count = len(set(v for v in values if v))
        null_count = sum(1 for v in values if not v)

        schema = ColumnSchema(
            name=header, dtype=dtype, unique_count=unique_count, null_count=null_count
        )

        if dtype == "numeric":
            stats = _compute_numeric_stats(values)
            schema.mean = stats.get("mean")
            schema.std_dev = stats.get("std_dev")
            schema.min_val = stats.get("min_val")
            schema.max_val = stats.get("max_val")

        columns.append(schema)

    logger.info(
        "Extracted table fingerprint with %d rows and %d columns.",
        row_count,
        len(columns),
    )

    return TableFingerprint(
        row_count=row_count, column_count=len(columns), columns=columns
    )
