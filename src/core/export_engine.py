"""LMS-compatible incident export generation and safe file writing."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from src.utils.html_report import generate_html_report

logger = logging.getLogger(__name__)

import csv
import io
import json
import logging
from pathlib import Path

from src.errors import (
    EXPORT_GENERATION_IO_FAILED,
    EXPORT_WRITE_FAILED,
    ExportFailedError,
)

logger = logging.getLogger(__name__)

SAFE_DOWNLOAD_CONTENT_TYPES = {
    "text/csv": "text/csv; charset=utf-8",
    "text/plain": "text/plain; charset=utf-8",
    "application/json": "application/json; charset=utf-8",
    "application/xml": "application/xml; charset=utf-8",
    "application/octet-stream": "application/octet-stream",
}


class LMSExportEngine:
    """Generate LMS-compatible incident exports."""

    @staticmethod
    def generate_incident_html(
        incidents: Sequence[Mapping[str, Any]],
    ) -> str | None:
        """Generate a standardized HTML incident report."""
        if not incidents:
            logger.warning("Attempted to export an empty incident list to HTML.")
            return None

        try:
            return generate_html_report(incidents)
        except Exception as exception:
            logger.error("Failed to format incident data as HTML: %s", exception)
            return None
    def build_download_response(
        data: str | bytes,
        *,
        filename: str,
        content_type: str,
    ) -> tuple[bytes, dict[str, str]]:
        """Build a byte payload with safe download headers for browsers."""
        payload = data.encode("utf-8") if isinstance(data, str) else data

        normalized_type = (
            (content_type or "application/octet-stream")
            .split(";", 1)[0]
            .strip()
            .lower()
        )
        resolved_content_type = SAFE_DOWNLOAD_CONTENT_TYPES.get(
            normalized_type,
            "application/octet-stream",
        )
        safe_filename = (
            str(filename or "download")
            .replace("\r", "")
            .replace("\n", "")
            .replace('"', "")
            .replace(";", "")
        )

        headers = {
            "Content-Type": resolved_content_type,
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": (f'attachment; filename="{safe_filename}"'),
        }
        return payload, headers

    @staticmethod
    def _calculate_severity(sim_score: float) -> str:
        """Classify severity from a numeric similarity score."""
        if sim_score > 0.90:
            return "CRITICAL"
        if sim_score > 0.80:
            return "HIGH"
        return "MODERATE"

    @staticmethod
    def _safe_document_name(value: object) -> str:
        """Return a readable document name for exports."""
        name = str(value or "").strip()
        return name or "Unknown"

    @staticmethod
    def _format_similarity_percent(sim_score: float) -> str:
        """Return a human-readable similarity percentage."""
        return f"{sim_score * 100:.1f}%"

    @staticmethod
    def _wrap_generation_io_error(
        format_name: str,
        exception: OSError,
    ) -> ExportFailedError:
        """Build a consistent generation failure with a safe message."""
        message = EXPORT_GENERATION_IO_FAILED.format(
            format_name=format_name.upper(),
        )
        logger.error(
            "%s export generation failed due to an I/O error: %s",
            format_name.upper(),
            exception,
        )
        return ExportFailedError(message)

    @staticmethod
    def generate_incident_txt(
        incidents: Sequence[Mapping[str, Any]],
    ) -> str | None:
        """Generate a readable plain-text summary of flagged incidents."""
        if not incidents:
            logger.warning("Attempted to export an empty incident list to TXT.")
            return None

        try:
            lines = [
                "SEMANTIC PLAGIARISM INCIDENT REPORT",
                "=" * 38,
                f"Total flagged pairs: {len(incidents)}",
                "",
            ]

            for index, row in enumerate(incidents, start=1):
                sim_score = float(row.get("similarity", 0))
                severity = LMSExportEngine._calculate_severity(sim_score)
                doc_a = LMSExportEngine._safe_document_name(row.get("doc_a"))
                doc_b = LMSExportEngine._safe_document_name(row.get("doc_b"))

                lines.extend(
                    [
                        f"Incident #{index}",
                        "-" * 24,
                        f"Document A: {doc_a}",
                        f"Document B: {doc_b}",
                        (
                            "Similarity: "
                            f"{LMSExportEngine._format_similarity_percent(sim_score)} "
                            f"({sim_score:.4f})"
                        ),
                        f"Severity: {severity}",
                    ]
                )

                matched_length = row.get("matched_length")
                if matched_length not in (None, ""):
                    lines.append(f"Matched length: {matched_length} words")

                matched_text = str(
                    row.get("matched_text") or row.get("matching_text") or ""
                ).strip()
                if matched_text:
                    lines.extend(
                        [
                            "Matching text:",
                            matched_text,
                        ]
                    )

                lines.append("")

            lines.extend(
                [
                    "=" * 38,
                    "End of report",
                    "",
                ]
            )

            report = "\n".join(lines)
        except OSError as exception:
            raise LMSExportEngine._wrap_generation_io_error(
                "TXT",
                exception,
            ) from exception
        except (TypeError, ValueError) as exception:
            logger.error(
                "Failed to format incident data as TXT: %s",
                exception,
            )
            return None

        logger.info(
            "Successfully generated TXT export for %s incidents.",
            len(incidents),
        )
        return report

    @staticmethod
    def generate_incident_csv(
        incidents: Sequence[Mapping[str, Any]],
    ) -> str | None:
        """Generate a standardized LMS-compatible CSV string."""
        if not incidents:
            logger.warning("Attempted to export an empty incident list to CSV.")
            return None

        try:
            output = io.StringIO()
            fieldnames = [
                "Document A",
                "Document B",
                "Similarity Score",
                "Severity Flag",
            ]

            writer = csv.DictWriter(
                output,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()

            for row in incidents:
                sim_score = float(row.get("similarity", 0))
                writer.writerow(
                    {
                        "Document A": LMSExportEngine._safe_document_name(
                            row.get("doc_a")
                        ),
                        "Document B": LMSExportEngine._safe_document_name(
                            row.get("doc_b")
                        ),
                        "Similarity Score": f"{sim_score:.4f}",
                        "Severity Flag": (
                            LMSExportEngine._calculate_severity(sim_score)
                        ),
                    }
                )

            csv_data = output.getvalue()
            output.close()
        except OSError as exception:
            raise LMSExportEngine._wrap_generation_io_error(
                "CSV",
                exception,
            ) from exception
        except (TypeError, ValueError, csv.Error) as exception:
            logger.error(
                "Failed to format incident data as CSV: %s",
                exception,
            )
            return None

        logger.info(
            "Successfully generated LMS CSV export for %s incidents.",
            len(incidents),
        )
        return csv_data

    @staticmethod
    def generate_incident_json(
        incidents: Sequence[Mapping[str, Any]],
    ) -> str | None:
        """Generate a standardized JSON payload for LMS integrations."""
        if not incidents:
            logger.warning("Attempted to export an empty incident list to JSON.")
            return None

        try:
            payload: dict[str, Any] = {
                "metadata": {
                    "total_incidents": len(incidents),
                    "export_format": "LMS_JSON_v1",
                },
                "incidents": [],
            }

            for row in incidents:
                sim_score = float(row.get("similarity", 0))
                payload["incidents"].append(
                    {
                        "document_a": (
                            LMSExportEngine._safe_document_name(row.get("doc_a"))
                        ),
                        "document_b": (
                            LMSExportEngine._safe_document_name(row.get("doc_b"))
                        ),
                        "similarity_score": round(sim_score, 4),
                        "severity_flag": (
                            LMSExportEngine._calculate_severity(sim_score)
                        ),
                    }
                )

            json_data = json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
        except OSError as exception:
            raise LMSExportEngine._wrap_generation_io_error(
                "JSON",
                exception,
            ) from exception
        except (TypeError, ValueError) as exception:
            logger.error(
                "Failed to serialize incident data to JSON: %s",
                exception,
            )
            return None

        logger.info(
            "Successfully generated LMS JSON payload for %s incidents.",
            len(incidents),
        )
        return json_data

    @staticmethod
    def write_export_file(
        data: str | bytes,
        destination: str | Path,
        *,
        format_name: str,
        encoding: str = "utf-8",
    ) -> Path:
        """Write generated export data and wrap standard I/O failures.

        Args:
            data: Text or byte content to write.
            destination: Target output path.
            format_name: Human-readable export format, such as ``CSV``.
            encoding: Encoding used for text exports.

        Returns:
            The resolved output path after a successful write.

        Raises:
            ExportFailedError: When the operating system rejects the write.
            TypeError: When data is neither text nor bytes.
        """
        output_path = Path(destination).expanduser()

        try:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if isinstance(data, bytes):
                output_path.write_bytes(data)
            elif isinstance(data, str):
                output_path.write_text(
                    data,
                    encoding=encoding,
                    newline="",
                )
            else:
                raise TypeError("Export data must be text or bytes.")
        except OSError as exception:
            message = EXPORT_WRITE_FAILED.format(
                format_name=format_name.upper(),
                destination=output_path,
            )
            logger.error(
                "%s export write failed for %s: %s",
                format_name.upper(),
                output_path,
                exception,
            )
            raise ExportFailedError(message) from exception

        logger.info(
            "%s export written successfully to %s.",
            format_name.upper(),
            output_path,
        )
        return output_path.resolve()
