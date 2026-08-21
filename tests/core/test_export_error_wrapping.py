from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.export_engine import LMSExportEngine
from src.exceptions import ExportFailedError


def test_write_text_export_success(tmp_path):
    destination = tmp_path / "exports" / "incidents.csv"

    result = LMSExportEngine.write_export_file(
        "a,b\n1,2\n",
        destination,
        format_name="CSV",
    )

    assert result == destination.resolve()
    assert destination.read_text(encoding="utf-8") == ("a,b\n1,2\n")


def test_write_binary_export_success(tmp_path):
    destination = tmp_path / "exports" / "report.bin"

    result = LMSExportEngine.write_export_file(
        b"binary-data",
        destination,
        format_name="Binary",
    )

    assert result == destination.resolve()
    assert destination.read_bytes() == b"binary-data"


def test_permission_error_is_wrapped(tmp_path):
    destination = tmp_path / "incidents.csv"
    permission_error = PermissionError("Access denied")

    with patch.object(
        Path,
        "write_text",
        side_effect=permission_error,
    ):
        with pytest.raises(
            ExportFailedError,
            match="Unable to write the CSV export",
        ) as raised:
            LMSExportEngine.write_export_file(
                "content",
                destination,
                format_name="csv",
            )

    assert raised.value.__cause__ is permission_error
    assert "Check the destination permissions" in str(raised.value)


def test_disk_os_error_is_wrapped(tmp_path):
    destination = tmp_path / "incidents.json"
    disk_error = OSError("No space left on device")

    with patch.object(
        Path,
        "write_text",
        side_effect=disk_error,
    ):
        with pytest.raises(ExportFailedError) as raised:
            LMSExportEngine.write_export_file(
                "{}",
                destination,
                format_name="JSON",
            )

    assert raised.value.__cause__ is disk_error
    assert "available disk space" in str(raised.value)


def test_parent_directory_creation_error_is_wrapped(tmp_path):
    destination = tmp_path / "locked" / "incidents.txt"
    mkdir_error = PermissionError("Cannot create directory")

    with patch.object(
        Path,
        "mkdir",
        side_effect=mkdir_error,
    ):
        with pytest.raises(ExportFailedError) as raised:
            LMSExportEngine.write_export_file(
                "report",
                destination,
                format_name="TXT",
            )

    assert raised.value.__cause__ is mkdir_error


def test_invalid_data_type_is_not_misreported_as_io_failure(
    tmp_path,
):
    with pytest.raises(
        TypeError,
        match="Export data must be text or bytes",
    ):
        LMSExportEngine.write_export_file(
            {"not": "serialised"},
            tmp_path / "invalid.txt",
            format_name="TXT",
        )


def test_csv_generation_io_error_is_wrapped():
    incidents = [
        {
            "doc_a": "a.pdf",
            "doc_b": "b.pdf",
            "similarity": 0.9,
        }
    ]
    io_error = OSError("buffer unavailable")

    with patch(
        "src.core.export_engine.io.StringIO",
        side_effect=io_error,
    ):
        with pytest.raises(
            ExportFailedError,
            match="Unable to generate the CSV export",
        ) as raised:
            LMSExportEngine.generate_incident_csv(incidents)

    assert raised.value.__cause__ is io_error
