"""Empty and populated bulk exports use the migrated corpus database."""

import io
import csv
import zipfile
from unittest.mock import Mock
import numpy as np
import pytest
from src.db import corpus_db
from src.utils.bulk_export import create_bulk_export_zip, create_documents_bulk_zip_archive


@pytest.mark.parametrize("exporter", [create_bulk_export_zip, create_documents_bulk_zip_archive])
def test_empty_export_is_a_valid_empty_zip(mock_db, exporter):
    progress = Mock()
    result = exporter([], progress_callback=progress)
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        assert archive.namelist() == []
        assert archive.testzip() is None
    progress.assert_not_called()


def test_exported_text_and_manifest_match_the_database(mock_db):
    corpus_db.add_document("report.txt", "report-hash")
    corpus_db.add_chunks([(0, "report.txt", 0, "First paragraph.", np.ones(384, dtype=np.float32))])
    progress = Mock()
    result = create_bulk_export_zip(["report.txt"], progress_callback=progress, preserve_hierarchy=False)
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        assert archive.read("report.txt").decode("utf-8") == "First paragraph."
        rows = list(csv.DictReader(io.StringIO(archive.read("export_manifest.csv").decode("utf-8-sig"))))
        assert len(rows) == 1
        assert rows[0]["filename"] == "report.txt"
        assert rows[0]["chunk_count"] == "1"
    progress.assert_called_once_with(1, 1)
