import io
import zipfile

from src.utils.zip_processor import process_zip_file


def make_zip(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return stream.getvalue()


def test_zip_entries_are_sanitized():
    result = process_zip_file(
        make_zip(
            [
                ("folder/<script>alert(1)</script>.pdf", b"pdf"),
                ("folder/report.pdf", b"one"),
                ("other/report.pdf", b"two"),
            ]
        )
    )

    assert list(result) == [
        "alert_1.pdf",
        "report.pdf",
        "report_1.pdf",
    ]


def test_zip_path_traversal_entry_is_skipped():
    result = process_zip_file(
        make_zip(
            [
                ("../../evil.pdf", b"bad"),
                ("safe.pdf", b"good"),
            ]
        )
    )

    assert result == {"safe.pdf": b"good"}
