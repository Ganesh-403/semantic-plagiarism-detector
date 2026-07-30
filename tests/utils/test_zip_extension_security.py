import io
import zipfile

from src.utils.zip_processor import process_zip_file


def make_zip(entries):
    stream = io.BytesIO()

    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)

    return stream.getvalue()


def test_zip_rejects_double_extension_executables():
    result = process_zip_file(
        make_zip(
            [
                ("safe.pdf", b"safe"),
                ("document.pdf.exe", b"unsafe"),
                ("notes.txt.cmd", b"unsafe"),
                ("report.docx", b"safe-docx"),
            ]
        )
    )

    assert result == {
        "safe.pdf": b"safe",
        "report.docx": b"safe-docx",
    }


def test_zip_extension_validation_is_case_insensitive():
    result = process_zip_file(
        make_zip(
            [
                ("SAFE.PDF", b"safe"),
                ("EVIL.PDF.EXE", b"unsafe"),
            ]
        )
    )

    assert result == {"SAFE.pdf": b"safe"}
