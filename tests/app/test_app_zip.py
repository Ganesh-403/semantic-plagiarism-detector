import io
import zipfile
from src.db.corpus_db import get_all_documents


def test_app_zip_upload_integration(upload_dashboard):
    at, queue = upload_dashboard
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("assignment1.txt", "Semantic analysis measures common ideas within written assignments. " * 6)
        zipped.writestr("assignment2.txt", "Semantic analysis measures common concepts within student assignments. " * 6)
    queue["assignments.zip"] = archive.getvalue()
    at.run()
    assert not at.exception, [(e.message,e.stack_trace) for e in at.exception]
    assert {doc.filename for doc in get_all_documents()} == {"assignment1.txt", "assignment2.txt"}
    assert any(m.label == "Pairs Evaluated" and str(m.value) == "1" for m in at.metric)
