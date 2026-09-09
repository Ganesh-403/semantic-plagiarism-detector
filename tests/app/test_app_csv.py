from src.db.corpus_db import get_chunk_registry


def test_app_csv_upload_integration(upload_dashboard):
    at, queue = upload_dashboard
    queue["assignments.csv"] = ("student_name,essay_response\nAlice," + "Semantic analysis measures common ideas within written assignments. " * 6 + "\nBob," + "Semantic analysis measures common concepts in student assignments. " * 6).encode()
    at.run()
    assert not at.exception, [(e.message,e.stack_trace) for e in at.exception]
    assert get_chunk_registry()
    assert any("Semantic analysis" in chunk.chunk_text for chunk in get_chunk_registry())
