import io
from reportlab.pdfgen import canvas
from src.db.corpus_db import get_all_documents, get_chunk_registry


def test_app_smoke(upload_dashboard):
    at, queue = upload_dashboard
    for name, text in [("first.pdf", "Semantic comparisons measure related ideas across written assignments."), ("second.pdf", "Semantic comparisons find shared concepts across student documents.")]:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer)
        for i in range(6):
            pdf.drawString(40, 750-i*25, text)
        pdf.showPage()
        pdf.save()
        queue[name] = buffer.getvalue()
    at.run()
    assert not at.exception, [(e.message,e.stack_trace) for e in at.exception]
    assert len(get_all_documents()) == 2
    assert get_chunk_registry()
    assert any(m.label == "Pairs Evaluated" and str(m.value) == "1" for m in at.metric)
    at.run()
    assert not at.exception
    assert len(get_all_documents()) == 2
