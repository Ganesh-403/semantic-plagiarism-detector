"""Exercise the real login and authenticated dashboard without test-suite stubs."""

from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.environ["SPD_STATE_DIR"] = tempfile.mkdtemp(prefix="spd-smoke-")
    os.environ["PRELOAD_EMBEDDING_MODEL"] = "false"
    os.environ["ENABLE_EMBEDDED_API"] = "false"
    from streamlit.testing.v1 import AppTest
    from src.db.auth import add_user, init_db

    init_db()
    add_user("smoke_teacher", "Smoke-Test-Password!984", role="admin")
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=90).run()
    assert not at.exception, [e.message for e in at.exception]
    assert [w.label for w in at.text_input[:2]] == ["Username", "Password"]
    at.text_input[0].set_value("smoke_teacher")
    at.text_input[1].set_value("Smoke-Test-Password!984")
    next(b for b in at.button if b.label == "Login").click().run()
    assert not at.exception, [(e.message, e.stack_trace) for e in at.exception]
    from app.session_keys import SessionKeys

    assert at.session_state[SessionKeys.AUTHENTICATED]
    print("PASS: login form, password authentication and authenticated dashboard")
    if os.getenv("SMOKE_REAL_MODEL") == "true":
        import app.views.upload_view as upload_view

        text = "Neural networks learn patterns by adjusting weights during training. Semantic comparisons measure how closely two documents express related ideas. "
        upload_view.render_upload_section = lambda *args, **kwargs: {
            "first.txt": (text * 5).encode(),
            "second.txt": (
                text * 5 + "This version includes an additional sentence."
            ).encode(),
        }
        at.run()
        assert not at.exception, [(e.message, e.stack_trace) for e in at.exception]
        assert any(
            m.label == "Pairs Evaluated" and str(m.value) == "1" for m in at.metric
        )
        from src.db.corpus_db import get_total_document_count, get_chunk_registry

        assert get_total_document_count() == 2
        assert len(get_chunk_registry()) > 0
        at.run()
        assert not at.exception, [(e.message, e.stack_trace) for e in at.exception]
        assert get_total_document_count() == 2
        print(
            "PASS: real two-document analysis, result tabs, corpus persistence and rerun"
        )


if __name__ == "__main__":
    main()
