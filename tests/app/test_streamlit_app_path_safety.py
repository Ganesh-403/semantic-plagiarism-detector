"""
tests/app/test_streamlit_app_path_safety.py
-------------------------------------------
Unit tests to verify app/streamlit_app.py does not use dangerous sys.path.insert(0) (Issue #2784).
"""

from pathlib import Path


def test_streamlit_app_no_dangerous_sys_path_insert():
    """Verify app/streamlit_app.py does not prepend to sys.path with sys.path.insert(0, ...)."""
    app_path = (
        Path(__file__).resolve().parent.parent.parent / "app" / "streamlit_app.py"
    )
    assert app_path.is_file(), f"File not found: {app_path}"

    content = app_path.read_text(encoding="utf-8")

    # Assert sys.path.insert(0 is not in the file
    assert "sys.path.insert(0" not in content, (
        "Found dangerous 'sys.path.insert(0, ...)' in app/streamlit_app.py. "
        "Use sys.path.append(...) or native module resolution to prevent standard library shadowing."
    )


def test_streamlit_app_syntax_compilation():
    """Verify app/streamlit_app.py compiles without syntax errors."""
    app_path = (
        Path(__file__).resolve().parent.parent.parent / "app" / "streamlit_app.py"
    )
    content = app_path.read_text(encoding="utf-8")
    compiled = compile(content, str(app_path), "exec")
    assert compiled is not None
