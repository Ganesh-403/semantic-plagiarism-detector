from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")
WARNING_LIST_PATH = Path("src/utils/warning_list.py")

def test_copy_controls_rendered():
    app_source = APP_PATH.read_text(encoding="utf-8")
    assert "render_copy_button(" in app_source
    assert 'copy_label="📋 Copy Snippet"' in app_source

def test_component_keys_are_unique():
    app_source = APP_PATH.read_text(encoding="utf-8")
    # Verify that different button_ids are generated, one for ca and one for cb
    assert 'button_id=f"copy_ca_{rank}"' in app_source
    assert 'button_id=f"copy_cb_{rank}"' in app_source

def test_plain_text_snippets_supplied_to_clipboard():
    app_source = APP_PATH.read_text(encoding="utf-8")
    # Make sure we pass plain 'ca' and 'cb' rather than 'highlighted_ca'
    assert "text_to_copy=ca" in app_source
    assert "text_to_copy=cb" in app_source

def test_render_copy_button_exists():
    warning_list_source = WARNING_LIST_PATH.read_text(encoding="utf-8")
    assert 'def render_copy_button(' in warning_list_source
    assert "document.execCommand('copy')" in warning_list_source
