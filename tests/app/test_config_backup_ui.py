import json
from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_admin_settings_contains_config_backup_download():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'label="📥 Backup Configuration (JSON)"' in source
    assert 'file_name="plagiarism_config_backup.json"' in source
    assert 'mime="application/json"' in source
    assert 'key="backup_config_button"' in source


def test_config_backup_is_inside_admin_settings_block():
    source = APP_PATH.read_text(encoding="utf-8")

    admin_position = source.index('if user_role == "admin":')
    backup_position = source.index('label="📥 Backup Configuration (JSON)"')

    assert backup_position > admin_position


def test_config_backup_serialization_logic():
    # Verify that a dummy config dict serializes cleanly to valid JSON
    dummy_config = {
        "theme": "Dark",
        "threshold": 0.75,
        "class_filter": "CS101",
        "use_chunk_matrix": True,
        "faiss_top_k": 10,
        "ignore_phrases": "test phrase",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "ocr_language": "eng",
        "ocr_dpi": 300,
    }
    json_str = json.dumps(dummy_config, indent=2)
    loaded = json.loads(json_str)
    assert loaded == dummy_config
    assert loaded["theme"] == "Dark"
    assert loaded["threshold"] == 0.75
