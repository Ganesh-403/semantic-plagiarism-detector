from tests.app.settings_helpers import settings_page
from src.db.migrations.auth import AUTH_SCHEMA_VERSION
from src.db.migrations.corpus import CORPUS_SCHEMA_VERSION


def test_db_schema_status_ui_elements(mock_db):
    at = settings_page().run()
    at.button(key="check_db_schema_btn").click().run()
    assert not at.exception
    message = at.session_state["db_schema_status_msg"]
    assert f"Corpus Schema: v{CORPUS_SCHEMA_VERSION}" in message
    assert f"Auth Schema: v{AUTH_SCHEMA_VERSION}" in message
