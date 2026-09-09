"""Password failure audit events contain a reason, never submitted passwords."""
import json
import pytest
from src.db import auth


@pytest.mark.parametrize("new_password,old_password,reason", [("Audit-New-Password!984", "wrong", "incorrect_old_password"), ("short", "Audit-Old-Password!984", "complexity_failed")])
def test_failed_event_is_logged(mock_db, new_password, old_password, reason):
    auth.add_user("audit_subject", "Audit-Old-Password!984")
    with pytest.raises(ValueError):
        auth.update_password("audit_subject", new_password, old_password=old_password)
    with auth._connect() as connection:
        rows = connection.execute("SELECT details FROM security_audit_log WHERE event_type = 'password_change_failed' AND username = 'audit_subject'").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == {"reason": reason}
    assert new_password not in rows[0][0] and old_password not in rows[0][0]
