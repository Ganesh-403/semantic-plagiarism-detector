"""Recovery API uses real SQLite state; only outbound email is mocked."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.db import auth
from src.security import password_recovery as recovery
from src.security.jwt_utils import create_access_token

OLD = "Recovery-Old-Password!984"
NEW = "Recovery-New-Password!985"
EMAIL = "account@example.org"


@pytest.fixture
def recovery_environment(mock_db, monkeypatch):
    for name,value in {"SMTP_SERVER":"mail.example.org", "SMTP_USERNAME":"sender@example.org", "SMTP_PASSWORD":"test-only"}.items():  # pragma: allowlist secret
        monkeypatch.setenv(name,value)
    mail = MagicMock()
    monkeypatch.setattr(recovery,"send_recovery_email",mail)
    auth.add_user(EMAIL, OLD)
    return TestClient(app), mail


def issue(environment):
    client, mail = environment
    response = client.post("/api/v1/auth/forgot-password",json={"email":EMAIL})
    assert response.status_code == 200, response.text
    return mail.call_args.args[1]


def test_forgot_password_response_does_not_reveal_existence(recovery_environment):
    client, mail = recovery_environment
    existing = client.post("/api/v1/auth/forgot-password",json={"email":EMAIL})
    missing = client.post("/api/v1/auth/forgot-password",json={"email":"absent@example.org"})
    assert existing.status_code == missing.status_code == 200
    assert existing.json() == missing.json() == {"message":recovery.RESET_MESSAGE}
    mail.assert_called_once()


def test_reset_password_success_and_single_use(recovery_environment,caplog):
    client,mail = recovery_environment
    old_access = create_access_token(EMAIL,["read"])
    token = issue(recovery_environment)
    with auth._connect() as connection:
        stored = connection.execute("SELECT token_hash FROM password_reset_tokens").fetchone()[0]
    assert stored == hashlib.sha256(token.encode()).hexdigest() and token not in stored
    response = client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":NEW})
    assert response.status_code == 200, response.text
    assert auth.verify_user(EMAIL, NEW)
    assert not auth.verify_user(EMAIL, OLD)
    assert auth.is_token_revoked(old_access)
    assert token not in caplog.text and NEW not in caplog.text
    assert client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":OLD}).status_code == 400
    assert mail.call_args.args == (EMAIL,)


def test_reset_password_invalid_token(recovery_environment):
    client,_ = recovery_environment
    assert client.post("/api/v1/auth/reset-password",json={"token":"invalid", "new_password":NEW}).status_code == 400
    assert auth.verify_user(EMAIL,OLD)


def test_reset_password_deleted_user(recovery_environment):
    client,_ = recovery_environment
    token = issue(recovery_environment)
    auth.delete_user(EMAIL)
    assert client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":NEW}).status_code == 400


def test_expired_token_cannot_change_password(recovery_environment):
    client,_ = recovery_environment
    token = issue(recovery_environment)
    with auth._connect() as connection:
        connection.execute("UPDATE password_reset_tokens SET expires_at = 0")
        connection.commit()
    assert client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":NEW}).status_code == 400
    assert auth.verify_user(EMAIL,OLD)


def test_concurrent_redemption_changes_password_once(recovery_environment):
    token = issue(recovery_environment)
    def redeem():
        try:
            recovery.reset_password(token,NEW)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _:redeem(),range(2))) == [False,True]
    assert auth.verify_user(EMAIL,NEW)


def test_reused_password_rolls_back_token_consumption(recovery_environment):
    client,_ = recovery_environment
    token = issue(recovery_environment)
    assert client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":OLD}).status_code == 400
    assert client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":NEW}).status_code == 200


def test_weak_password_does_not_consume_token(recovery_environment):
    client,_ = recovery_environment
    token = issue(recovery_environment)
    assert client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":"Password123!"}).status_code == 400  # pragma: allowlist secret
    assert client.post("/api/v1/auth/reset-password",json={"token":token,"new_password":NEW}).status_code == 200


def test_unconfigured_mail_returns_honest_error(recovery_environment,monkeypatch):
    client,mail = recovery_environment
    monkeypatch.delenv("SMTP_PASSWORD")
    response = client.post("/api/v1/auth/forgot-password",json={"email":EMAIL})
    assert response.status_code == 503
    mail.assert_not_called()


def test_failed_delivery_removes_token(recovery_environment,caplog):
    client,mail = recovery_environment
    mail.side_effect = OSError("private transport details")
    assert client.post("/api/v1/auth/forgot-password",json={"email":EMAIL}).status_code == 200
    with auth._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM password_reset_tokens").fetchone()[0] == 0
    assert "private transport details" not in caplog.text


def test_per_account_cooldown(recovery_environment):
    client,mail = recovery_environment
    issue(recovery_environment)
    client.post("/api/v1/auth/forgot-password",json={"email":EMAIL})
    mail.assert_called_once()


def test_suspended_account_is_not_recovered(recovery_environment):
    client,mail = recovery_environment
    auth.set_user_active_status(EMAIL,False)
    assert client.post("/api/v1/auth/forgot-password",json={"email":EMAIL}).status_code == 200
    mail.assert_not_called()


def test_password_rotation_invalidates_outstanding_recovery_token(recovery_environment):
    token = issue(recovery_environment)
    auth.update_password(EMAIL,NEW,old_password=OLD)
    with pytest.raises(ValueError,match="invalid or expired"):
        recovery.reset_password(token,"Another-Recovery-Password!986")
