from unittest.mock import MagicMock
from src.security import password_recovery as recovery


def test_recovery_mail_uses_tls_and_does_not_send_password(monkeypatch):
    for key,value in {"SMTP_SERVER":"mail.example.org", "SMTP_USERNAME":"sender@example.org", "SMTP_PASSWORD":"smtp-only-secret", "SMTP_PORT":"587"}.items():  # pragma: allowlist secret
        monkeypatch.setenv(key,value)
    smtp = MagicMock()
    monkeypatch.setattr(recovery.smtplib,"SMTP",smtp)
    recovery.send_recovery_email("recipient@example.org","one-time-token")
    smtp.assert_called_once_with("mail.example.org",587,timeout=10)
    server = smtp.return_value.__enter__.return_value
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("sender@example.org","smtp-only-secret")
    message = server.send_message.call_args.args[0]
    assert message["To"] == "recipient@example.org"
    assert "one-time-token" in message.get_content()
    assert "smtp-only-secret" not in message.get_content()
    methods = [call[0] for call in server.method_calls]
    assert methods.index("starttls") < methods.index("login") < methods.index("send_message")
