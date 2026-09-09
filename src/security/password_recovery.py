"""Email recovery using expiring token digests and atomic token consumption."""
import hashlib
import logging
import os
import secrets
import smtplib
import ssl
import time
from email.message import EmailMessage

from src.db import auth

logger = logging.getLogger(__name__)
RESET_MESSAGE = "If an eligible account exists, password reset instructions will be emailed."


def mail_is_configured():
    return all(os.getenv(key, "").strip() for key in ("SMTP_SERVER", "SMTP_USERNAME", "SMTP_PASSWORD"))


def send_recovery_email(email, token=None):
    message = EmailMessage()
    message["From"] = os.getenv("SMTP_FROM", os.environ["SMTP_USERNAME"])
    message["To"] = email
    message["Subject"] = "Password reset instructions" if token else "Your password was changed"
    message.set_content(
        "A password reset was requested for your account. Open the application's login page, expand Password recovery, and enter this token. It expires after 15 minutes and works once.\n\n"
        + token + "\n\nIf you did not request this, ignore this email."
        if token else "Your application password was changed. If this was not you, contact your administrator immediately."
    )
    with smtplib.SMTP(os.environ["SMTP_SERVER"], int(os.getenv("SMTP_PORT", "587")), timeout=10) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()
        server.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        server.send_message(message)


def request_password_reset(email):
    """Background operation: never return the token or reveal account existence."""
    email = email.strip().lower()
    now = time.time()
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode()).hexdigest()
    with auth._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM password_reset_tokens WHERE expires_at <= ?", (now,))
        user = connection.execute("SELECT status FROM users WHERE username = ?", (email,)).fetchone()
        previous = connection.execute("SELECT created_at FROM password_reset_tokens WHERE username = ?", (email,)).fetchone()
        if not user or user[0] != "active" or (previous and now - previous[0] < 60):
            return
        connection.execute("DELETE FROM password_reset_tokens WHERE username = ?", (email,))
        connection.execute("INSERT INTO password_reset_tokens VALUES (?, ?, ?, ?)", (digest, email, now, now + 900))
        connection.commit()
    try:
        send_recovery_email(email, token)
    except (OSError, smtplib.SMTPException, ValueError):
        with auth._connect() as connection:
            connection.execute("DELETE FROM password_reset_tokens WHERE token_hash = ?", (digest,))
            connection.commit()
        # SMTP errors can contain addresses or message data; don't log their payloads.
        logger.error("Password recovery email delivery failed; check SMTP configuration.")


def reset_password(token, new_password):
    if not isinstance(token, str) or len(token) > 256:
        raise ValueError("Reset token is invalid or expired.")
    digest = hashlib.sha256(token.encode()).hexdigest()
    with auth._connect() as connection:
        row = connection.execute("SELECT username FROM password_reset_tokens WHERE token_hash = ? AND expires_at > ?", (digest, time.time())).fetchone()
    if row is None:
        raise ValueError("Reset token is invalid or expired.")
    # Consumption and password/history update share one SQLite transaction.
    auth.update_password(row[0], new_password, _reset_token=token)
    if mail_is_configured():
        try:
            send_recovery_email(row[0])
        except (OSError, smtplib.SMTPException, ValueError):
            logger.error("Password change notification could not be delivered.")
