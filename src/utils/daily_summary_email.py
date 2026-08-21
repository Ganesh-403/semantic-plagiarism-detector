"""
daily_summary_email.py
----------------------
Scheduled task to aggregate daily plagiarism incidents and send a summary email to administrators.
Features modular, inline-CSS styled HTML template generation for maximum email client compatibility.
"""

import logging
import os
import re
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from app.theme import COLORS
from src.db.auth import get_all_users
from src.db.incidents import DEFAULT_DB_PATH, get_recent_incidents

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_incidents_last_24h(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieve all incidents flagged in the last 24 hours.

    Args:
        db_path: Path to the SQLite database

    Returns:
        List of incident dictionaries
    """
    cutoff_time = (
        (datetime.now(timezone.utc) - timedelta(hours=24))
        .replace(microsecond=0)
        .isoformat()
    )

    return get_recent_incidents(cutoff_time=cutoff_time, db_path=db_path)


def is_valid_email(email: Optional[str]) -> bool:
    """
    Validate email address format requiring an @ and valid TLD domain.

    Args:
        email: Email string to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not email or not isinstance(email, str):
        return False
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()))


def get_admin_emails() -> List[str]:
    """
    Retrieve email addresses for all admin users with valid email formats.
    Falls back to ADMIN_EMAIL environment variable if no DB admin users have valid emails.

    Returns:
        List of valid admin email addresses
    """
    users = get_all_users()
    admin_emails = []

    for user in users:
        if user.get("role") == "admin":
            email = user.get("email")
            if is_valid_email(email):
                admin_emails.append(email.strip())

    if admin_emails:
        return admin_emails

    env_email = os.getenv("ADMIN_EMAIL")
    if env_email and is_valid_email(env_email):
        return [env_email.strip()]

    return []


def build_incident_row_html(inc: Dict[str, Any]) -> str:
    """
    Build a single HTML table row for an incident.

    Args:
        inc: Dictionary containing incident data.

    Returns:
        str: HTML <tr> element with inline styles.
    """
    doc_a = inc.get("document_a", "Unknown")
    doc_b = inc.get("document_b", "Unknown")
    similarity = inc.get("similarity_score", 0.0)
    date_flagged = inc.get("date_flagged", "Unknown")
    incident_id = inc.get("incident_id")
    app_base_url = os.getenv("APP_BASE_URL", "http://localhost:8501").rstrip("/")

    if incident_id:
        doc_a_display = f'<a href="{app_base_url}/incident/{incident_id}" style="color: #007bff; text-decoration: none;">{doc_a}</a>'
    else:
        doc_a_display = doc_a

    return f"""
    <tr>
        <td style="padding: 12px; border-bottom: 1px solid #eeeeee; color: #333333;">{doc_a_display}</td>
        <td style="padding: 12px; border-bottom: 1px solid #eeeeee; color: #333333;">{doc_b}</td>
        <td style="padding: 12px; border-bottom: 1px solid #eeeeee; color: #333333; font-weight: bold;">{similarity:.2%}</td>
        <td style="padding: 12px; border-bottom: 1px solid #eeeeee; color: #666666;">{date_flagged}</td>
    </tr>
    """


def build_severity_section_html(severity: str, incidents: List[Dict[str, Any]]) -> str:
    """
    Build an HTML section for a specific severity level.

    Args:
        severity: The severity level (e.g., "High", "Medium", "Low").
        incidents: List of incidents matching this severity.

    Returns:
        str: HTML section with a table of incidents.
    """
    color_map = {
        "High": COLORS.get("danger", "#d32f2f"),
        "Medium": COLORS.get("warning", "#f57c00"),
        "Low": COLORS.get("success", "#388e3c"),
    }
    color = color_map.get(severity, "#666666")

    html = f"""
    <h3 style="color: {color}; margin-top: 24px; margin-bottom: 12px; font-size: 18px; border-bottom: 2px solid {color}; padding-bottom: 4px;">
        {severity} Severity Incidents ({len(incidents)})
    </h3>
    """

    if not incidents:
        html += f'<p style="color: #666666; font-style: italic;">No {severity.lower()} severity incidents detected.</p>'
        return html

    html += """
    <table style="width: 100%; border-collapse: collapse; margin-top: 8px; background-color: #ffffff; border-radius: 4px; overflow: hidden;">
        <thead>
            <tr style="background-color: #f5f5f5;">
                <th style="padding: 12px; text-align: left; color: #333333; font-weight: 600; border-bottom: 2px solid #dddddd;">Document A</th>
                <th style="padding: 12px; text-align: left; color: #333333; font-weight: 600; border-bottom: 2px solid #dddddd;">Document B</th>
                <th style="padding: 12px; text-align: left; color: #333333; font-weight: 600; border-bottom: 2px solid #dddddd;">Similarity</th>
                <th style="padding: 12px; text-align: left; color: #333333; font-weight: 600; border-bottom: 2px solid #dddddd;">Date Flagged</th>
            </tr>
        </thead>
        <tbody>
    """

    for inc in incidents:
        html += build_incident_row_html(inc)

    html += "</tbody></table>"
    return html


def build_email_html_body(
    incidents_data: List[Dict[str, Any]],
    total_scans: int,
    footer_note: Optional[str] = None,
) -> str:
    """
    Build a clean, inline-CSS styled HTML email body for the daily summary.

    Issue #1252: Adds optional footer_note parameter to append administrator notes.

    Args:
        incidents_data: List of incident dictionaries.
        total_scans: Total number of scans performed in the period.
        footer_note: Optional custom administrator note to display above the signature/footer.

    Returns:
        str: The fully formatted HTML email body.
    """
    footer_note_html = ""
    if footer_note:
        footer_note_html = f"""
        <div style="margin-top: 20px; padding: 12px; background-color: #eef6ff; border-left: 4px solid #007bff; color: #333333; font-size: 14px; border-radius: 4px;">
            <strong>Note from Administrator:</strong><br>{footer_note}
        </div>
        """

    if not incidents_data:
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta charset="UTF-8">
            <title>Daily Plagiarism Summary</title>
        </head>
        <body>
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; border-radius: 8px;">
            <h2 style="color: #333333; text-align: center;">Daily Plagiarism Summary</h2>
            <p style="color: #666666; text-align: center; font-size: 16px;">
                No new plagiarism incidents detected in the last 24 hours.
            </p>
            {footer_note_html}
            <p style="color: #888888; text-align: center; font-size: 14px; margin-top: 40px;">
                Total scans processed: <strong>{total_scans}</strong>
            </p>
        </div>
        </body>
        </html>
        """

    high_severity = [
        inc for inc in incidents_data if inc.get("severity_rank") == "High"
    ]
    medium_severity = [
        inc for inc in incidents_data if inc.get("severity_rank") == "Medium"
    ]
    low_severity = [inc for inc in incidents_data if inc.get("severity_rank") == "Low"]

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="UTF-8">
        <title>Daily Plagiarism Summary</title>
    </head>
    <body>
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; border-radius: 8px;">
        <h2 style="color: #333333; text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 10px;">Daily Plagiarism Summary</h2>
        <p style="color: #666666; font-size: 14px; text-align: right;">
            Report generated on: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
        </p>

        <div style="background-color: #ffffff; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <p style="font-size: 16px; color: #333333; margin-bottom: 20px;">
                <strong>Total new incidents:</strong> {len(incidents_data)}<br>
                <strong>Total scans processed:</strong> {total_scans}
            </p>

            <p style="font-size: 14px; color: #666666; margin-bottom: 20px;">
                <strong>Severity Breakdown:</strong><br>
                🔴 High: {len(high_severity)} | 🟡 Medium: {len(medium_severity)} | 🟢 Low: {len(low_severity)}
            </p>

            {build_severity_section_html("High", high_severity)}
            {build_severity_section_html("Medium", medium_severity)}
            {build_severity_section_html("Low", low_severity)}
        </div>

        {footer_note_html}

        <p style="color: #888888; text-align: center; font-size: 14px; margin-top: 30px;">
            <a href="{os.getenv("APP_BASE_URL", "http://localhost:8501")}" style="color: #007bff; text-decoration: none;">Review all incidents in the dashboard</a>
        </p>
    </div>
    </body>
    </html>
    """
    return html


def generate_daily_summary_html(stats: Dict[str, Any]) -> str:
    """Generate the HTML content for the daily plagiarism summary email.

    Creates a responsive, email-client-safe HTML email with modern system
    font stacks for optimal readability across all devices and email clients.

    Args:
        stats: Dictionary containing daily statistics:
            - total_scans: int
            - flagged_incidents: int
            - avg_similarity: float
            - top_pairs: List[Dict]

    Returns:
        Complete HTML string ready for email delivery.
    """
    total_scans = stats.get("total_scans", 0)
    flagged_incidents = stats.get("flagged_incidents", 0)
    avg_similarity = stats.get("avg_similarity", 0.0)
    top_pairs = stats.get("top_pairs", [])

    # Format date for the email header
    report_date = datetime.now().strftime("%B %d, %Y")

    # Build top pairs HTML rows
    top_pairs_html = ""
    for pair in top_pairs[:5]:  # Limit to top 5
        doc_a = pair.get("doc_a", "Unknown")
        doc_b = pair.get("doc_b", "Unknown")
        similarity = pair.get("similarity", 0.0)

        top_pairs_html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{doc_a}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0;">{doc_b}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e2e8f0; font-weight: bold; color: #ef4444;">{similarity:.1%}</td>
        </tr>
        """

    if not top_pairs_html:
        top_pairs_html = """
        <tr>
            <td colspan="3" style="padding: 20px; text-align: center; color: #64748b;">
                No high-similarity pairs detected today.
            </td>
        </tr>
        """

    # Issue #2576: Use robust system font stack for modern email clients
    # Replaced "Arial, sans-serif" with comprehensive system font stack
    # This ensures native rendering on macOS, iOS, Windows, Android, and Linux
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="UTF-8">
        <title>Daily Plagiarism Summary</title>
    </head>
    <body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f8fafc;">
            <tr>
                <td align="center" style="padding: 40px 20px;">
                    <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        
                        <!-- Header -->
                        <tr>
                            <td style="background-color: #2563eb; padding: 30px; border-radius: 8px 8px 0 0; text-align: center;">
                                <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                                    📊 Daily Plagiarism Summary
                                </h1>
                                <p style="color: #bfdbfe; margin: 10px 0 0 0; font-size: 14px;">
                                    {report_date}
                                </p>
                            </td>
                        </tr>
                        
                        <!-- Stats Grid -->
                        <tr>
                            <td style="padding: 30px;">
                                <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                                    <tr>
                                        <td width="33%" style="padding: 15px; text-align: center; background-color: #f1f5f9; border-radius: 6px;">
                                            <div style="font-size: 28px; font-weight: bold; color: #0f172a;">{total_scans:,}</div>
                                            <div style="font-size: 12px; color: #64748b; text-transform: uppercase; margin-top: 5px;">Total Scans</div>
                                        </td>
                                        <td width="33%" style="padding: 15px; text-align: center; background-color: #fef2f2; border-radius: 6px;">
                                            <div style="font-size: 28px; font-weight: bold; color: #ef4444;">{flagged_incidents:,}</div>
                                            <div style="font-size: 12px; color: #64748b; text-transform: uppercase; margin-top: 5px;">Flagged</div>
                                        </td>
                                        <td width="33%" style="padding: 15px; text-align: center; background-color: #f0fdf4; border-radius: 6px;">
                                            <div style="font-size: 28px; font-weight: bold; color: #16a34a;">{avg_similarity:.1%}</div>
                                            <div style="font-size: 12px; color: #64748b; text-transform: uppercase; margin-top: 5px;">Avg Similarity</div>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        
                        <!-- Top Pairs Table -->
                        <tr>
                            <td style="padding: 0 30px 30px 30px;">
                                <h2 style="font-size: 18px; color: #0f172a; margin-bottom: 15px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                                    🔝 Top Flagged Pairs
                                </h2>
                                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden;">
                                    <thead>
                                        <tr style="background-color: #f8fafc;">
                                            <th style="padding: 12px; text-align: left; font-size: 12px; color: #64748b; text-transform: uppercase;">Document A</th>
                                            <th style="padding: 12px; text-align: left; font-size: 12px; color: #64748b; text-transform: uppercase;">Document B</th>
                                            <th style="padding: 12px; text-align: left; font-size: 12px; color: #64748b; text-transform: uppercase;">Similarity</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {top_pairs_html}
                                    </tbody>
                                </table>
                            </td>
                        </tr>
                        
                        <!-- Footer -->
                        <tr>
                            <td style="padding: 20px 30px; background-color: #f8fafc; border-radius: 0 0 8px 8px; text-align: center;">
                                <p style="margin: 0; font-size: 12px; color: #94a3b8;">
                                    This is an automated message from the Semantic Plagiarism Detection System.
                                </p>
                            </td>
                        </tr>
                        
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    return html_content


def send_email(
    to_emails: List[str],
    subject: str,
    html_body: str,
    status_callback: Optional[Callable[[bool, str], None]] = None,
    attachment_filename: str = "daily_plagiarism_summary.csv",
    timeout: float = 10.0,
    reply_to: Optional[str] = None,
) -> bool:
    """
    Send an email using SMTP.

    Args:
        to_emails: List of recipient email addresses
        subject: Email subject line
        html_body: HTML formatted email body
        status_callback: Optional callback receiving (success: bool, message: str)
        timeout: Socket connection timeout in seconds (default 10.0)
        reply_to: Optional Reply-To email address header

    Returns:
        True if email sent successfully, False otherwise
    """
    if not to_emails:
        msg = "No recipients configured for daily summary email."
        logger.warning(msg)
        if status_callback:
            status_callback(False, msg)
        return False

    email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    for email in to_emails:
        if not email_pattern.match(email):
            raise ValueError(f"Invalid recipient email address: {email}")

    if reply_to and not email_pattern.match(reply_to):
        raise ValueError(f"Invalid reply-to email address: {reply_to}")

    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL", smtp_username)

    if not all([smtp_server, smtp_username, smtp_password]):
        msg = "SMTP configuration incomplete. Please set SMTP_SERVER, SMTP_USERNAME, and SMTP_PASSWORD."
        logger.error(msg)
        if status_callback:
            status_callback(False, msg)
        return False

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            msg_obj = MIMEMultipart("alternative")
            msg_obj["Subject"] = subject
            msg_obj["From"] = from_email
            msg_obj["To"] = ", ".join(to_emails)

            if reply_to:
                msg_obj["Reply-To"] = reply_to

            html_part = MIMEText(html_body, "html")
            msg_obj.attach(html_part)
            attachment = MIMEApplication(b"", _subtype="csv")
            attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=attachment_filename,
            )
            msg_obj.attach(attachment)

            if smtp_port == 465:
                logger.debug(
                    "Using SMTP_SSL (implicit SSL) on port %d with timeout %.1fs (attempt %d/%d)",
                    smtp_port,
                    timeout,
                    attempt + 1,
                    max_retries + 1,
                )
                with smtplib.SMTP_SSL(
                    smtp_server, smtp_port, timeout=timeout
                ) as server:
                    server.login(smtp_username, smtp_password)
                    server.send_message(msg_obj)
            else:
                logger.debug(
                    "Using SMTP with STARTTLS on port %d with timeout %.1fs (attempt %d/%d)",
                    smtp_port,
                    timeout,
                    attempt + 1,
                    max_retries + 1,
                )
                with smtplib.SMTP(smtp_server, smtp_port, timeout=timeout) as server:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                    server.send_message(msg_obj)

            success_msg = (
                f"Daily summary email sent successfully to {len(to_emails)} recipients."
            )
            logger.info(success_msg)
            if status_callback:
                status_callback(True, success_msg)
            return True

        except (
            ConnectionError,
            TimeoutError,
            smtplib.SMTPConnectError,
            smtplib.SMTPException,
            OSError,
        ) as e:
            # We catch connection/socket/SMTP related issues.
            # OSError covers socket.timeout and low-level socket errors.
            is_last_attempt = attempt == max_retries
            attempt_msg = f"Attempt {attempt + 1} failed: {e}."
            if not is_last_attempt:
                backoff_time = 2**attempt
                logger.warning(f"{attempt_msg} Retrying in {backoff_time}s...")
                time.sleep(backoff_time)
            else:
                error_msg = f"Failed to send daily summary email after {max_retries + 1} attempts: {e}"
                logger.error(error_msg)
                if status_callback:
                    status_callback(False, error_msg)
                return False
        except Exception as e:
            # For non-network/validation exceptions, fail immediately
            error_msg = f"Failed to send daily summary email: {e}"
            logger.error(error_msg)
            if status_callback:
                status_callback(False, error_msg)
            return False


def send_daily_summary(
    subject_prefix: str = "[Plagiarism Alert]",
    footer_note: Optional[str] = None,
    status_callback: Optional[Callable[[bool, str], None]] = None,
    reply_to: Optional[str] = None,
) -> bool:
    """
    Main function to aggregate daily incidents and send summary email.

    Args:
        subject_prefix: Prefix to prepend to the email subject line
        footer_note: Optional custom administrator note to append to the email body
        status_callback: Optional callback receiving (success: bool, message: str)
        reply_to: Optional Reply-To email address header

    Returns:
        True if email sent successfully, False otherwise
    """
    logger.info("Starting daily summary email generation...")

    incidents = get_incidents_last_24h()
    logger.info(f"Found {len(incidents)} incidents in the last 24 hours")

    admin_emails = get_admin_emails()
    logger.info(f"Sending to {len(admin_emails)} admin recipients")

    html_body = build_email_html_body(
        incidents_data=incidents, total_scans=100, footer_note=footer_note
    )

    prefix = f"{subject_prefix} " if subject_prefix else ""
    subject = (
        f"{prefix}Daily Plagiarism Summary - {datetime.now().strftime('%Y-%m-%d')}"
    )
    success = send_email(
        admin_emails,
        subject,
        html_body,
        status_callback=status_callback,
        reply_to=reply_to,
    )

    return success


if __name__ == "__main__":
    success = send_daily_summary()
    exit(0 if success else 1)
