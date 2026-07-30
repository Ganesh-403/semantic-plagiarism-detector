# Daily Summary Email Setup Guide

This guide explains how to configure the scheduled daily summary email feature for administrators.

## Overview

The daily summary email feature aggregates all plagiarism incidents from the last 24 hours and sends a formatted HTML email to administrators at 5 PM daily. This reduces notification noise during grading seasons while keeping administrators informed.

## Prerequisites

1. Configure SMTP settings in your `.env` file (see `.env.example` for required variables)
2. Ensure you have admin users in the database or set `ADMIN_EMAIL` environment variable

## Environment Variables

Add the following to your `.env` file:

```bash
# SMTP Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password
FROM_EMAIL=your-email@gmail.com
ADMIN_EMAIL=admin@example.com
```

### SMTP Configuration Examples

**Gmail:**
- Server: `smtp.gmail.com`
- Port: `587` (TLS)
- Password: Use an [App Password](http://web.archive.org/web/20260713122540/https://support.google.com/accounts/answer/185833)

**Outlook/Office 365:**
- Server: `smtp.office365.com`
- Port: `587` (TLS)

**SendGrid:**
- Server: `smtp.sendgrid.net`
- Port: `587` (TLS)

## Windows Task Scheduler Setup

### Method 1: Using Task Scheduler GUI

1. Open Task Scheduler (`taskschd.msc`)
2. Click "Create Task" in the right panel
3. **General Tab:**
   - Name: `Daily Plagiarism Summary Email`
   - Description: Sends daily summary of plagiarism incidents to administrators
   - Select "Run whether user is logged on or not"
   - Check "Run with highest privileges"
4. **Triggers Tab:**
   - Click "New"
   - Begin the task: "On a schedule"
   - Settings: "Daily"
   - Start: 5:00:00 PM
   - Repeat every: 1 day
5. **Actions Tab:**
   - Click "New"
   - Action: "Start a program"
   - Program/script: `python.exe`
   - Add arguments: `src/utils/daily_summary_email.py`
   - Start in: `c:\Users\SANJAY\Desktop\semantic-plagiarism-detector`
6. **Conditions Tab:**
   - Uncheck "Start the task only if the computer is on AC power"
   - Uncheck "Stop if the computer switches to battery power"
7. **Settings Tab:**
   - Check "Allow task to be run on demand"
   - Check "Run task as soon as possible after a scheduled start is missed"
8. Click OK and enter your Windows password when prompted

### Method 2: Using PowerShell Script

Run the following PowerShell command as Administrator:

```powershell
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "src/utils/daily_summary_email.py" -WorkingDirectory "c:\Users\SANJAY\Desktop\semantic-plagiarism-detector"
$trigger = New-ScheduledTaskTrigger -Daily -At 5:00PM
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "Daily Plagiarism Summary Email" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Sends daily summary of plagiarism incidents to administrators"
```

## Linux Cron Setup

Add the following cron job to run daily at 5 PM:

```bash
# Edit crontab
crontab -e

# Add this line (adjust paths as needed)
0 17 * * * cd /path/to/semantic-plagiarism-detector && /usr/bin/python3 src/utils/daily_summary_email.py >> /var/log/plagiarism_summary.log 2>&1
```

## Manual Testing

To test the email configuration manually:

```bash
# From the project root directory
python src/utils/daily_summary_email.py
```

## Troubleshooting

### Email not sending:
- Verify SMTP credentials are correct
- Check firewall settings allow SMTP traffic
- Review logs for error messages
- Ensure your SMTP provider allows sending from your IP

### No incidents in email:
- Verify incidents are being created in the database
- Check the `date_flagged` timestamps are within the last 24 hours
- Run the script with `--verbose` flag if available

### Task not running:
- Check Task Scheduler history for errors
- Verify the Python path is correct
- Ensure the working directory path is absolute
- Check that the task is enabled

## Email Format

The daily summary email includes:
- Total count of new incidents
- Breakdown by severity (High, Medium, Low)
- Detailed table for High and Medium severity incidents
- Link to the dashboard for review
- Timestamp of report generation

## Security Notes

- Never commit SMTP credentials to version control
- Use app-specific passwords when available
- Consider using environment-specific configuration files
- Rotate SMTP credentials regularly
- Use TLS/SSL for SMTP connections
