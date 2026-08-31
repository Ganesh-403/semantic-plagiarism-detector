"""
src/utils/watchlist_exporter.py
--------------------------------
Export utilities for similarity watchlist data.

Produces JSON, CSV, and HTML reports from watchlist entries and alerts,
providing instructors with clear monitoring dashboards.
"""

from __future__ import annotations

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

from src.core.similarity_watchlist import (
    WatchlistAlert,
    WatchlistEntry,
    WatchlistStatus,
    WatchlistSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_watchlist_json(
    entries: List[WatchlistEntry],
    alerts: List[WatchlistAlert],
    output_path: str,
    indent: int = 2,
) -> str:
    """Write watchlist entries and alerts to a JSON file.

    Args:
        entries: List of watchlist entries.
        alerts: List of watchlist alerts.
        output_path: Destination file path.
        indent: JSON indentation.

    Returns:
        The path written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "entries": [e.to_dict() for e in entries],
        "alerts": [a.to_dict() for a in alerts],
        "exported_at": _now_iso(),
        "entry_count": len(entries),
        "alert_count": len(alerts),
    }
    destination.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Watchlist JSON exported to %s", destination)
    return str(destination)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_entries_csv(
    entries: List[WatchlistEntry],
    output_path: str,
) -> str:
    """Write watchlist entries to a CSV file.

    Args:
        entries: List of watchlist entries.
        output_path: Destination file path.

    Returns:
        The path written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "entry_id", "watchlist_type", "target", "label", "description",
        "status", "similarity_threshold", "created_by", "created_at",
    ]

    with open(destination, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for entry in entries:
            row = entry.to_dict()
            writer.writerow({k: row.get(k, "") for k in columns})

    logger.info("Watchlist entries CSV exported to %s", destination)
    return str(destination)


def export_alerts_csv(
    alerts: List[WatchlistAlert],
    output_path: str,
) -> str:
    """Write watchlist alerts to a CSV file.

    Args:
        alerts: List of watchlist alerts.
        output_path: Destination file path.

    Returns:
        The path written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "alert_id", "entry_id", "triggered_by", "matched_document",
        "similarity_score", "severity", "scan_timestamp", "acknowledged",
        "created_at",
    ]

    with open(destination, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for alert in alerts:
            row = alert.to_dict()
            writer.writerow({k: row.get(k, "") for k in columns})

    logger.info("Watchlist alerts CSV exported to %s", destination)
    return str(destination)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def export_watchlist_markdown(
    entries: List[WatchlistEntry],
    alerts: List[WatchlistAlert],
    summary: WatchlistSummary,
    output_path: str,
    title: str = "Similarity Watchlist Report",
) -> str:
    """Write a watchlist report as Markdown.

    Args:
        entries: List of watchlist entries.
        alerts: List of watchlist alerts.
        summary: Watchlist summary statistics.
        output_path: Destination file path.
        title: Report title.

    Returns:
        The path written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _render_markdown(entries, alerts, summary, title),
        encoding="utf-8",
    )
    logger.info("Watchlist Markdown exported to %s", destination)
    return str(destination)


def _render_markdown(
    entries: List[WatchlistEntry],
    alerts: List[WatchlistAlert],
    summary: WatchlistSummary,
    title: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated:** {_now_iso()}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total entries | {summary.total_entries} |")
    lines.append(f"| Active entries | {summary.active_entries} |")
    lines.append(f"| Total alerts | {summary.total_alerts} |")
    lines.append(f"| Unacknowledged alerts | {summary.unacknowledged_alerts} |")
    lines.append("")

    if summary.entries_by_type:
        lines.append("### Entries by Type")
        lines.append("")
        for etype, count in sorted(summary.entries_by_type.items()):
            lines.append(f"- **{etype}:** {count}")
        lines.append("")

    if summary.alerts_by_severity:
        lines.append("### Alerts by Severity")
        lines.append("")
        for sev, count in sorted(summary.alerts_by_severity.items()):
            lines.append(f"- **{sev}:** {count}")
        lines.append("")

    # Entries
    active = [e for e in entries if e.status == WatchlistStatus.ACTIVE]
    if active:
        lines.append("## Active Watchlist Entries")
        lines.append("")
        lines.append("| ID | Type | Target | Label | Threshold | Created By |")
        lines.append("|----|------|--------|-------|-----------|------------|")
        for e in active:
            threshold_str = (
                f"{e.similarity_threshold:.2f}"
                if e.similarity_threshold > 0
                else "default"
            )
            lines.append(
                f"| {e.entry_id} | {e.watchlist_type.value} "
                f"| {e.target} | {e.label} "
                f"| {threshold_str} | {e.created_by} |"
            )
        lines.append("")

    # Recent alerts
    recent = [a for a in alerts if not a.acknowledged][:20]
    if recent:
        lines.append("## Recent Alerts")
        lines.append("")
        lines.append("| ID | Entry | Matched Doc | Score | Severity | Time |")
        lines.append("|----|-------|------------|-------|----------|------|")
        for a in recent:
            lines.append(
                f"| {a.alert_id} | {a.entry_id} "
                f"| {a.matched_document} "
                f"| {a.similarity_score:.4f} "
                f"| {a.severity} | {a.created_at[:19]} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def export_watchlist_html(
    entries: List[WatchlistEntry],
    alerts: List[WatchlistAlert],
    summary: WatchlistSummary,
    output_path: str,
    title: str = "Similarity Watchlist Dashboard",
) -> str:
    """Write a self-contained HTML watchlist dashboard.

    Args:
        entries: List of watchlist entries.
        alerts: List of watchlist alerts.
        summary: Watchlist summary statistics.
        output_path: Destination file path.
        title: Page title.

    Returns:
        The path written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _render_html(entries, alerts, summary, title),
        encoding="utf-8",
    )
    logger.info("Watchlist HTML exported to %s", destination)
    return str(destination)


def _render_html(
    entries: List[WatchlistEntry],
    alerts: List[WatchlistAlert],
    summary: WatchlistSummary,
    title: str,
) -> str:
    # Entry rows
    entry_rows = ""
    for e in entries:
        status_class = e.status.value
        threshold_str = (
            f"{e.similarity_threshold:.2f}"
            if e.similarity_threshold > 0
            else "default"
        )
        entry_rows += (
            f"<tr>"
            f"<td>{e.entry_id}</td>"
            f"<td>{e.watchlist_type.value}</td>"
            f"<td>{e.target}</td>"
            f"<td>{e.label}</td>"
            f'<td><span class="status-badge status-{status_class}">'
            f"{e.status.value}</span></td>"
            f"<td>{threshold_str}</td>"
            f"<td>{e.created_by}</td>"
            f"<td>{e.created_at[:10] if e.created_at else ''}</td>"
            f"</tr>\n"
        )

    # Alert rows
    alert_rows = ""
    for a in alerts[:50]:
        ack_class = "acknowledged" if a.acknowledged else "unacknowledged"
        alert_rows += (
            f"<tr class='{ack_class}'>"
            f"<td>{a.alert_id}</td>"
            f"<td>{a.entry_id}</td>"
            f"<td>{a.matched_document}</td>"
            f"<td>{a.similarity_score:.4f}</td>"
            f'<td><span class="severity-badge sev-{a.severity.lower()}">'
            f"{a.severity}</span></td>"
            f"<td>{a.scan_timestamp[:19] if a.scan_timestamp else ''}</td>"
            f"<td>{'✓' if a.acknowledged else '—'}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 2rem; color: #1a1a2e;
         background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #1e3a8a; padding-bottom: 0.5rem; }}
  h2 {{ color: #1e3a8a; margin-top: 2rem; }}
  .meta {{ color: #6b7280; font-size: 0.9rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                   gap: 1rem; margin: 1.5rem 0; }}
  .stat-card {{ background: white; border: 1px solid #d1d5db; border-radius: 8px;
                padding: 1.25rem; text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .stat-card .value {{ font-size: 1.75rem; font-weight: 700; color: #1e3a8a; }}
  .stat-card .label {{ font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0;
           background: white; border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ border: 1px solid #e5e7eb; padding: 0.5rem 0.75rem; text-align: left; font-size: 0.9rem; }}
  th {{ background: #1e3a8a; color: white; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f9fafb; }}
  .status-badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
  .status-active {{ background: #d1fae5; color: #065f46; }}
  .status-paused {{ background: #fef3c7; color: #92400e; }}
  .status-resolved {{ background: #e5e7eb; color: #374151; }}
  .severity-badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
  .sev-low {{ background: #d1fae5; color: #065f46; }}
  .sev-medium {{ background: #fef3c7; color: #92400e; }}
  .sev-high {{ background: #fee2e2; color: #991b1b; }}
  .unacknowledged {{ background: #fffbeb; }}
  @media print {{ body {{ padding: 1rem; background: white; }} }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {_now_iso()}</p>

<div class="summary-grid">
  <div class="stat-card"><div class="value">{summary.total_entries}</div><div class="label">Total Entries</div></div>
  <div class="stat-card"><div class="value">{summary.active_entries}</div><div class="label">Active</div></div>
  <div class="stat-card"><div class="value">{summary.total_alerts}</div><div class="label">Total Alerts</div></div>
  <div class="stat-card"><div class="value">{summary.unacknowledged_alerts}</div><div class="label">Unacknowledged</div></div>
</div>

<h2>Watchlist Entries</h2>
<table>
<thead><tr>
  <th>ID</th><th>Type</th><th>Target</th><th>Label</th>
  <th>Status</th><th>Threshold</th><th>Created By</th><th>Created</th>
</tr></thead>
<tbody>
{entry_rows if entry_rows else '<tr><td colspan="8" style="text-align:center;color:#6b7280;">No entries</td></tr>'}
</tbody>
</table>

<h2>Alerts</h2>
<table>
<thead><tr>
  <th>ID</th><th>Entry</th><th>Matched Document</th><th>Score</th>
  <th>Severity</th><th>Scan Time</th><th>Ack'd</th>
</tr></thead>
<tbody>
{alert_rows if alert_rows else '<tr><td colspan="7" style="text-align:center;color:#6b7280;">No alerts</td></tr>'}
</tbody>
</table>

</body>
</html>"""
    return html


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
