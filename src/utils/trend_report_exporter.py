"""
src/utils/trend_report_exporter.py
-----------------------------------
Export utilities for plagiarism trend analysis reports.

Produces JSON, Markdown, and HTML reports from TrendAnalysisResult
objects, with optional chart data for integration with frontend
visualization libraries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.trend_tracker import (
    AlertSeverity,
    PlagiarismTrendTracker,
    TrendAlert,
    TrendAnalysisResult,
    TrendDirection,
    TrendMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_trend_json(
    result: TrendAnalysisResult,
    output_path: str,
    indent: int = 2,
) -> str:
    """Write a trend analysis result to a JSON file.

    Args:
        result: The trend analysis result to export.
        output_path: Destination file path.
        indent: JSON indentation level.

    Returns:
        The path that was written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Trend report JSON exported to %s", destination)
    return str(destination)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def export_trend_markdown(
    result: TrendAnalysisResult,
    output_path: str,
    title: str = "Plagiarism Trend Analysis Report",
) -> str:
    """Write a trend analysis report as Markdown.

    Args:
        result: The trend analysis result to export.
        output_path: Destination file path.
        title: Report title.

    Returns:
        The path that was written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _render_markdown(result, title),
        encoding="utf-8",
    )
    logger.info("Trend report Markdown exported to %s", destination)
    return str(destination)


def _render_markdown(
    result: TrendAnalysisResult,
    title: str,
) -> str:
    """Render a TrendAnalysisResult as Markdown."""
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated:** {result.generated_at}")
    lines.append(f"**Snapshots analyzed:** {result.snapshots_analyzed}")
    lines.append(f"**Time range:** {result.time_range_days} days")
    lines.append(f"**Moving average window:** {result.moving_average_window}")
    lines.append("")

    # Alerts section
    if result.alerts:
        lines.append("## Alerts")
        lines.append("")
        lines.append("| Severity | Title | Metric | Current | Expected | Deviation |")
        lines.append("|----------|-------|--------|---------|----------|-----------|")
        for alert in result.alerts:
            icon = {
                AlertSeverity.CRITICAL: "🔴",
                AlertSeverity.WARNING: "🟡",
                AlertSeverity.INFO: "ℹ️",
            }.get(alert.alert_severity, "")
            lines.append(
                f"| {icon} {alert.alert_severity.value.title()} "
                f"| {alert.title} "
                f"| {alert.metric_name} "
                f"| {alert.current_value:.4f} "
                f"| {alert.expected_value:.4f} "
                f"| {alert.deviation:+.4f} |"
            )
        lines.append("")

    # Metrics section
    lines.append("## Trend Metrics")
    lines.append("")
    for name, tm in result.metrics.items():
        dir_icon = {
            TrendDirection.INCREASING: "📈",
            TrendDirection.DECREASING: "📉",
            TrendDirection.STABLE: "➡️",
        }.get(tm.direction, "")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Direction:** {dir_icon} {tm.direction.value}")
        lines.append(f"- **Slope:** {tm.slope:.6f}")
        lines.append(f"- **R²:** {tm.r_squared:.4f}")
        lines.append(f"- **Volatility:** {tm.volatility:.6f}")
        lines.append(f"- **Latest value:** {tm.latest_value:.4f}")
        lines.append(f"- **Mean:** {tm.mean_value:.4f}")
        lines.append(f"- **Range:** [{tm.min_value:.4f}, {tm.max_value:.4f}]")
        lines.append(f"- **Change rate:** {tm.change_rate:+.2%}")
        lines.append(f"- **Data points:** {tm.data_points}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def export_trend_html(
    result: TrendAnalysisResult,
    output_path: str,
    title: str = "Plagiarism Trend Analysis Report",
) -> str:
    """Write a self-contained HTML trend report with embedded CSS.

    The HTML includes a modern dashboard-style layout suitable for
    printing or sharing via email.

    Args:
        result: The trend analysis result to export.
        output_path: Destination file path.
        title: Page title.

    Returns:
        The path that was written.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _render_html(result, title),
        encoding="utf-8",
    )
    logger.info("Trend report HTML exported to %s", destination)
    return str(destination)


def _render_html(result: TrendAnalysisResult, title: str) -> str:
    """Render a TrendAnalysisResult as self-contained HTML."""
    # Alert cards
    alert_cards = ""
    for alert in result.alerts:
        sev_class = alert.alert_severity.value
        sev_icon = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.WARNING: "🟡",
            AlertSeverity.INFO: "ℹ️",
        }.get(alert.alert_severity, "")
        alert_cards += (
            f'<div class="alert-card alert-{sev_class}">'
            f'<span class="alert-icon">{sev_icon}</span>'
            f'<div class="alert-content">'
            f'<strong>{alert.title}</strong>'
            f'<p>{alert.description}</p>'
            f'<small>Metric: {alert.metric_name} | '
            f'Current: {alert.current_value:.4f} | '
            f'Expected: {alert.expected_value:.4f}</small>'
            f'</div></div>\n'
        )

    # Metric rows
    metric_rows = ""
    dir_icons = {
        TrendDirection.INCREASING: "📈",
        TrendDirection.DECREASING: "📉",
        TrendDirection.STABLE: "➡️",
    }
    for name, tm in result.metrics.items():
        icon = dir_icons.get(tm.direction, "")
        change_class = "positive" if tm.change_rate > 0 else "negative" if tm.change_rate < 0 else "neutral"
        metric_rows += (
            f"<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{icon} {tm.direction.value}</td>"
            f"<td>{tm.slope:.6f}</td>"
            f"<td>{tm.r_squared:.4f}</td>"
            f"<td>{tm.volatility:.4f}</td>"
            f"<td>{tm.latest_value:.4f}</td>"
            f"<td>{tm.mean_value:.4f}</td>"
            f'<td class="{change_class}">{tm.change_rate:+.2%}</td>'
            f"</tr>\n"
        )

    # Chart data (JSON for frontend consumption)
    chart_data = {}
    for name, tm in result.metrics.items():
        chart_data[name] = {
            "timestamps": tm.timestamps,
            "values": tm.values,
            "moving_average": tm.moving_average,
        }

    alert_summary = {
        "critical": len(result.critical_alerts),
        "warning": len(result.warning_alerts),
        "info": len(result.alerts) - len(result.critical_alerts) - len(result.warning_alerts),
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem; color: #1a1a2e;
         background: #f8f9fa; line-height: 1.6; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #1e3a8a; padding-bottom: 0.5rem; }}
  h2 {{ color: #1e3a8a; margin-top: 2rem; }}
  .meta {{ color: #6b7280; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                   gap: 1rem; margin: 1.5rem 0; }}
  .stat-card {{ background: white; border: 1px solid #d1d5db; border-radius: 8px;
                padding: 1.25rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .stat-card .value {{ font-size: 1.75rem; font-weight: 700; color: #1e3a8a; }}
  .stat-card .label {{ font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0;
           background: white; border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ border: 1px solid #e5e7eb; padding: 0.6rem 0.8rem; text-align: left; }}
  th {{ background: #1e3a8a; color: white; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f9fafb; }}
  .positive {{ color: #059669; font-weight: 600; }}
  .negative {{ color: #dc2626; font-weight: 600; }}
  .neutral {{ color: #6b7280; }}
  .alert-card {{ display: flex; align-items: flex-start; gap: 0.75rem;
                 padding: 1rem; margin: 0.75rem 0; border-radius: 8px;
                 background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .alert-critical {{ border-left: 4px solid #dc2626; }}
  .alert-warning {{ border-left: 4px solid #d97706; }}
  .alert-info {{ border-left: 4px solid #2563eb; }}
  .alert-icon {{ font-size: 1.25rem; }}
  .alert-content strong {{ display: block; margin-bottom: 0.25rem; }}
  .alert-content p {{ margin: 0.25rem 0; color: #374151; }}
  .alert-content small {{ color: #6b7280; }}
  @media print {{ body {{ padding: 1rem; background: white; }} .stat-card {{ box-shadow: none; }} }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {result.generated_at} | {result.snapshots_analyzed} snapshots over {result.time_range_days} days</p>

<div class="summary-grid">
  <div class="stat-card"><div class="value">{result.snapshots_analyzed}</div><div class="label">Snapshots</div></div>
  <div class="stat-card"><div class="value">{result.time_range_days}</div><div class="label">Days Tracked</div></div>
  <div class="stat-card"><div class="value">{alert_summary['critical']}</div><div class="label">Critical Alerts</div></div>
  <div class="stat-card"><div class="value">{alert_summary['warning']}</div><div class="label">Warnings</div></div>
</div>

<h2>Alerts</h2>
{alert_cards if alert_cards else '<p>✅ No active alerts.</p>'}

<h2>Trend Metrics</h2>
<table>
<thead>
<tr>
  <th>Metric</th><th>Direction</th><th>Slope</th><th>R²</th>
  <th>Volatility</th><th>Latest</th><th>Mean</th><th>Change</th>
</tr>
</thead>
<tbody>
{metric_rows}
</tbody>
</table>

<script type="application/json" id="chart-data">
{json.dumps(chart_data, indent=2)}
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Chart data helper
# ---------------------------------------------------------------------------

def get_chart_data(result: TrendAnalysisResult) -> Dict[str, Any]:
    """Extract chart-ready data from a trend analysis result.

    Returns a dict suitable for consumption by Chart.js, Plotly, or
    similar visualization libraries.

    Args:
        result: The trend analysis result.

    Returns:
        Dict mapping metric names to {timestamps, values, moving_average}.
    """
    chart_data: Dict[str, Any] = {}
    for name, tm in result.metrics.items():
        chart_data[name] = {
            "timestamps": tm.timestamps,
            "values": tm.values,
            "moving_average": tm.moving_average,
            "direction": tm.direction.value,
            "latest": tm.latest_value,
        }
    return chart_data
