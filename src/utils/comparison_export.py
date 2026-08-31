"""
comparison_export.py
--------------------
Export document comparison results in multiple formats:
HTML (standalone styled report), JSON (structured data), and CSV (tabular).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from src.core.document_comparison_engine import ComparisonResult

logger = logging.getLogger(__name__)


# ── JSON Export ───────────────────────────────────────────────────────────────


def export_comparison_json(result: ComparisonResult, indent: int = 2) -> str:
    """Export comparison result to a JSON string."""
    data = result.to_dict()
    data["_metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "format_version": "1.0",
        "engine": "document_comparison_engine",
    }
    return json.dumps(data, indent=indent, default=str)


# ── CSV Export ────────────────────────────────────────────────────────────────


def export_comparison_csv(result: ComparisonResult) -> str:
    """Export paragraph matches to a CSV string."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "source_paragraph_index", "target_paragraph_index",
        "similarity", "is_exact_match",
        "source_text_preview", "target_text_preview",
    ])
    for m in result.paragraph_matches:
        w.writerow([
            m.source_index, m.target_index, m.similarity, m.is_exact,
            m.source_text[:120] + ("..." if len(m.source_text) > 120 else ""),
            m.target_text[:120] + ("..." if len(m.target_text) > 120 else ""),
        ])
    return buf.getvalue()


# ── HTML Export ───────────────────────────────────────────────────────────────


def _severity_color(severity: str) -> str:
    return {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e", "None": "#6b7280"}.get(severity, "#6b7280")


def export_comparison_html(result: ComparisonResult) -> str:
    """Generate a standalone HTML comparison report."""
    color = _severity_color(result.severity)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build paragraph match rows
    match_rows = ""
    for m in result.paragraph_matches:
        src_preview = m.source_text[:200].replace("<", "&lt;").replace(">", "&gt;")
        tgt_preview = m.target_text[:200].replace("<", "&lt;").replace(">", "&gt;")
        badge = "EXACT" if m.is_exact else f"{m.similarity * 100:.1f}%"
        badge_color = "#ef4444" if m.is_exact else "#f59e0b"
        match_rows += f"""
        <tr>
            <td>{m.source_index}</td>
            <td>{m.target_index}</td>
            <td><span class="badge" style="background:{badge_color}">{badge}</span></td>
            <td class="text-cell">{src_preview}</td>
            <td class="text-cell">{tgt_preview}</td>
        </tr>"""

    # Common words list
    common_words_html = ""
    for word, count in result.word_overlap.top_common[:10]:
        common_words_html += f'<span class="word-tag">{word} ({count})</span> '

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Document Comparison — {result.source_filename} vs {result.target_filename}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;line-height:1.6}}
.container{{max-width:1100px;margin:0 auto}}
h1{{color:#f1f5f9;font-size:1.5rem;margin-bottom:4px}}
.subtitle{{color:#94a3b8;font-size:.9rem;margin-bottom:24px}}
.section{{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:20px}}
.section h3{{color:#38bdf8;font-size:1rem;margin-bottom:12px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.metric{{background:#0f172a;border-radius:8px;padding:14px;text-align:center}}
.metric-val{{font-size:1.5rem;font-weight:700;color:{color}}}
.metric-lbl{{font-size:.75rem;color:#94a3b8;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;padding:8px 10px;color:#94a3b8;border-bottom:1px solid #334155;font-weight:500}}
td{{padding:8px 10px;border-bottom:1px solid #1e293b}}
.text-cell{{max-width:350px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.8rem;color:#cbd5e1}}
.badge{{display:inline-block;padding:2px 8px;border-radius:8px;font-size:.7rem;font-weight:600;color:#fff}}
.word-tag{{display:inline-block;background:#334155;color:#cbd5e1;padding:2px 8px;border-radius:6px;font-size:.75rem;margin:2px}}
.coverage-bar{{height:8px;background:#334155;border-radius:4px;overflow:hidden;margin-top:4px}}
.coverage-fill{{height:100%;background:{color};border-radius:4px;transition:width .3s}}
.footer{{text-align:center;margin-top:24px;color:#64748b;font-size:.75rem}}
</style>
</head>
<body>
<div class="container">
<h1>Document Comparison Report</h1>
<p class="subtitle">{result.source_filename} vs {result.target_filename} — Generated {gen_time}</p>

<div class="section">
<h3>Similarity Overview</h3>
<div class="metrics">
<div class="metric"><div class="metric-val">{result.document_similarity * 100:.1f}%</div><div class="metric-lbl">Document Similarity</div></div>
<div class="metric"><div class="metric-val">{result.max_paragraph_similarity * 100:.1f}%</div><div class="metric-lbl">Max Paragraph Sim</div></div>
<div class="metric"><div class="metric-val">{result.avg_paragraph_similarity * 100:.1f}%</div><div class="metric-lbl">Avg Paragraph Sim</div></div>
<div class="metric"><div class="metric-val">{result.severity}</div><div class="metric-lbl">Severity</div></div>
</div>
</div>

<div class="section">
<h3>Coverage</h3>
<p style="font-size:.85rem;color:#94a3b8;margin-bottom:4px">Source coverage: {result.source_coverage * 100:.1f}%</p>
<div class="coverage-bar"><div class="coverage-fill" style="width:{result.source_coverage * 100}%"></div></div>
<p style="font-size:.85rem;color:#94a3b8;margin:10px 0 4px">Target coverage: {result.target_coverage * 100:.1f}%</p>
<div class="coverage-bar"><div class="coverage-fill" style="width:{result.target_coverage * 100}%"></div></div>
</div>

<div class="section">
<h3>Word Overlap</h3>
<div class="metrics">
<div class="metric"><div class="metric-val">{result.word_overlap.common_words}</div><div class="metric-lbl">Common Words</div></div>
<div class="metric"><div class="metric-val">{result.word_overlap.jaccard_similarity * 100:.1f}%</div><div class="metric-lbl">Jaccard Similarity</div></div>
<div class="metric"><div class="metric-val">{result.word_overlap.unique_source_words}</div><div class="metric-lbl">Source Unique Words</div></div>
<div class="metric"><div class="metric-val">{result.word_overlap.unique_target_words}</div><div class="metric-lbl">Target Unique Words</div></div>
</div>
<p style="font-size:.85rem;color:#94a3b8;margin-top:12px;margin-bottom:6px">Top common words:</p>
{common_words_html}
</div>

<div class="section">
<h3>Paragraph Matches ({result.matched_paragraph_count} of {result.total_paragraphs})</h3>
<table>
<thead><tr><th>Source #</th><th>Target #</th><th>Score</th><th>Source Text</th><th>Target Text</th></tr></thead>
<tbody>{match_rows}</tbody>
</table>
</div>

<div class="footer">Semantic Plagiarism Detector — Document Comparison Engine v1.0</div>
</div>
</body>
</html>"""


# ── Unified dispatcher ────────────────────────────────────────────────────────


def export_comparison(result: ComparisonResult, format: str = "json") -> str:
    """Export a comparison result in the given format.

    Args:
        result: ComparisonResult object.
        format: 'json', 'csv', or 'html'.

    Returns:
        Formatted string.
    """
    fmt = format.lower().strip()
    if fmt == "json":
        return export_comparison_json(result)
    elif fmt == "csv":
        return export_comparison_csv(result)
    elif fmt == "html":
        return export_comparison_html(result)
    raise ValueError(f"Unsupported comparison export format: '{format}'. Supported: json, csv, html.")
