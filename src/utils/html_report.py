"""html_report.py - HTML report template and rendering functions."""

from typing import Any, Mapping, Sequence

from src.core.config import severity_from_score


def generate_html_report(
    incidents: Sequence[Mapping[str, Any]],
    *,
    min_match_length: int = 0,
) -> str:
    """Generate a clean, standardized HTML report for plagiarism incidents.

    Args:
        incidents: A sequence of incident dictionaries (or mappings).
        min_match_length: If > 0, only incidents with a ``matched_length``
            field greater than or equal to this value are included in the
            report. This ensures the exported HTML matches the UI view when
            the user has applied a min-match-length filter (Issue #2474).

    Returns:
        A string containing the complete HTML document.
    """
    if min_match_length > 0:
        incidents = [
            i
            for i in incidents
            if int(i.get("matched_length", 0) or 0) >= min_match_length
        ]

    if not incidents:
        return "<p>No plagiarism incidents to report.</p>"

    rows = []
    for idx, incident in enumerate(incidents, 1):
        doc_a = incident.get("doc_a", "Unknown")
        doc_b = incident.get("doc_b", "Unknown")
        similarity = incident.get("similarity", 0.0)

        # Use the shared severity logic so the report stays in sync with the
        # UI when the similarity thresholds are adjusted (Issue #2443).
        severity = severity_from_score(similarity)
        color = {
            "High": "#ff4b4b",
            "Medium": "#ffa500",
            "Low": "#21c55d",
        }.get(severity, "#21c55d")

        rows.append(
            f"<tr>"
            f"<td>{idx}</td>"
            f"<td>{doc_a}</td>"
            f"<td>{doc_b}</td>"
            f"<td>{similarity:.1%}</td>"
            f"<td style='color: {color}; font-weight: bold;'>{severity}</td>"
            f"</tr>"
        )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Plagiarism Incident Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f8fafc;
            color: #0f172a;
        }}
        h1 {{
            color: #1e3a8a;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background-color: #ffffff;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f1f5f9;
            color: #475569;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
    </style>
</head>
<body>
    <h1>Plagiarism Incident Report</h1>
    <p>Total flagged pairs: {len(incidents)}</p>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Document A</th>
                <th>Document B</th>
                <th>Similarity</th>
                <th>Severity</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
</body>
</html>"""
    return html_content
