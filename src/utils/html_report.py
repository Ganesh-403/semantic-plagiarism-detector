"""html_report.py - HTML report template and rendering functions."""

from typing import Any, Mapping, Sequence


def generate_html_report(incidents: Sequence[Mapping[str, Any]]) -> str:
    """Generate a clean, standardized HTML report for plagiarism incidents."""
    if not incidents:
        return "<p>No plagiarism incidents to report.</p>"

    rows = []
    for idx, incident in enumerate(incidents, 1):
        doc_a = incident.get("doc_a", "Unknown")
        doc_b = incident.get("doc_b", "Unknown")
        similarity = incident.get("similarity", 0.0)

        # Calculate severity rank
        if similarity > 0.90:
            severity = "CRITICAL"
            color = "#ff4b4b"
        elif similarity > 0.80:
            severity = "HIGH"
            color = "#ffa500"
        else:
            severity = "MODERATE"
            color = "#21c55d"

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
