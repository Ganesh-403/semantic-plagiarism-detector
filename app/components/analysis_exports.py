"""Download the current comparison results in machine-readable formats."""
import csv
import io
import json
import streamlit as st


def render_analysis_exports(flags, document_names, threshold):
    if not document_names:
        return
    report = {"documents": list(document_names), "threshold": float(threshold), "matches": flags}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    st.download_button("Download Analysis JSON", payload, "plagiarism_analysis.json", "application/json", key="json_export_button")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["Document A", "Document B", "Similarity"])
    def safe_cell(value):
        text = str(value or "")
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")) else text
    for flag in flags:
        writer.writerow([safe_cell(flag.get("doc_a", flag.get("document_a", ""))), safe_cell(flag.get("doc_b", flag.get("document_b", ""))), flag.get("similarity", flag.get("similarity_score", 0))])
    st.download_button("Download Analysis CSV", buffer.getvalue(), "plagiarism_analysis.csv", "text/csv", key="csv_export_button")
