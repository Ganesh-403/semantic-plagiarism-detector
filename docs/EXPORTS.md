# Bulk Export File Formats & Data Fields

## Overview
(explain the app can export results in three formats — JSON, CSV, and
 Excel — from two sources: the similarity matrix, and the incident log)

## JSON Exports

### Similarity Matrix JSON (src/utils/json_export.py)
- Function: export_similarity_matrix_to_json()
- Sample structure:
  [
    {
      "document_1": "essay_a.pdf",
      "document_2": "essay_b.pdf",
      "similarity_score": 0.92
    }
  ]
- Field definitions: document_1/document_2 (doc names), similarity_score
  (0.0–1.0, rounded to 4 decimals)
- Note: only unique pairs are included (upper triangle, no duplicates/self-pairs)

### LMS Incident JSON (src/core/export_engine.py → generate_incident_json())
- Sample structure:
  {
    "metadata": { "total_incidents": 2, "export_format": "LMS_JSON_v1" },
    "incidents": [
      {
        "document_a": "essay_a.pdf",
        "document_b": "essay_b.pdf",
        "similarity_score": 0.92,
        "severity_flag": "CRITICAL"
      }
    ]
  }
- Field definitions, including severity_flag tiers (CRITICAL > 0.90,
  HIGH > 0.80, MODERATE otherwise)

## CSV Exports

### Similarity Matrix CSV (app/streamlit_app.py, similarity_matrix.csv)
- Plain pandas to_csv() dump of the doc × doc similarity matrix

### Incident Log CSV (src/db/incidents.py → incidents_to_csv())
- Column list (CSV_COLUMNS): Incident ID, Document A, Document B,
  Similarity Score, Threshold at Time of Flag, Severity Rank,
  Review Status, Date Flagged
- Brief description of each column

### LMS Incident CSV (src/core/export_engine.py → generate_incident_csv())
- Column list: Document A, Document B, Similarity Score, Severity Flag

## Excel Exports (src/utils/excel_export.py)
- Function: export_similarity_matrix_to_excel() / build_similarity_workbook()
- Sheet: "Similarity Matrix" — doc × doc grid, values as percentages
- Conditional 3-color scale formatting (white → yellow at threshold → red at 100%)
- Header row/column styled with dark fill + white bold font

## Adding a New Export Format (short code example)
(brief pointer: where to add a new export function and how it's wired
 into the download buttons in app/streamlit_app.py)