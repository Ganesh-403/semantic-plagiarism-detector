# Exporting supplied analysis results

`src.reports.ReportGenerator` writes HTML, PDF, CSV and JSON reports. Pass the
results of an analysis explicitly; an analysis ID alone does not retrieve data.
The library is independent of the current Streamlit export components.

```python
from src.models.report import ReportConfig, ReportFormat, ReportRequest
from src.reports import ReportGenerator

generator = ReportGenerator("reports")
request = ReportRequest(
    analysis_id=analysis_id,
    analysis_data={
        "document_names": document_names,
        "similarity_matrix": similarity_matrix,
        "matches": matches,
    },
    format=ReportFormat.PDF,
    title="Document comparison",
    config=ReportConfig(max_matches=100),
)
response = generator.generate_report(request)
if not response.success:
    raise RuntimeError(response.error)
print(response.file_path)
```

`document_names` must contain unique, nonempty strings. `similarity_matrix` must
be square, symmetric and match their order. All scores must be finite and between
zero and one. Empty lists represent an analysis with no documents; they produce
zero comparisons and zero similarity statistics.

Each optional match contains `source_document`, `target_document` and
`hybrid_score` (or `score`). The documents must be distinct and present in the
analysis. Optional `lexical_score` and `semantic_score` retain their supplied
values; absent values export as blank CSV cells. If `matches` is omitted, the
generator derives unique pairs scoring at least 0.3 from the supplied matrix.
An explicitly empty match list remains empty. Severity boundaries are 0.8 for
high, 0.5 for medium, and 0.3 for low; they are review categories, not verdicts.

Statistics exclude the matrix diagonal. They are computed from all unique
document pairs, while severity counts describe the match list. Caller-provided
statistics do not override these calculations.

`ReportConfig` controls summary statistics, heatmaps, matches, the maximum number
of exported matches and the chart threshold. Setting `include_details=False` on
the request disables heatmaps and matches. The in-memory report retains the full
analysis; the artifact honors the chosen sections. `response.record_count` counts
exported matches. JSON artifacts omit their own size and local filesystem path.

Reports are published with an atomic file replacement. A failed write records a
failed report and leaves an existing artifact intact. Listing and cleanup apply
only to this generator's in-memory registry. Generated files persist, but the
registry does not reload after a process restart. The library creates no HTTP
download route, external delivery integration or background schedule.
