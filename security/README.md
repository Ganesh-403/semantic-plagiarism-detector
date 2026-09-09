# Dependency and license review

Reviewed on 2026-09-09. CI audits the installed runtime, retains complete reports
as workflow artifacts, and fails on new or unassessed vulnerabilities.

## NLTK applicability assessment

NLTK 3.10.3 has an unresolved finding,
[GHSA-8mgp-746c-j5xp](https://github.com/advisories/GHSA-8mgp-746c-j5xp), also reported
as PYSEC-2026-3740. The advisory concerns caller-controlled model import/export
paths in six artifact APIs. This application uses text inference and fixed-name
language assets and does not call those six APIs or accept NLTK model paths from
users.

`dependency-assessments.json` records the exact version, advisory, reasoning,
review date, expiry and hashes of every Python file mentioning NLTK in `src`,
`app` or `scripts`. `scripts/audit_dependencies.py` retains the original finding
and rejects the assessment if the reviewed source changes, the version changes,
a patched release becomes available, the assessment expires, or an audit is
incomplete. Re-review is required by **2026-10-09**. This is an applicability
assessment, not a claim that NLTK is patched.

## Runtime licenses

The Windows/Python 3.13 runtime closure contains 160 installed distributions and
passes the repository's GPL/AGPL/LGPL deny-list check. Development-only tools are
not part of the deployed runtime. Linux CI produces its own complete inventory;
platform-specific wheels and transitive dependencies can differ.

Two distributions publish incomplete license metadata. Their upstream license
files state MIT terms:

- `streamlit-autorefresh==1.0.1`: [upstream license](https://github.com/kmcgrady/streamlit-autorefresh/blob/main/LICENSE).
- `streamlit-plotly-events==0.0.6`: [upstream license](https://github.com/null-jones/streamlit-plotly-events/blob/master/LICENSE).

The runtime no longer requires PyMuPDF, EbookLib or deep-translator. The replacement
PDF adapter uses pypdf, PDFium and ReportLab; EPUB parsing uses bounded ZIP/XML
reading; translation providers use explicit HTTP requests with timeouts. License
checks continue to inspect transitive runtime dependencies.
