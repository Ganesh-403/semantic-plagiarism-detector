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

## Installed package audit

The audit enumerates every installed distribution and pins the complete inventory
before querying advisories. `--no-deps --disable-pip` avoids resolving a different
environment; transitive installed packages are included explicitly, and missing
results fail the check. The JSON artifact preserves both installed and audited
versions. Only official PyTorch `+cpu` builds map to the corresponding public
release because PyPI does not index the CPU suffix. Other local builds are not
silently mapped or exempted.

Runtime requirements now require `setuptools>=83.0.0`, the fixed version for
[PYSEC-2026-3447](https://github.com/pypa/advisory-database/blob/main/vulns/setuptools/PYSEC-2026-3447.yaml).
The first Linux run exposed the older runner package and the CPU-wheel lookup
failure; neither is addressed by a vulnerability allowlist.

## Runtime licenses

The Windows/Python 3.13 runtime closure contains 160 installed distributions and
passes the repository's GPL/AGPL/LGPL deny-list check. Development-only tools are
not part of the deployed runtime. Linux CI produces its own complete inventory;
platform-specific wheels and transitive dependencies can differ. The Linux license
gate also passed for PR #4328 at commit `af5f0b42`.

Two distributions publish incomplete license metadata. Their upstream license
files state MIT terms:

- `streamlit-autorefresh==1.0.1`: [upstream license](https://github.com/kmcgrady/streamlit-autorefresh/blob/main/LICENSE).
- `streamlit-plotly-events==0.0.6`: [upstream license](https://github.com/null-jones/streamlit-plotly-events/blob/master/LICENSE).

The runtime no longer requires PyMuPDF, EbookLib or deep-translator. The replacement
PDF adapter uses pypdf, PDFium and ReportLab; EPUB parsing uses bounded ZIP/XML
reading; translation providers use explicit HTTP requests with timeouts. License
checks continue to inspect transitive runtime dependencies.
