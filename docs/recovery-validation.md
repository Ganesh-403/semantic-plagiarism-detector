# Recovery validation

[PR #4328](https://github.com/Ganesh-403/semantic-plagiarism-detector/pull/4328)
remains in draft because overall coverage is below the required 85% gate.
The full Linux test matrix passes at `b64e71a8`; changed-line coverage,
dependency, license, security and startup checks also pass.

## Complete Linux suite at `b64e71a8`

[CI run 34359175413](https://github.com/Ganesh-403/semantic-plagiarism-detector/actions/runs/34359175413),
2026-09-09, two pytest workers with `--dist=loadfile`, covering both `src` and `app`:

| Python | Passed | Failed | Errors | Skipped | Xfailed | Duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.11 | 9,869 | 0 | 0 | 24 | 1 | 249.21 s |
| 3.12 | 9,869 | 0 | 0 | 24 | 1 | 323.28 s |
| 3.13 | 9,869 | 0 | 0 | 24 | 1 | 300.51 s |

All three versions pass Streamlit and API startup smoke checks and the **90%
changed-line coverage gate**. Each reports approximately **60% combined line and
branch coverage**, below the unchanged **85% overall gate**. That gate is the only
failing CI step. Python 3.13 covers 40,401 of 65,209 executable lines (61.96%).
No source exclusions or reduced thresholds were used to reach these results.

The earlier Windows run at `af5f0b42` passed 9,849 tests with two parser timing
failures under coverage. PDF reader reuse and uninstrumented timing subprocesses
resolve these failures; correctness assertions still run under coverage and the
original two-second DOCX / three-second PDF budgets remain. The 200-page PDF
production benchmark improved from 16.79 to 1.71 seconds locally.

Linux-only failures were fixed by normalizing Windows path spellings independently
of the host OS, isolating Windows test adapters, covering both directory iteration
implementations and stabilizing constant-feature numerical calculations.

## Additional search and comparison regressions

Subsequent tests exercise the production document search index and comparison
engine, including real Streamlit views. They repair stale entries after reindexing,
filtering before result limits, invalid vectors, hybrid search argument ordering,
filter serialization, widget reruns, pagination and HTML escaping of snippets.
The standalone search view offers full-text search; it no longer invents random
query embeddings. Its API still accepts real, compatible semantic vectors.

Comparison scores are bounded and symmetric for repeated phrases, the results
view renders once per comparison, and high similarity is labeled for review.
The view explicitly identifies its word-frequency semantic proxy.

The first 41 checks pass both with and without coverage. The focused coverage run
measures 92.04% of lines in `Doc_Search_Filtering.py` and 99.66% in
`advanced_comparison_engine.py`. These are module measurements, not a claim that
the project-wide gate passes. All **47** checks pass locally after adding date-filter,
saved-search and missing-encoder views; consult the PR's latest CI results for the head.

At `bd3d9434`, Python 3.12 and 3.13 each pass **9,916 tests** and the changed-line
gate. Overall coverage rises to approximately **61% combined** (63.07% of lines
on Python 3.13), still below 85%. Python 3.11 passes 9,914 with two failures caused
by tests contacting a live translation provider that returned HTTP 429. Those
tests now supply responses only at the HTTP boundary and assert both cache behavior
and rate-limit fallback. The 76-test translation module passes locally; its final
five HTTP/cache checks and pre-commit also pass. The latest-head matrix must verify
this test isolation change before relying on its final counts.

Trend-chart regressions add **14 passing checks**, with **96.53% line coverage and
98.39% branch coverage** for `src/visualization/trend_charts.py`. Tests inspect real
Plotly traces, decode generated PNGs and verify figures close when saving fails.
Forecast values now align with their timestamps, nonpositive result limits return
empty charts, and invalid moving-average periods raise a clear error.

At `8c7abce5`, the test runner, both startup smoke tests and changed-line coverage
pass on Python 3.11, 3.12 and 3.13. Python 3.13 reports **9,931 passed**, 24 skipped
and one expected failure. The overall gate still fails at **61% combined**; all
other workflows pass. The preceding revision exposed intermittent contention in
a SQLite stress test whose workers bypassed the production connection factory.
The revised test uses production connections and retains 20 workers and 1,000
writes, checks every worker's persisted rows, concurrent readers and database
integrity. Both database checks pass locally under coverage.

Pattern-recognition regressions add **24 passing checks**, measuring **99.41% line
and 94.09% branch coverage** of that module. They exercise all four detectors,
rescans, a real random forest, evolution forecasts and the Streamlit controls.
Detection toggles now affect results, rescans preserve identity, self-similarity
is excluded from risk features, and NumPy score arrays no longer crash the view.
The dashboard no longer trains a model from its own fabricated labels; heuristic
scores are explicitly uncalibrated. The combined pattern/database group passes
all 26 checks. These module results do not satisfy the overall coverage gate.

At `4c43432b`, Python 3.11 and 3.13 each pass **9,956 tests**. Python 3.12 passes
9,955 with one failure: corruption tests leave a mocked Redis client attached to
the singleton, causing a later real compression round-trip to read corrupt data.
This failure also reproduces locally when those two test files run in that order.
Overall coverage rises to **62% combined**; changed-line coverage and both startup
smokes pass on all three versions. All other workflows pass.

Redis state is now isolated per test using the real cache's no-Redis mode;
connection tests retain their own transport setup. All **101 cache checks pass**
in the previously failing order, including cold initialization and corruption
fallback. A delete test now creates its own fallback entry, and six unconditional
assertions now verify actual availability, lookup and hit-rate results. Secret
baseline changes update existing line locations only.

At `5ba668d3`, Python 3.12 and 3.13 each pass **10,002 tests**; Python 3.11 passes
10,001 with one timer microbenchmark failure. All startup, changed-line coverage,
dependency, security, license and lint checks pass. Overall coverage reaches
**63% combined**, still below 85%. The timer benchmark measured coverage tracing
overhead against its 0.1 ms production budget. Its timing-only cases now execute
in clean child processes, retaining every original budget and iteration count.
All **77 timer/benchmark checks pass** when the parent runs under coverage.

Analysis logging adapters now use the shared SQLite connection factory for WAL,
15-second busy timeout, normal synchronization and foreign-key enforcement. This
also prevents federation signatures from referencing unregistered institutions.
Reviewer and patchwriting repository classes now persist records in SQLite and
return independent snapshots; their former lists lost all records on restart.
All **73 adapter/contention checks pass under coverage**, measuring **100% line
coverage in all 34 changed adapters**. Tests read actual stored values, preserve
Unicode and quoted identifiers, check rollback and closed connections, reject
invalid writes, and reload repositories from their files. These focused results
do not replace the latest full-matrix coverage measurement.

The legacy `src.reports` package now imports and accepts explicit analysis results
through `ReportRequest.analysis_data`. It validates matrices and matches, derives
statistics from unique document pairs and exports populated, completed snapshots.
It no longer fabricates document names, similarity scores or processing times.
Display limits apply to exports; failed publication preserves existing files and
sets the correct report's failure state. HTML text is escaped, CSV names are kept
literal and PDF tables paginate. Deletion is restricted to the output directory.
All **44 artifact/lifecycle checks pass under coverage**. They measure **100% line
coverage** of the reports package and new model, and exercise actual PDF, PNG,
HTML, CSV and JSON artifacts. See the [report API guide](report-exports.md).

## Runtime, dependency and quality checks

- Fresh Streamlit process: password login and authenticated dashboard pass.
- Real `all-MiniLM-L6-v2` model: two-document analysis, result rendering, corpus
  persistence and rerun without duplicate documents pass. Only upload bytes are
  injected; parsing, embeddings and FAISS run normally.
- Fresh API process: password login, signed-token corpus read and unauthorized
  access rejection pass.
- Linux dependency audit, dependency update, security scan, license, memory leak,
  syntax/Ruff/YAML and Markdown jobs pass at `b64e71a8`.
- The complete installed inventory audit preserves CPU-wheel versions while using
  the matching public PyTorch version for advisory lookup. Setuptools must be 83+.
- The audit retains one unpatched NLTK finding with a reviewed, source-hash-bound
  not-affected assessment expiring on 2026-10-09. Zero findings are blocking;
  this does not mean NLTK has been patched.
- The 160-package local runtime closure passes the existing GPL/AGPL/LGPL gate;
  Linux also passes and uploads its inventory. See the
  [dependency review](../security/README.md).
- Local `pip check`, pre-commit including secret detection, and Markdown checks pass.
- The public hosted site displayed “Error running app” when inspected on 2026-09-09.
  It has not been redeployed with this follow-up.

## Test integrity and scope

The merged PR #4327 baseline was **677 failed, 8,982 passed, 25 skipped, 1 xfailed
and 25 errors**, with **52.26% line coverage** across `src` and `app`.

This follow-up replaces tests of copied stand-ins with production calls and real
Streamlit views. A duplicate normalization suite was consolidated. Constant-return
padding classes were removed from six modules; real implementations remain.
Five legacy utility exclusions were removed after repairing their APIs and tests;
remaining exclusions are visible in `pytest.ini`. Passing counts alone are not a
coverage claim.

New regression coverage includes authentication recovery and password rotation,
account suspension/deletion, parsing and exports, cancellation, soft deletion,
document revision lifecycle and storage, analytics, dashboard navigation, search
and comparison. Five standalone dashboards explicitly label illustrative data and
retain it across navigation; they do not analyze uploaded documents.

Container deployment and the updated public hosted app still require verification.
The restored `src.reports` library is separate from the active Streamlit export
paths. Its report registry is local to the generator instance; it does not provide
an HTTP download endpoint or durable job scheduling.
For configuration, backups, SMTP and Cloud reboot/redeployment, use the
[Streamlit redeployment guide](streamlit-redeploy.md).
