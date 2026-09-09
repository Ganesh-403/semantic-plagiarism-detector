# Recovery validation

The follow-up is still in progress in [PR #4328](https://github.com/Ganesh-403/semantic-plagiarism-detector/pull/4328).
Startup and license checks pass on Linux. Overall coverage remains below the
required gate; this is not a fully verified release.

## Complete suite at `af5f0b42`

Windows, Python 3.13.7, 2026-09-09, application source held unchanged for the run:

```console
python -m pytest -n 2 --dist=loadfile --cov=src --cov=app --junitxml=suite.xml --cov-report=xml --cov-report=term tests
```

Result: **9,849 passed, 2 failed, 24 skipped, 1 xfailed, 0 errors in 1,242.77 seconds**.
The failures were parser timing assertions under coverage: DOCX took 2.68 seconds
against a two-second limit, and PDF took 5.34 seconds against a three-second limit.
The earlier Streamlit three-second timeouts were resolved and pass in this run.

The snapshot measured **61.84% line coverage** (40,350 / 65,246 lines),
**51.72% branch coverage** (8,981 / 17,366 branches), and **59.71% combined
coverage**, below the retained **85%** gate. Changed-line coverage was **89.19%**
(1,848 / 2,072 lines), below the retained **90%** gate. No source modules were
excluded to raise these numbers.

The initial Linux matrix at the same commit reported 14–17 failures depending on
Python version, with 9,834–9,837 passing tests. The failures concern cross-platform
cache paths, Windows-only test adapters, global path mocks, directory enumeration,
and numerical precision in constant-feature change-point detection. All three
versions passed both Streamlit and API startup smoke checks. Linux syntax/Ruff/YAML,
Markdown, license, and memory-leak checks passed.

## Subsequent repairs being validated

- Normalize cache path separators independently of the host OS and use stable
  summation/constant-feature handling for change-point detection.
- Isolate simulated Windows adapters in tests and cover both Python directory
  enumeration implementations.
- Reuse one PDF reader across all pages. A production-process benchmark fell from
  16.79 to 1.71 seconds for the same 200-page PDF. Benchmark assertions retain
  their original two-/three-second limits in an uninstrumented subprocess, while
  extraction correctness remains covered in the parent test.
- Preserve table rows without duplicate extraction and verify a reader opens once.
- Exercise the recovery form with real token creation, password policy validation,
  atomic consumption and replay rejection; replace only the outbound email boundary.
- Audit the full pinned installed inventory, preserving CPU-wheel versions while
  looking up the matching public PyTorch release. Require fixed setuptools 83+.

The first post-repair group passed 280 checks with four outdated PDF-page fixture
failures; those fixtures were corrected, and the OCR/recovery UI group then passed
all 14 checks. New audit-inventory checks also pass. These targeted results do not
replace a complete run at the updated head.

## Runtime, dependency and quality checks

- Fresh Streamlit process: password login and the authenticated dashboard pass.
- Real embedding model (`all-MiniLM-L6-v2`): two-document analysis, result rendering,
  corpus persistence and rerun without duplicate documents pass. Upload bytes are
  injected at the file-upload boundary; parsing, embeddings and FAISS run normally.
- Fresh API process: password login, signed-token corpus read and unauthorized
  access rejection pass.
- The 160-package installed runtime dependency closure passes the existing
  GPL/AGPL/LGPL license gate. Runtime PDF, EPUB and translation adapters replace
  PyMuPDF, EbookLib and deep-translator. See the [dependency review](../security/README.md).
- The local dependency audit retains one NLTK finding with a reviewed,
  source-hash-bound not-affected assessment expiring on 2026-10-09. It reports zero
  blocking findings; NLTK itself has no patched release for this advisory.
- Ruff, YAML, Markdown, pre-commit (including secret detection), and all 1,267
  Python files parsed with Python 3.11 grammar pass. The configured Bandit check
  reports no findings.
- Markdown lint was repaired across the repository and is now mandatory in CI.
- The initial Linux audit exposed old setuptools and an unresolved CPU-wheel
  lookup; the updated audit and dependency floor require a new CI run. The public
  hosted site was inspected on 2026-09-09 and displays “Error running app.” It has
  not been redeployed with this follow-up.

## Test integrity and scope

The merged PR #4327 baseline was **677 failed, 8,982 passed, 25 skipped, 1 xfailed
and 25 errors**, with **52.26% line coverage** across `src` and `app`.

The follow-up replaces several tests of copied local stand-ins with tests of
production functions and real Streamlit views. The misspelled duplicate
`test_teext_normalization_properties.py` was consolidated into the production
normalization suite. Constant-return padding classes were removed from six
modules; their real scanner, security and runtime implementations remain.
Five legacy utility exclusions were removed after repairing their production APIs
and tests; the remaining exclusions are documented in `pytest.ini`. Test counts
therefore differ from the baseline, and passing counts alone are not a coverage
claim.

New regression coverage includes authentication recovery and password rotation,
account suspension/deletion, file parsing and export payloads, cancellation,
soft deletion, document revision lifecycle and storage, analytics, and dashboard
navigation. Five standalone dashboards generate illustrative data; they now label
that data explicitly and retain it across navigation. These demonstrations do not
provide analysis of uploaded documents.

For deployment configuration, backups, SMTP and the Cloud reboot/redeploy
procedure, use the [Streamlit redeployment guide](streamlit-redeploy.md).
