# Recovery validation

The follow-up is still in progress. Streamlit and API startup checks pass, and
remaining test failures have been reduced from 677 failures and 25 errors to two
Streamlit test timeouts in the latest complete test run. Overall and changed-line coverage gates remain
blocking; this work is not ready to merge or deploy as a fully verified release.

## Latest complete suite

Windows, Python 3.13.7, 2026-09-09, source held unchanged for the entire run:

```console
python -m pytest -n 2 --dist=loadfile --cov=src --cov=app --junitxml=suite.xml --cov-report=xml --cov-report=term tests
```

Result: **9,720 passed, 2 failed, 24 skipped, 1 xfailed, 0 errors in 818.99 seconds**.
Both failures were the three-second AppTest deadline in `tests/app/test_app_clear.py`.
The tests now allow 30 seconds, retain their assertions, and pass in the 169-test
Streamlit/utility regression group. The Unicode normalization regressions pass.

The complete snapshot measured **61.32% line coverage** (40,019 / 65,264 lines)
and **51.02% branch coverage** (8,858 / 17,362 branches). The combined coverage
measurement used by `coverage report` is **59.15%**, below the existing **85%**
requirement. The **90% changed-line** requirement is also retained. No source
modules were excluded to raise these numbers.

A subsequent parallel run stopped during collection because a restored warning-sort
test parametrized an unordered set. Its case ordering has been made deterministic;
a replacement full run is required. Coverage from that collection-only run is not
presented as full-suite coverage.

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
- The complete dependency audit retains one NLTK finding with a reviewed,
  source-hash-bound not-affected assessment expiring on 2026-10-09. It reports zero
  blocking findings; NLTK itself has no patched release for this advisory.
- Ruff, YAML, Markdown, pre-commit (including secret detection), and all 1,267
  Python files parsed with Python 3.11 grammar pass. The configured Bandit check
  reports no findings.
- Markdown lint was repaired across the repository and is now mandatory in CI.
- Linux/Python 3.11–3.13 GitHub checks still need verification. The public hosted
  site was inspected on 2026-09-09 and displays “Error running app.” It has not been
  redeployed with this follow-up.

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
