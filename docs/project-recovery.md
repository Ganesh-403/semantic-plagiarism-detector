# Project recovery and deployment

This change restores the main Streamlit analysis flow and addresses the open issue
backlog. The full configured Linux test suite passes on Python 3.11–3.13, but
overall coverage remains below the required gate. Validation results and remaining
blockers are recorded below so reviewers can assess the PR.

For the hosted app, follow the [redeployment checklist](streamlit-redeploy.md).

## Run the application

Use Python 3.11–3.13 and install `requirements.txt`. It includes the full runtime
and prefers CPU PyTorch wheels. `packages.txt` supplies libmagic, OpenMP and
Tesseract on Streamlit Community Cloud. In Community Cloud, select
`app/streamlit_app.py` as the entry point and Python 3.11 or newer.

Set `ADMIN_BOOTSTRAP_PASSWORD` to a unique strong password before first startup.
For local development, it can be placed in `.env`; for hosted deployments use the
platform secret/environment configuration. It creates `admin` only if that account
does not exist. Remove the setting after setup. Existing accounts retain their
passwords; change credentials inherited from development seed databases before
exposing a deployment.

Run `streamlit run app/streamlit_app.py`. Set `SPD_STATE_DIR` to a writable persistent
directory for user databases, corpus embeddings and indexes. Community Cloud local
files are not a durable storage service; preserve state externally if retention is
required. Model weights download on the first analysis, not the login page.
`SEMANTIC_PLAGIARISM_MODEL=all-MiniLM-L6-v2` selects the smaller English model;
below 1.5 GiB available RAM the loader also falls back to that model and logs it.
`PRELOAD_EMBEDDING_MODEL` and `ENABLE_EMBEDDED_API` default to disabled.

For a standalone API use `uvicorn src.api.app:app`. Configure JWT keys or bearer
tokens before exposing authenticated endpoints. Metrics and liveness probes remain
separate from private document-health endpoints. `POST /auth/login` requires
username/password and, when enabled, `otp_code`; it returns signed scoped tokens.
`POST /auth/2fa/setup` now requires the current password and refuses to reveal or
replace an existing second-factor secret.

## Docker

Use `docker compose -f docker-compose.yml up --build` for the image deployment.
The automatically discovered override file is for development source mounts.
Application code stays in `/app`; state lives at `/state`. Existing named volumes
`plagiarism_users` and `plagiarism_data` are reused at `/state` and `/state/data`,
respectively, so upgrading the image no longer serves old source from a volume.
Back up these volumes before upgrading. Do not delete volumes to apply code changes.

## Backlog coverage

The regression module is `tests/test_remaining_issues.py`.

| Issues                     | Implementation or verified behavior                                       |
| -------------------------- | ------------------------------------------------------------------------- |
| #4192, #4191               | Restored clone detector and four component class entry points             |
| #4190                      | MetricSample/MetricFamily response schema and metrics documentation       |
| #4057                      | Document deletion cascades across ten stored chunks                       |
| #4028, #4027               | Shared reentrant FAISS lock and default autosave on incremental additions |
| #4026, #4023, #4021        | Jaccard symmetry, overlap coefficient and normalized Levenshtein          |
| #4016, #4013               | Empty embedding dimensions and 1.5 GiB model fallback                     |
| #4003, #4001, #4000        | Compiled boundaries, short-tail merging and Markdown section boundaries   |
| #3994, #3695, #3692        | Translation errors, fidelity warnings and configurable detection length   |
| #3981, #3976, #3972        | Encrypted/rotated PDFs and score-based annotation colors                  |
| #3963, #3962               | Failed release lookup caching and prerelease filtering                    |
| #3777                      | Twenty-request rate-limit burst with Retry-After                          |
| #3763, #3757               | Unique session keys and authentication failure counters                   |
| #3746, #3732, #3705, #3702 | Docker detection, stopword warning and academic text statistics           |
| #3245, #3181               | Disk-full upload cleanup and accurate cleanup logging                     |
| #3168                      | Require a verified primary GitHub email                                   |

Further recovery fixes include duplicate Streamlit controls/OAuth paths, competing
FastAPI app instances, deferred package imports, missing public model records,
SQLite migration drift, malformed SQL, role alternatives, state isolation,
transactional document ingestion, and report/analytics rendering.

## Validation

The independent Streamlit smoke script deliberately runs outside pytest's model
stubs. `python scripts/smoke_streamlit.py` verifies login and the authenticated
empty dashboard. Set `SMOKE_REAL_MODEL=true` to additionally compare two text
files using the real embedding model, render result tabs, save documents/chunks,
and rerun without duplicate records. The upload widget itself is replaced with
in-memory file bytes because Streamlit AppTest does not drive file selection.
The parser, embeddings, similarity, FAISS and application views run normally.

`python scripts/smoke_api.py` verifies fresh API database initialization, real
password login, signed-token access to corpus statistics and unauthenticated
access rejection. Both smoke scripts use isolated temporary state.

The full suite uses `python -m pytest -n 2 --dist=loadfile`. Existing coverage
thresholds (85% overall, 90% changed lines) and existing legacy exclusions remain
visible. No new exclusions or expected failures were added to hide failures.
See [the validation report](recovery-validation.md) for run counts and failing test
modules. Linux installation and tests have passed in CI. Docker execution and the
maintained Streamlit Community Cloud deployment still need verification; local
success does not deploy the upstream site.

## Remaining blockers

- Overall coverage remains below 85%; the 90% changed-line gate passes. The latest
  complete-suite result and subsequent targeted repairs are recorded in the
  validation report. Keep the PR in draft until the full suite and CI gates pass.
- `deep-translator`, PyMuPDF and EbookLib have been removed from runtime dependencies.
  Translation uses first-party HTTP clients; PDF operations use pypdf/PDFium/ReportLab;
  EPUB parsing uses bounded archive reads and hardened XML parsing.
- NLTK 3.10.3 still has an unpatched advisory. The complete pip-audit report retains
  it. `security/dependency-assessments.json` records why the application is not
  affected: it never exposes the affected model-artifact import/export operations.
  The assessment is version-specific, expires on 2026-10-09, and stops applying if
  NLTK call sites change or an upstream fix becomes available. See the
  [upstream advisory](https://github.com/advisories/GHSA-8mgp-746c-j5xp).
- The resolved local runtime dependency closure passes the strengthened copyleft
  license gate. The Linux workflow uploads the full license inventory. Development
  tools such as yamllint/pygit2 are not runtime packages.

## Translation providers and demo data

`GOOGLE_TRANSLATE_API_KEY` enables the official Google Cloud Translation API.
Without that key, the default online translator uses MyMemory's public endpoint;
its quota failures are reported explicitly. DeepL uses its authenticated API and
splits requests at 50 texts. Timeouts are scoped to each translation call.

The seed generator writes sample text files to `--seed-dir` and populates the
configured application databases. `--state-dir PATH` creates an isolated demo
state directory containing `users.db`, `corpus.db` and `corpus.index`.
It uses the real embedding model; `--skip-index` explicitly creates metadata only.
`--dry-run` performs no database writes or model calls. Account passwords come
from `ADMIN_BOOTSTRAP_PASSWORD` and optional `SEED_TEACHER_PASSWORD`; there are no
built-in demo passwords. Never point demonstration seeding at production data.
