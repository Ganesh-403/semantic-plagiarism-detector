# Project recovery and deployment

This change restores the main Streamlit analysis flow and addresses the open issue
backlog. It is not a claim that the entire legacy test suite is green. Validation
results and remaining blockers are recorded below so reviewers can assess the PR.

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

| Issues | Implementation or verified behavior |
| --- | --- |
| #4192, #4191 | Restored clone detector and four component class entry points |
| #4190 | MetricSample/MetricFamily response schema and metrics documentation |
| #4057 | Document deletion cascades across ten stored chunks |
| #4028, #4027 | Shared reentrant FAISS lock and default autosave on incremental additions |
| #4026, #4023, #4021 | Jaccard symmetry, overlap coefficient and normalized Levenshtein |
| #4016, #4013 | Empty embedding dimensions and 1.5 GiB model fallback |
| #4003, #4001, #4000 | Compiled boundaries, short-tail merging and Markdown section boundaries |
| #3994, #3695, #3692 | Translation errors, fidelity warnings and configurable detection length |
| #3981, #3976, #3972 | Encrypted/rotated PDFs and score-based annotation colors |
| #3963, #3962 | Failed release lookup caching and prerelease filtering |
| #3777 | Twenty-request rate-limit burst with Retry-After |
| #3763, #3757 | Unique session keys and authentication failure counters |
| #3746, #3732, #3705, #3702 | Docker detection, stopword warning and academic text statistics |
| #3245, #3181 | Disk-full upload cleanup and accurate cleanup logging |
| #3168 | Require a verified primary GitHub email |

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

The full suite uses `python -m pytest -n 2 --dist=loadscope`. Existing coverage
thresholds (85% overall, 90% changed lines) and existing legacy exclusions remain
visible. No new exclusions or expected failures were added to hide failures.
See the PR description for final run counts. Docker execution, Linux installation,
and the maintained Streamlit Community Cloud deployment require their respective
runners; local success does not deploy the upstream site.

## Remaining blockers

- The legacy full test suite still has failures, including stale mocked interfaces,
  report/CLI contracts and feature-specific paths. The PR should remain a draft
  while these are unresolved; this recovery is not a clean-CI certification.
- Dependency audit reports `deep-translator` PYSEC-2022-252 and NLTK
  PYSEC-2026-3740. The advisory database lists no fixed version for either result.
  Findings remain unsuppressed. Sources: [deep-translator advisory](https://osv.dev/vulnerability/PYSEC-2022-252)
  and [NLTK advisory](https://github.com/advisories/GHSA-8mgp-746c-j5xp).
- The existing blanket copyleft policy conflicts with the runtime's PyMuPDF and
  EbookLib dependencies. Their license findings are preserved for maintainer
  resolution; no license exception is silently granted here.
