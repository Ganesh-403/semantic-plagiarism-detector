# CI/CD Guide

This repository uses GitHub Actions for continuous integration, issue and pull request automation, and draft release generation. This document summarizes the current workflows, the events that trigger them, the permissions they need, and the deployment/runtime settings that must be provided outside GitHub Actions.

## Workflows

### 1. CI Pipeline

File: [.github/workflows/ci.yml](../.github/workflows/ci.yml)

**Purpose**

Runs code quality checks, static analysis, dependency vulnerability scanning, and the test suite.

**Triggers**

- `push` to `main` and `develop`
- `pull_request` targeting `main` and `develop`

**Behavior**

- Cancels in-progress runs for the same ref when a newer run starts.
- Uses Python 3.10 for linting, type checking, and dependency scanning.
- Runs `ruff` for linting.
- Runs `mypy` against `src/`.
- Runs `pip-audit` against `requirements.txt` and uploads a JSON report when the job fails.
- Runs the test suite on Python 3.9, 3.10, and 3.11 with coverage enabled.
- Uploads coverage artifacts from each matrix job.

**Notes**

- The test job currently declares a dependency on `security-scan` in addition to `lint-and-type-check`.
- If that job is not defined in the workflow file, the workflow definition should be corrected before relying on it in CI.

**Secrets / permissions**

- No repository secret is referenced directly in this workflow.
- The workflow relies on the standard `GITHUB_TOKEN` that GitHub provides to Actions.

### 2. Release Draft Generator

File: [.github/workflows/release.yml](../.github/workflows/release.yml)

**Purpose**

Creates a draft GitHub Release automatically when a semantic version tag is pushed.

**Triggers**

- `push` of tags matching `v<major>.<minor>.<patch>`
- `push` of prerelease tags such as `v1.2.3-beta`

**Behavior**

- Checks out the full repository history so previous tags are available.
- Extracts the pushed tag name and uses it as the release name.
- Generates `CHANGELOG.md` from commit history since the previous tag.
- Creates a draft release with `softprops/action-gh-release`.
- Marks prerelease tags as prereleases.

**Secrets / permissions**

- Requires `contents: write` so the workflow can create a release.
- Uses the built-in `GITHUB_TOKEN`; no custom secret is listed in the workflow.

### 3. ECSoC Automation

File: [.github/workflows/ecsoc-automation.yml](../.github/workflows/ecsoc-automation.yml)

**Purpose**

Automates issue claiming, pull request onboarding, and stale-claim cleanup for the ECSoC contribution flow.

**Triggers**

- `issue_comment` with `created`
- `pull_request_target` with `opened`
- `schedule` at `0 0 * * *` UTC
- `workflow_dispatch`

**Behavior**

- On issue comments, ignores pull request comments, bot comments, the issue author, the repository owner, and already assigned issues.
- Adds the `ECSoC26` label and assigns the commenter when a claim is accepted.
- Enforces a maximum of five open claimed issues per user.
- On pull request open events, assigns the author, adds the `ECSoC26` label, and posts a welcome comment.
- On scheduled or manual runs, unassigns users who exceed the open-issue limit and clears stale claims older than four days when there is no recent activity and no linked PR.

**Secrets / permissions**

- Requests `issues: write` and `pull-requests: write`.
- Uses the standard GitHub Actions token; no extra repository secrets are referenced.

## Deployment and Runtime Setup

The repository does not define a separate production deployment workflow. The documented deployment path is container-based local deployment with Docker Compose, plus direct local execution for development.

### Docker Compose deployment

1. Ensure Docker Engine 20.10+ and Docker Compose v2+ are installed.
2. Create a `.env` file in the repository root if you need to override defaults from [.env.example](../.env.example).
3. Start the application with:

```bash
docker compose up --build
```

4. Open the dashboard at `http://localhost:8501`.
5. Stop the stack with `docker compose down`.
6. Remove the Redis volume as well with `docker compose down -v` if needed.

### Direct local run

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Launch the app with:

```bash
streamlit run app/streamlit_app.py
```

### Runtime configuration

The following settings are configured through environment variables rather than GitHub Actions secrets:

| Variable | Purpose |
|---|---|
| `APP_TITLE` | Optional application branding |
| `PLAGIARISM_WEBHOOK_URL` | Slack or Discord notifications for plagiarism events |
| `APP_BASE_URL` | Base URL used in notification links |
| `REDIS_URL` | Redis connection string |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD` | Fallback Redis settings |
| `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` | Daily summary email delivery |
| `FROM_EMAIL` | Sender address for daily summary emails |
| `ADMIN_EMAIL` | Fallback admin email address |
| `API_BEARER_TOKEN` | Bearer token for REST API access from LMS integrations |
| `LOCK_TIMEOUT_SECONDS` | Timeout for synchronization locks |

See [.env.example](../.env.example) for default values and comments.

## Setup Checklist

- Confirm the three workflow files exist under [.github/workflows](../.github/workflows).
- Keep `requirements.txt` aligned with the tools used in CI: `ruff`, `mypy`, `pytest`, `pytest-cov`, and `pip-audit`.
- Ensure the semantic version tag format used for releases is `vX.Y.Z` or a prerelease variant such as `vX.Y.Z-beta`.
- Provide the runtime environment variables needed for webhooks, Redis, email, and LMS API access before deployment.
