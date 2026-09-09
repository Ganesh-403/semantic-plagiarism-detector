````markdown
# E2E Playwright Test for Critical Path (Issue #3030)

First true end-to-end test for the Semantic Plagiarism Detector. Launches a real Streamlit server against an isolated SQLite DB, drives a real Chromium browser through the login → upload → scan → score-render critical path, and asserts the plagiarism score appears on-screen.

## Why

The repo has ~700 unit tests but no test that wires the Streamlit frontend to the API, DB, and ML pipeline together. Issue #3030 asks for exactly that.

## Layout

```text
tests/e2e/
├── __init__.py
├── conftest.py                  # session-scoped Streamlit launcher + page fixtures
├── README.md                    # this file
├── fixtures/
│   └── sample_docs.py           # generates student_a/b/c .txt files at run time
├── pages/
│   ├── __init__.py
│   ├── login_page.py            # Page Object for the auth screen
│   └── upload_page.py           # Page Object for upload + scan + score
└── test_critical_path.py        # the two E2E tests (happy + negative path)
```text
````

## Install

```bash
pip install -r requirements-e2e.txt
playwright install chromium

```

## Run

```bash
# Full E2E suite
pytest tests/e2e/ -m e2e

# Single test, with stdout
pytest tests/e2e/test_critical_path.py::test_login_upload_and_assert_plagiarism_score -m e2e -s

# Skip E2E in the default CI unit-test job
pytest -m "not e2e and not slow and not integration"

```

## What the test asserts

### `test_login_upload_and_assert_plagiarism_score`

- Browser launches against the live Streamlit server.
- Login form is filled with seeded credentials (`e2e_user` / `TestPass123!`).
- Two near-duplicate `.txt` files are uploaded via the file uploader.
- The "Run Quick Verification" search runs against a snippet of the uploaded text.
- A result expander with the pattern `#1 · Document-001 (chunk #1) — 87.5%` appears, and the `Similarity: NN.N%` HTML badge inside it is parsed back out as a number between 0 and 100.

### `test_unrelated_documents_yield_no_false_positive`

- Uploads two unrelated documents (`student_a.txt` + `student_c.txt`).
- Searches for a snippet about quantum entanglement that isn't in the corpus.
- Asserts the app prints "✅ No significant matches found in the assignment database." (or shows the empty result UI).

## Isolation guarantees

- The `streamlit_url` session fixture points the auth + corpus DBs at a `tmp_path_factory` temp directory, so the test never touches the developer's `users.db` / `corpus.db`.
- A test user is seeded via the project's own `src.db.auth.add_user` helper so the password is Argon2-hashed with the same parameters the live app uses.
- The Streamlit server is launched with `--server.headless true` and `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`.
- Chromium is launched with `headless=True` and a fixed 1280×900 viewport for deterministic screenshots.

## CI integration

Add a separate CI job (do not add E2E to the existing unit-test job — the Playwright binary download would slow it down):

```yaml
# .github/workflows/e2e.yml
name: E2E
on: [push, pull_request]

jobs:
  playwright:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-e2e.txt
      - run: playwright install --with-deps chromium
      - run: pytest tests/e2e/ -m e2e
```

```text

```
