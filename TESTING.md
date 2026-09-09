# Testing Strategy

This document outlines the testing strategy, architecture, and developer workflows for the Semantic Plagiarism Detector platform.

## Architecture & Mocking Strategy

Tests use dependency injection, temporary directories and scoped fixtures to keep
application data separate from test data. Fixtures are defined in `tests/conftest.py`
and in the test subdirectories.

### Database fixtures

The opt-in `mock_db` fixture creates separate temporary SQLite files for corpus and
authentication data and patches the corresponding database paths for that test.
Other suites provide their own scoped fixtures. These are file-backed databases,
not in-memory databases. New database tests must explicitly use an isolated fixture.

### FAISS & Redis Infrastructure

- **Redis Mocking:** The system uses `fakeredis` to mock the Redis backend for the `TelemetryService` and caching layers, simulating connection failures, TTL expiry, and cache misses.
- **FAISS Isolation:** FAISS vectors are generated dynamically and saved to temporary file descriptors to test desync recovery paths (`synchronization.py`) without modifying the disk index.

## Running Tests

We provide a robust testing framework designed to enforce code coverage and execute targeted subsets across multiple CPU cores.

### Parallel Test Execution (`pytest-xdist`)

Parallel execution is opt-in. The CI runner uses two workers and `--dist=loadfile`
to keep each test file on one worker. Fixtures must still isolate database paths,
model singletons and cached state; the scheduler does not prevent shared-state bugs.

```bash
# Serial default
python -m pytest

# Same worker scheduling as CI, including application coverage
python -m pytest -n 2 --dist=loadfile --cov=src --cov=app

# Focused test without a repository-wide coverage report
python -m pytest --no-cov tests/app/test_app_clear.py

# Explicitly selected integration tests
python -m pytest -m integration
```

`pytest.ini` still contains legacy test exclusions. Passing the discovered suite
does not validate those excluded files. See [recovery validation](docs/recovery-validation.md)
for the current measured results and remaining coverage gap.

### Automated Test Runner

Instead of calling `pytest` directly, you can also use the provided `scripts/run_tests.py` automation script. It selects pytest markers, worker scheduling and coverage options; external services must be configured separately.

```bash
# Run the entire test suite and generate an HTML coverage report
python scripts/run_tests.py --all

# Run the test suite with parallel execution explicitly enabled
python scripts/run_tests.py --all --parallel

# Run only isolated unit tests (excludes network/DB-heavy operations)
python scripts/run_tests.py --unit

# Run full integration tests (tests full stack against local mock DBs)
python scripts/run_tests.py --integration

# Force coverage enforcement (fails if coverage drops below 85%)
python scripts/run_tests.py --all --enforce-coverage 85
```

### Makefile Targets

For convenience, `Makefile` encapsulates these commands:

```bash
make test         # Runs standard test suite
make test-unit    # Runs only unit tests
make test-cov     # Runs tests with coverage enforcement
```

## Generating Mock Data

To quickly populate the dashboard with realistic dummy essays, use the built-in mock data generator available from the Streamlit application.

### Prerequisites

The mock data generator requires the `faker` package. If it is not already installed, run:

```bash
pip install faker
```

### Steps

1. Launch the Streamlit application.
2. Log in as an administrator.
3. Open the **🧪 Developer Tools** section in the sidebar.
4. (Optional) Enter a custom **Mock Class/Section** and **Mock Assignment Title**.
5. Click **⚗️ Generate Mock Data**.

The generator will:

- Create five realistic student essays using the Faker library.
- Store the generated essays in `corpus.db`.
- Skip essays that already exist in the database.
- Rebuild the FAISS index (`corpus.index`).
- Automatically refresh the application so the demo essays are immediately available in the dashboard.

## Adding New Tests

1. **File Location:** Place new tests in `tests/` mirroring the `src/` directory structure.
2. **Naming Convention:** Prefix test files with `test_` and functions with `test_`.
3. **Markers:** Always decorate tests with `@pytest.mark.unit` or `@pytest.mark.integration`.
4. **Coverage:** Ensure any new feature branches meet the 85% coverage threshold before requesting a Pull Request review.
