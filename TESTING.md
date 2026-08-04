# Testing Strategy

This document outlines the testing strategy, architecture, and developer workflows for the Semantic Plagiarism Detector platform.

## Architecture & Mocking Strategy
To ensure hermetic, deterministic test execution without side-effects, the repository utilizes robust dependency injection and global fixtures via `conftest.py`.

### Database Fixtures (`mock_db` & `mock_auth_db`)
The core database layers (`src.db.corpus_db`, `src.db.auth`, `src.db.incidents`) are built on SQLite databases stored in the root directory. To prevent tests from corrupting the production or local seed databases, we globally patch the internal `_DB_PATH` constants across the entire `src.db` namespace.
- **In-Memory Speed:** The fixtures route all test I/O to transient, file-backed SQLite instances in a temporary directory.
- **State Reset:** Databases are fully rebuilt and seeded at the start of each test phase, ensuring isolated execution and avoiding cross-test contamination.

### FAISS & Redis Infrastructure
- **Redis Mocking:** The system uses `fakeredis` to mock the Redis backend for the `TelemetryService` and caching layers, simulating connection failures, TTL expiry, and cache misses.
- **FAISS Isolation:** FAISS vectors are generated dynamically and saved to temporary file descriptors to test desync recovery paths (`synchronization.py`) without modifying the disk index.

## Running Tests

We provide a robust testing framework designed to enforce code coverage and execute targeted subsets.

### Automated Test Runner
Instead of calling `pytest` directly, use the provided `scripts/run_tests.py` automation script. It handles configuration parsing, coverage aggregation, and Docker container provisioning.

```bash
# Run the entire test suite and generate an HTML coverage report
python scripts/run_tests.py --all

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
