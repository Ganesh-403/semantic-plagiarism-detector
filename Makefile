.PHONY: run test lint format clean load-seed save-seed

# Run the Streamlit dashboard application
run:
	streamlit run app/streamlit_app.py

# Run the entire test suite via the automated runner
test:
	python scripts/run_tests.py --all

# Run only fast, isolated unit tests
test-unit:
	python scripts/run_tests.py --unit

# Run full integration tests against local mock DBs
test-integration:
	python scripts/run_tests.py --integration

# Run tests with strict coverage enforcement
test-cov:
	python scripts/run_tests.py --all --enforce-coverage 85

# Run code formatters and linters check
lint:
	ruff check .
	black --check .
	isort --check-only --profile black .

# Auto-format codebase
format:
	black .
	isort --profile black .
	ruff check --fix .

# Clean temporary Python, testing, and IDE cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	@echo "✅ Cleaned all Python, test, and lint cache artifacts."

# Load pre-populated seed data
load-seed:
	python scripts/manage_seed.py load

# Save current databases as seed data
save-seed:
	python scripts/manage_seed.py save
