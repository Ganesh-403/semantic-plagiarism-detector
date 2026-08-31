# Contributing

## Welcome

Thank you for your interest in contributing to the Semantic Plagiarism Detection System! We welcome contributions from everyone, whether you're fixing a bug, improving documentation, adding a feature, or suggesting an idea.

## Getting Started

### Fork the repository

Fork this repository to your GitHub account using the "Fork" button on the repository page.

### Clone locally

```bash
git clone [https://github.com/](https://github.com/)<your-username>/semantic-plagiarism-detector.git
cd semantic-plagiarism-detector

```

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install pytest-cov

```

`requirements-dev.txt` contains development-only tools (ruff, pylint, black,
isort, pre-commit) and is not needed for deployment.

### Run the project

## 5. Run the project

Ensure the project builds and runs correctly before making any changes.

### 💡 Testing with Pre-populated Seed Data

Instead of manually registering admin accounts and uploading test documents to verify dashboard visual styles or logic behavior, you can load pre-populated seed data:

```bash
make load-seed   # Or: python scripts/manage_seed.py load

```

This loads pre-configured accounts:

* **Admin**: `admin` / `admin123`
* **Teacher**: `teacher` / `teacher123`

### Running Tests and Coverage

To run tests with coverage reporting:

```bash
pytest --cov=src --cov-report=term-missing

```

The test suite enforces an 80% minimum coverage threshold.

---

# 📋 Before You Start

Before requesting an issue:

* Read the issue description carefully.
* Ask questions if anything is unclear.
* Wait for maintainer approval before beginning work.
* Work on **one assigned issue at a time** unless approved otherwise.
* Search existing Issues and Discussions before opening a new one.

---

# 📝 Contribution Guidelines

Please follow these guidelines when contributing:

* Keep code clean, readable, and modular.
* Follow the existing project architecture.
* Write meaningful commit messages.
* Add comments and docstrings where appropriate.
* Test your implementation before opening a Pull Request.
* Avoid unrelated code changes within the same PR.

### Project Structure

* Core NLP algorithms, parsers, embeddings, indexing, and similarity modules belong in:

```
src/core/

```

* Database managers and persistence utilities belong in:

```
src/db/

```

* Visualization and plotting components belong in:

```
src/visualization/

```

* Unit tests should be added to the corresponding directory inside:

```
tests/

```

Example:

```
src/core/parser.py
tests/core/test_parser.py

```

### Architecture Decisions

If your contribution introduces major architectural changes, please document them by writing an Architecture Decision Record (ADR). Follow the format defined in the [ADR Template](https://www.google.com/search?q=docs/adr/adr-template.md).

---

# 📌 Issue Assignment Policy

Issues are assigned on a first-come, first-served basis unless stated otherwise.

For larger or more complex features, maintainers may request a short implementation plan before assigning the issue.

Once assigned:

```bash
streamlit run app/streamlit_app.py

```

The application will open at http://localhost:8501. Default credentials are username `admin` and password `admin123`.

## Branching Strategy

Use descriptive branch names following these patterns:

* `feature/add-search` - For new features
* `fix/login-bug` - For bug fixes
* `docs/update-contributing` - For documentation changes
* `refactor/embedding-model` - For code refactoring

## Code Style

This project uses **Ruff** for linting and formatting:

* **Check code style**: `ruff check .`
* **Format code**: `ruff format .`

Follow PEP 8 conventions and keep code clean, readable, and modular.

## Running Tests

Run the test suite using pytest:

```bash
python -m pytest

```

The project uses pytest with configuration in `pytest.ini`. Tests are located in the `tests/` directory and mirror the structure of the `src/` directory.

## Submitting Pull Requests

* Keep PRs focused on a single issue or feature
* Write meaningful commit messages that explain "why" not just "what"
* Link related issues using "Fixes #issue_number" in your PR description
* Ensure all tests pass before submitting
* Run `ruff check .` and `ruff format .` to maintain code quality
* Respond to review feedback promptly and make requested changes

## Changelog

This project maintains `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

If your PR changes behavior, adds a feature, or fixes a bug, add an entry under the `## [Unreleased]` section at the top of `CHANGELOG.md` (create this section if it doesn't exist yet, directly below the file header). Use these subsections as needed:

* `### Added` - new features
* `### Changed` - changes to existing functionality
* `### Fixed` - bug fixes

Each entry should be a short bullet describing the change, referencing the relevant module or file path where helpful. Skip the changelog for purely internal changes with no user-facing or API impact (e.g. test-only additions, typo fixes in comments).

Example:

```markdown
## [Unreleased]

### Added
- Support for `.rtf` file uploads in the document parser (`src/core/document_parser.py`).

### Changed
- Increased default similarity threshold from 0.55 to 0.59 for improved precision.

### Fixed
- Corrected off-by-one error in paragraph chunk indexing (`src/core/text_chunking.py`).

```

When a new version is released, the maintainer will rename `[Unreleased]` to the version number and date (e.g. `## [1.1.0] - 2026-08-15`) and open a fresh `[Unreleased]` section above it.

## Bumping the Version

The application version is defined in a single place: `src/version.py`
(the `version` string). `src/utils/version_check.py` imports `APP_VERSION`
from that file, so there is only one place to update.

To release a new version:

1. Update `version` in `src/version.py` (e.g. `"1.0.0"` → `"1.1.0"`).
2. Update `CHANGELOG.md` as described above, renaming `[Unreleased]` to the
new version and date.
3. Commit both changes together in the same PR.

Do not hardcode the version anywhere else — always import it from
`src.version`.

## Reporting Issues

When opening bug reports or feature requests:

* Search existing issues first to avoid duplicates
* Use the provided issue templates in `.github/ISSUE_TEMPLATE/`
* Provide clear steps to reproduce bugs
* Include relevant error messages, logs, or screenshots
* Describe the expected behavior versus actual behavior

## Adding Custom NLP Scorers

We welcome contributions of new lexical similarity metrics and advanced embedding models! To integrate a new NLP algorithm into the `semantic-plagiarism-detector` codebase, please follow these steps:

1. **Implement the Metric/Model:**
Create a new Python file in the `src/nlp/scorers/` directory (or the designated NLP module). Your new scorer should inherit from the base `BaseScorer` class and implement the required `calculate_similarity(text_a, text_b)` method.
2. **Add Configuration Parameters:**
If your model requires specific thresholds, batch sizes, or API keys, add these default parameters to `config/nlp_settings.yaml`.
3. **Register the Scorer:**
Update the scorer factory (e.g., `src/nlp/factory.py`) to register your new algorithm so the main detection pipeline can dynamically instantiate it via command-line flags or config files.
4. **Write Unit Tests:**
Create corresponding tests in `tests/nlp/` to verify your algorithm's accuracy, handling of edge cases, and performance on standard text pairs.
5. **Document Your Code:**
Ensure your class and methods include detailed docstrings explaining the mathematical or logical basis of the metric, any external dependencies (like HuggingFace transformers), and expected output ranges.

```

```