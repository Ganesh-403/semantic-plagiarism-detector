# Contributing

## Welcome

Thank you for your interest in contributing to the Semantic Plagiarism Detection System! We welcome contributions from everyone, whether you're fixing a bug, improving documentation, adding a feature, or suggesting an idea.

## Getting Started

### Fork the repository

Fork this repository to your GitHub account using the "Fork" button on the repository page.

### Clone locally

```bash
git clone https://github.com/<your-username>/semantic-plagiarism-detector.git
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
pip install pytest-cov
```

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

- Read the issue description carefully.
- Ask questions if anything is unclear.
- Wait for maintainer approval before beginning work.
- Work on **one assigned issue at a time** unless approved otherwise.
- Search existing Issues and Discussions before opening a new one.

---

# 📝 Contribution Guidelines

Please follow these guidelines when contributing:

- Keep code clean, readable, and modular.
- Follow the existing project architecture.
- Write meaningful commit messages.
- Add comments and docstrings where appropriate.
- Test your implementation before opening a Pull Request.
- Avoid unrelated code changes within the same PR.

### Project Structure

- Core NLP algorithms, parsers, embeddings, indexing, and similarity modules belong in:

```
src/core/
```

- Database managers and persistence utilities belong in:

```
src/db/
```

- Visualization and plotting components belong in:

```
src/visualization/
```

- Unit tests should be added to the corresponding directory inside:

```
tests/
```

Example:

```
src/core/parser.py
tests/core/test_parser.py
```

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

- `feature/add-search` - For new features
- `fix/login-bug` - For bug fixes
- `docs/update-contributing` - For documentation changes
- `refactor/embedding-model` - For code refactoring

## Code Style

This project uses **Ruff** for linting and formatting:

- **Check code style**: `ruff check .`
- **Format code**: `ruff format .`

Follow PEP 8 conventions and keep code clean, readable, and modular.

## Running Tests

Run the test suite using pytest:

```bash
python -m pytest
```

The project uses pytest with configuration in `pytest.ini`. Tests are located in the `tests/` directory and mirror the structure of the `src/` directory.

## Submitting Pull Requests

- Keep PRs focused on a single issue or feature
- Write meaningful commit messages that explain "why" not just "what"
- Link related issues using "Fixes #issue_number" in your PR description
- Ensure all tests pass before submitting
- Run `ruff check .` and `ruff format .` to maintain code quality
- Respond to review feedback promptly and make requested changes

## Reporting Issues

When opening bug reports or feature requests:

- Search existing issues first to avoid duplicates
- Use the provided issue templates in `.github/ISSUE_TEMPLATE/`
- Provide clear steps to reproduce bugs
- Include relevant error messages, logs, or screenshots
- Describe the expected behavior versus actual behavior
