# Development Guide

This document explains how to set up the local development environment and use the project's pre-commit hooks.

## Pre-commit Hooks

This project uses **pre-commit** to automatically run formatting, linting, and basic validation checks before each commit.

### Install pre-commit

```bash
pip install pre-commit
```

### Install Git hooks

```bash
pre-commit install
```

### Run hooks manually

To run all configured hooks on every file:

```bash
pre-commit run --all-files
```

To run the hooks only on staged files:

```bash
pre-commit run
```

## Configured Hooks

The project currently runs the following hooks:

* Ruff (Python linting with automatic fixes)
* Black (Python code formatting)
* isort (Import sorting)
* Trailing Whitespace
* End-of-File Fixer
* YAML Validation
* Large File Check

These checks help maintain consistent code quality and reduce formatting and linting issues before code reaches CI.
## Generating Seed Data

The project includes a utility script located at `scripts/generate_seed_data.py` to help developers quickly populate a local development database with mock student assignment submissions, corpora, and plagiarism test cases.

### Usage

Run the script directly using Python:

```bash
python scripts/generate_seed_data.py
```
### Optional Arguments
You can customize the amount and type of seed data generated using command-line flags:

--num-documents INTEGER: Number of mock document submissions to generate (Default: 20).

--output-dir PATH: Directory where generated mock text/PDF files should be saved (Default: data/seed_documents).

--include-plagiarism: Include deliberate plagiarized sentence pairs for testing detection algorithms.

--reset-db: Clear existing corpus database records before inserting new seed data.

Example Commands
Generate default set of mock submissions:

```bash
python scripts/generate_seed_data.py
```
Generate 50 documents with synthetic plagiarism pairs and reset local DB:

```bash
python scripts/generate_seed_data.py --num-documents 50 --include-plagiarism --reset-db
```
