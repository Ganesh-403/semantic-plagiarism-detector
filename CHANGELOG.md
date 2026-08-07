# Changelog

All notable changes to the **Semantic Plagiarism Detection System** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Quickstart example script demonstrating the core pipeline (`extract_text`, `chunk_text`, `embed_documents`, similarity scoring) with execution instructions (`examples/basic_plagiarism_check.py`).
- Unit tests for the quickstart example (`tests/scripts/test_basic_plagiarism_check.py`).

### Fixed
- Restored `src` package importability by removing orphaned syntax from `src/core/document_parser.py`.
- Re-pointed `with_sqlite_retry` imports from the removed `src.db.common` location to `src.core.concurrency` (`src/core/__init__.py`, `src/db/auth.py`, `src/db/corpus_db.py`, `src/db/incidents.py`).
- Restored dual-mode (`bare`/parameterized) `with_sqlite_retry` decorator usage in `src/core/concurrency.py`.
- Restored missing `clean_text(..., remove_stopwords=...)` parameter and removed duplicate `extract_text_from_zip` definition in `src/core/document_parser.py`.
- Fixed `src/db/__init__.py` missing re-exports (`update_user_profile`, incident helpers).

## [1.0.0] - 2026-07-21

### Added
- Cross-lingual preprocessing pipeline supporting language detection and automatic English alignment (`src/core/cross_lingual.py`, `src/core/translator.py`).
- SQLite-backed corpus database and chunk vector persistence (`src/db/corpus_db.py`).
- Plagiarism incident tracking and review status management (`src/db/incidents.py`).
- PDF report export utility (`src/utils/pdf_report.py`).
- Webhook alert integration for high-similarity matches (`src/core/webhook.py`).
- RoBERTa-based AI-generated text detection module (`src/core/ai_detector.py`).
- Redis caching utility for multi-node deployments (`src/utils/redis_cache.py`).
- Originality certificate generator (`src/utils/badge_generator.py`).
- Daily summary email notification service (`src/utils/daily_summary_email.py`).
- Standard open-source governance documents (`CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `CHANGELOG.md`, GitHub issue templates).

### Changed
- Reorganized `tests/` directory structure into modular `tests/app/`, `tests/core/`, `tests/db/`, `tests/utils/`, and `tests/visualization/`.
- Moved `warning_list.py` into `src/utils/` to maintain strict `src/` modular encapsulation.
- Updated `requirements.txt` to remove duplicate dependency entries.

### Fixed
- Handled missing `redis` dependency in `src/utils/redis_cache.py` to prevent import crashes during test collection.
- Removed legacy duplicate `utils/pdf_report.py` file to ensure tests validate the production PDF report module.
