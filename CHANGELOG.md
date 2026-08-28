# Changelog

All notable changes to the **Semantic Plagiarism Detection System** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `docker-compose.override.yml` mounting `./src` and `./app` into container for live hot-reloading during local development (`docker-compose.override.yml`).
- Automated fault tolerance test for mid-session Redis connection drop and graceful in-memory failover (`tests/core/test_fault_tolerance.py`, `tests/utils/test_redis_fallback_failover.py`).
- Added `--recursive` support to the CLI scan command for scanning documents in nested subdirectories.
- Admin DB repair CLI with `--check` / `--vacuum` / `--reindex` for `corpus.db` and `users.db` (`scripts/repair_db.py`).
- Run `pre-commit run --all-files` in the CI lint job (`.github/workflows/ci.yml`).
- Daily storage snapshots and projected days-until-full estimate via `storage_history` (`src/utils/storage_metrics.py`).
- Prometheus gauge `spd_active_threads` updated from `threading.active_count()` on each `/metrics` scrape (`src/core/metrics.py`).
- `reset_analysis_session_state()` clears document lists, matrices, and scan flags while keeping theme and session id (`app/state_manager.py`).
- Custom low/mid/high color-scale thresholds in `build_similarity_workbook` (`src/utils/excel_export.py`).
- `show_notification()` toast helper with success/warning/error/info icons (`app/components/notifications.py`).

### Fixed
- Mobile viewports (<768px): tighter main padding and shorter plotly chart heights (`app/css_constants.py`).
- Graceful degradation when reportlab is not installed: badge generator module loads without reportlab and raises a clear error only when PDF generation is requested (`src/utils/badge_generator.py`).
- Embed bundled DejaVu Sans / Roboto TTF in ReportLab PDF reports so non-ASCII document names render correctly (`src/utils/pdf_report.py`).
- Graceful degradation when reportlab is not installed: badge generator module loads without reportlab and raises a clear error only when PDF generation is requested (`src/utils/badge_generator.py`).
- Fixed assertion mismatch in `test_sync_flagged_incidents_bulk_upsert` to verify that `severity_rank` is updated to `"Critical"` and other ranks during bulk upsert (`tests/db/test_incidents.py`, `tests/db/test_incidents_bulk.py`).
- Fixed unreadable line overflowing for long URLs in ReportLab PDF reports by adding `wordWrap='CJK'` to paragraph styles and inserting zero-width spaces into long URLs (`src/utils/pdf_report.py`).
- Robust claim parsing in `.github/workflows/ecsoc-automation.yml` using structured hidden HTML comments to prevent breaking on greeting message variations.
- Restored broken imports in `badge_generator.py` and kept invalid hex colors falling back to `DEFAULT_BADGE_COLOR`.

### Security
- Centralize spreadsheet formula sanitization in `export_sanitizer` and apply it across excel, bulk, and batch exports (`src/utils/export_sanitizer.py`).
- Sanitize bulk export ZIP entry names with `sanitize_filename` so document and incident metadata cannot introduce `..` or absolute paths (`src/utils/bulk_export.py`).
- Integrated `zxcvbn` password strength evaluation in `_validate_password_complexity` to block common dictionary passwords and weak credentials (`src/db/auth.py`, `requirements.txt`).
- Explicitly excluded `.env`, `.git/`, `.venv/`, `*.sqlite`, and bytecode caches in `.dockerignore` to prevent leaking secrets and development artifacts into production containers (`.dockerignore`).

### Changed
- Optimized `clear_session` and `clear_pattern` with Redis pipelining to batch deletions into a single network round-trip (`src/utils/redis_cache.py`).
- Warning list pagination no longer writes `st.session_state` directly; page updates are applied via a view-layer callback (`src/utils/warning_list.py`).
- Diff highlighter default match length is configurable via `DEFAULT_DIFF_MIN_MATCH_LENGTH` (`src/core/config.py`, `src/utils/diff_highlighter.py`).
- Build originality badge SVGs with `xml.etree.ElementTree` instead of f-string interpolation (`src/utils/badge_generator.py`).

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
