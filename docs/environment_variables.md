# Environment Variables Reference

This document lists every environment variable read by the application,
grouped by functional area. All variables are optional unless marked
**Yes** in the Required column — the app falls back to safe defaults for
everything else so it can run out of the box in development.

Variables marked **Yes** are only strictly required in specific deployment
modes (e.g. production, or when a given feature such as webhooks or SSO is
enabled); see the Description column for the condition.

---

## Core Application

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `APP_ENV` | *(unset)* | No | Set to `test` to enable test-mode behavior (e.g. relaxed JWT/auth checks used by the test suite). Leave unset in normal development and production. |
| `APP_ENVIRONMENT` | `production` | No | Controls production-only behaviors such as stricter logging and default-secret warnings. Set to anything other than `production` (e.g. `development`) to relax these checks locally. |
| `APP_TITLE` | *(empty, falls back to built-in default title)* | No | Overrides the application's display title shown in the UI. |
| `APP_WELCOME_MESSAGE` | *(empty, falls back to built-in default message)* | No | Custom welcome message shown on the app's landing/home view. |
| `APP_BASE_URL` | `http://localhost:8501` | No | Base URL used to build absolute links in webhook alerts, emails, and PDF reports. Should be set to the public URL in production. |
| `CORS_ALLOWED_ORIGINS` | `*` | No | Comma-separated list of origins allowed to make cross-origin requests to the API. Should be restricted (not `*`) in production. |
| `ENABLE_HSTS` | *(empty, HSTS disabled)* | No | Set to a truthy value (`true`/`1`/`yes`) to add the `Strict-Transport-Security` header. Only enable once HTTPS is confirmed working end-to-end. |

## API Server

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `API_HOST` | `0.0.0.0` | No | Host/interface the ASGI API server binds to. |
| `API_PORT` | `8000` | No | Port the ASGI API server listens on. |
| `API_BEARER_TOKEN` | `default-token-secret-key-12345` (dev only) | **Yes** (production) | Bearer token required to authenticate against protected REST API endpoints. **Must** be overridden with a strong secret in production — the default is insecure and intended for local development only. |
| `API_BEARER_TOKENS_MAPPING` | *(empty)* | No | JSON mapping of multiple named bearer tokens to client identities, for supporting more than one API consumer. |
| `API_SUPPORT_EMAIL` | *(empty)* | No | Support contact email surfaced in API error responses / documentation. |
| `API_SUPPORT_URL` | *(empty)* | No | Support URL surfaced in API error responses / documentation. |

## Authentication & Secrets

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `JWT_SECRET_KEY` | *(unset — raises `ValueError` when needed)* | **Yes** | Secret key used to sign and verify JWT access/refresh tokens. The application refuses to issue or verify tokens without it. |
| `SESSION_SECRET_PEPPER` | `fallback_secure_string_3442` (dev only) | **Yes** (production) | Pepper appended to session secrets before hashing. Must be overridden with a strong random value in production. |
| `ENCRYPTION_KEY` | *(unset)* | No | Fallback symmetric key used for encrypting stored secrets (e.g. OTP secrets) when `OTP_ENCRYPTION_KEY` is not set. |
| `OTP_ENCRYPTION_KEY` | *(unset, falls back to `ENCRYPTION_KEY`)* | No | Preferred symmetric key used specifically for encrypting stored OTP secrets. |
| `ALLOWED_USER_ROLES` | *(empty, falls back to built-in default role set)* | No | Comma-separated list of role names permitted at signup/admin-assignment time. |

## Single Sign-On (SSO)

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `GOOGLE_CLIENT_ID` | *(unset)* | **Yes** (to enable Google SSO) | OAuth client ID for "Sign in with Google". Google SSO is unavailable until this is set. |
| `GOOGLE_CLIENT_SECRET` | *(unset)* | **Yes** (to enable Google SSO) | OAuth client secret for "Sign in with Google". |
| `GITHUB_CLIENT_ID` | *(unset)* | **Yes** (to enable GitHub SSO) | OAuth client ID for "Sign in with GitHub". GitHub SSO is unavailable until this is set. |
| `GITHUB_CLIENT_SECRET` | *(unset)* | **Yes** (to enable GitHub SSO) | OAuth client secret for "Sign in with GitHub". |

## Redis Cache

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `REDIS_URL` | Built from `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`/`REDIS_PASSWORD` (e.g. `redis://localhost:6379/0`) | **Yes** (production) | Full Redis connection URL. Takes precedence over the individual `REDIS_HOST`/`REDIS_PORT`/`REDIS_DB`/`REDIS_PASSWORD` variables when set directly. |
| `REDIS_HOST` | `localhost` | No | Redis server hostname, used to build `REDIS_URL` when it isn't set explicitly. |
| `REDIS_PORT` | `6379` | No | Redis server port, used to build `REDIS_URL`. |
| `REDIS_DB` | `0` | No | Redis logical database index, used to build `REDIS_URL`. |
| `REDIS_PASSWORD` | *(unset)* | No | Redis password, URL-encoded and used to build `REDIS_URL` when present. |
| `REDIS_TIMEOUT_SECONDS` | `2.0` | No | Socket timeout (seconds) for Redis operations. |
| `REDIS_COMPRESSION_THRESHOLD` | *(unset, built-in default applies)* | No | Minimum payload size (bytes) above which cached values are compressed before storage. |
| `REDIS_COMPRESSION_LEVEL` | *(unset, built-in default applies)* | No | Compression level used when compressing cached values above the threshold. |
| `REDIS_CACHE_TTL` | `3600` (1 hour) | No | Default cache entry TTL in seconds. |
| `SESSION_TTL` | `900` (15 minutes) | No | TTL for cached session state entries. |
| `FAISS_INDEX_TTL` | `86400` (24 hours) | No | TTL for cached FAISS index data. |
| `ANALYSIS_RESULTS_TTL` | `7200` (2 hours) | No | TTL for cached analysis results. |
| `LOGIN_LOCKOUT_TTL` | `900` (15 minutes) | No | TTL for login-lockout tracking entries. |
| `UPLOAD_RATE_TTL` | `3600` (1 hour) | No | TTL for upload rate-limiting counters. |
| `BADGE_TTL` | `86400` (24 hours) | No | TTL for cached badge/achievement buffer data. |
| `DEFAULT_TTL` | `86400` (24 hours) | No | Fallback TTL applied to any cache key without an explicit TTL. |

## Plagiarism Detection & Embeddings

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `THRESHOLD_CONFIG_PATH` | `config/thresholds.json` (relative to repo root) | No | Path to an optional JSON file with `plagiarism`/`medium`/`high` similarity boundaries. If the file is missing, the built-in default thresholds (`0.59` / `0.75` / `0.90`) are used unchanged. |
| `EMBEDDING_BATCH_SIZE` | `32` | No | Number of text chunks embedded per batch when generating vectors. |
| `DEFAULT_DIFF_MIN_MATCH_LENGTH` | `4` | No | Minimum consecutive words required for a side-by-side diff highlight match. |
| `SEMANTIC_PLAGIARISM_MODEL` | Built-in default model name (`paraphrase-multilingual-MiniLM-L12-v2`) | No | SentenceTransformers model used to generate semantic embeddings. |
| `SEMANTIC_PLAGIARISM_FALLBACK_MODEL` | `all-MiniLM-L6-v2` | No | Fallback embedding model used if the primary model fails to load. |
| `AI_DETECTION_MODEL` | Built-in default AI-detection model name | No | Model used for AI-generated-text detection. |
| `PARSER_MAX_BATCH_SIZE` | `50` | No | Maximum number of documents processed per parsing batch. |
| `HF_HOME` | `~/.cache/huggingface` | No | Hugging Face cache root directory. |
| `HF_HUB_CACHE` | *(unset, falls back to `TRANSFORMERS_CACHE`)* | No | Preferred Hugging Face Hub cache directory. |
| `TRANSFORMERS_CACHE` | *(unset)* | No | Fallback cache directory for downloaded transformer models when `HF_HUB_CACHE` isn't set. |
| `STOPWORDS_FILE` | *(built-in default stopwords list)* | No | Path to a custom stopwords file for lexical analysis. |
| `TESSERACT_CMD` | *(auto-detected on `PATH`)* | No | Explicit path to the Tesseract OCR binary, for OCR-based text extraction from scanned/image documents. |

## Webhooks & Notifications

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `PLAGIARISM_WEBHOOK_URL` | *(unset)* | **Yes** (in deployments listing it in `REQUIRED_ENV_VARS`) | Webhook URL notified when a plagiarism incident is flagged. |
| `LMS_WEBHOOK_URL` | *(unset, falls back to `PLAGIARISM_WEBHOOK_URL`)* | No | Webhook URL used specifically for LMS-integration notifications. |
| `WEBHOOK_URL` | *(unset)* | No | Generic outbound webhook URL used by notification components. |
| `WEBHOOK_SECRET_KEY` | *(empty)* | No | Secret used to sign outbound webhook payloads (e.g. HMAC signature header), allowing receivers to verify authenticity. |
| `ALLOWED_WEBHOOK_DOMAINS` | *(empty, falls back to built-in allow-list)* | No | Comma-separated list of domains webhook URLs are permitted to target, to prevent SSRF via arbitrary webhook destinations. |
| `SLACK_WEBHOOK_URL` | *(unset)* | No | Slack incoming-webhook URL for notification integrations (currently referenced in commented-out code pending full integration). |
| `SECURITY_AUDIT_ALERT_WEBHOOK_URL` | *(unset)* | No | Webhook URL notified on security-audit alert events. |

## Email (SMTP)

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `SMTP_SERVER` | *(unset)* | **Yes** (to enable email sending) | SMTP server hostname. Email features (daily summaries, notifications) are skipped when unset. |
| `SMTP_PORT` | `587` | No | SMTP server port. |
| `SMTP_USERNAME` | *(unset)* | **Yes** (to enable email sending) | SMTP authentication username. |
| `SMTP_PASSWORD` | *(unset)* | **Yes** (to enable email sending) | SMTP authentication password. |
| `FROM_EMAIL` | *(falls back to `SMTP_USERNAME`)* | No | "From" address used on outgoing emails. |
| `ADMIN_EMAIL` | *(unset)* | No | Administrator email address used as the recipient for daily summary reports. |

## File Handling & Storage

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `MAX_UPLOAD_SIZE_BYTES` | `52428800` (50 MB) | No | Maximum allowed upload size, in bytes, for document uploads. |
| `BACKUP_DIR` | *(empty, falls back to built-in default backup path)* | No | Directory where database backup snapshots are written. |
| `BACKUP_IDLE_TIMEOUT_MINUTES` | `30` | No | Idle timeout (minutes) before an in-progress backup operation is considered stale. |
| `LOCK_TIMEOUT_SECONDS` | `30` | No | Timeout (seconds) for acquiring internal file/resource locks. |
| `CORPUS_RESCAN_INTERVAL_MINUTES` | `0` (disabled) | No | Interval, in minutes, for automatically rescanning the corpus directory for new documents. `0` disables automatic rescanning. |
| `GOOGLE_DRIVE_API_KEY` | *(unset)* | **Yes** (to enable Google Drive import) | API key used for importing documents from Google Drive. |

## Rate Limiting

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `TOKEN_BUCKET_CAPACITY` | *(unset, built-in default applies)* | No | Maximum burst capacity for the token-bucket rate limiter. |
| `TOKEN_BUCKET_REFILL_RATE` | *(unset, built-in default applies)* | No | Refill rate (tokens per second) for the token-bucket rate limiter. |

## Offline Mode

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `OFFLINE_MODE` | `false` | No | Master switch. Set to `true` to run without external network dependencies (no external APIs, telemetry, or live model downloads). |
| `OFFLINE_USE_LOCAL_CACHE` | `true` | No | When offline mode is enabled, use locally cached models/data instead of attempting network fetches. |
| `OFFLINE_CACHE_DIR` | `.cache/offline` | No | Directory used for general offline-mode caching. |
| `OFFLINE_MODEL_CACHE_DIR` | `.cache/models` | No | Directory used for caching ML models in offline mode. |
| `OFFLINE_PRELOAD_MODELS` | `true` | No | Preload models into cache at startup when offline mode is enabled. |
| `OFFLINE_DISABLE_TELEMETRY` | `true` | No | Disable telemetry/tracing reporting when offline mode is enabled. |
| `OFFLINE_DISABLE_EXTERNAL_APIS` | `true` | No | Disable calls to external APIs (webhooks, SSO, etc.) when offline mode is enabled. |
| `OFFLINE_USE_FALLBACK_EMBEDDING` | `true` | No | Use the lightweight fallback embedding model instead of the primary model when offline mode is enabled. |
| `OFFLINE_MAX_CACHE_SIZE_MB` | `500` | No | Maximum size (MB) of the offline cache before old entries are evicted. |
| `OFFLINE_AUTO_CLEANUP` | `true` | No | Automatically clean up stale offline-cache entries. |
| `OFFLINE_CLEANUP_INTERVAL_HOURS` | `24` | No | Interval (hours) between automatic offline-cache cleanup passes. |

## Observability (Tracing)

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `OTEL_SERVICE_NAME` | `semantic-plagiarism-detector` | No | Service name reported to the OpenTelemetry collector. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset — tracing export disabled)* | No | OTLP endpoint URL that traces are exported to. Tracing export is a no-op until this is set. |

## Miscellaneous / System

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `PDF_FOOTER_TEXT` | *(empty, falls back to built-in default footer)* | No | Custom footer text printed on generated PDF reports. |
| `TMPDIR` / `TMP` / `TEMP` | *(OS default temp directory)* | No | Standard OS temp-directory variables, respected for locating scratch space when writing intermediate files. |

## AWS S3 Report Export

| Variable Name | Default Value | Required | Description |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | *(unset — default credential chain used)* | No* | Access key used to sign S3 upload requests. Must be set together with the secret key; omit both to rely on boto3's standard credential chain (IAM roles, SSO, etc.). |
| `AWS_SECRET_ACCESS_KEY` | *(unset — default credential chain used)* | No* | Secret key paired with `AWS_ACCESS_KEY_ID`. |
| `AWS_S3_BUCKET` | *(unset — bucket must be passed explicitly)* | No | Default destination bucket for report exports (`src/utils/s3_export.py`) when no bucket is passed to `upload_to_s3()`. |

*S3 export additionally requires the optional `boto3` package (`pip install boto3`); uploads default to region `us-east-1`.

---

## Notes

- **Required (production)** variables have safe insecure defaults for local development but **must** be explicitly set before deploying to production; the app will run without them but with reduced security or missing functionality.
- **Required (to enable X)** variables gate an optional feature (SSO provider, email sending, Google Drive import, webhooks); the rest of the app functions normally without them, with that specific feature disabled.
- `REDIS_URL`, `PLAGIARISM_WEBHOOK_URL`, and `API_BEARER_TOKEN` are checked at startup against a `REQUIRED_ENV_VARS` list (see `app/streamlit_app.py`); missing values produce a startup warning rather than a hard failure, since some deployments may not need every integration.
- Boolean-style variables (`OFFLINE_MODE`, `ENABLE_HSTS`, etc.) accept case-insensitive `true`/`1`/`yes`-style truthy strings; any other value (including unset) is treated as `false`.
