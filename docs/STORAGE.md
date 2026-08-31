# Storage: Database & FAISS Index File Locations

This document describes where the application stores its SQLite databases and
FAISS index files, how those paths are discovered and configured, and the
backup / cleanup mechanisms available.

---

## Overview

The application persists data in three primary artifacts:

| Artifact | Default location | Contents |
|----------|------------------|----------|
| `corpus.db` | `<repo>/data/corpus.db` | Documents, text chunks, embeddings, plagiarism incidents |
| `users.db`  | `<repo>/users.db` | User authentication, roles, permissions, SSO state |
| `corpus.index` | `<repo>/corpus.index` | FAISS ANN index for similarity search |

Both SQLite databases and the FAISS index are discovered dynamically; a
`storage_metrics` module reports their combined disk usage.

---

## Path Centralization in `app_config.py`

All storage paths are centralized in `src/core/app_config.py` (issue #618) so
the same constants are used everywhere instead of each module computing its own
path.

```python
# Repo root: two levels above src/core/app_config.py
_REPO_ROOT = Path(__file__).resolve().parents[2]

CORPUS_DB_PATH: Final[Path] = _REPO_ROOT / "data" / "corpus.db"
AUTH_DB_PATH: Final[Path] = _REPO_ROOT / "users.db"
FAISS_INDEX_PATH: Final[Path] = _REPO_ROOT / "corpus.index"
HEALTHZ_DB_PATHS: Final[tuple[Path, ...]] = (CORPUS_DB_PATH, AUTH_DB_PATH)

FALLBACK_DATA_DIR: Final[Path] = (
    Path(tempfile.gettempdir()) / "semantic_plagiarism_detector" / "data"
)
FALLBACK_CORPUS_DB_PATH: Final[Path] = FALLBACK_DATA_DIR / "corpus.db"
```

Notes:

- **Corpus DB** lives in `<repo>/data/corpus.db`; the `data/` directory is
  created on demand by the DB layer (`os.makedirs(..., exist_ok=True)`).
- **Auth DB** and the **FAISS index** live at the repo root (`<repo>/users.db`,
  `<repo>/corpus.index`) for historical compatibility.
- **Fallback**: if the primary data directory is not writable, the corpus DB
  falls back to `<tmpdir>/semantic_plagiarism_detector/data/corpus.db`.
- **Per-module mutators** are preserved as thin wrappers:
  `src/db/corpus_db.configure_db_path(path)` and
  `src/db/auth.configure_db_path(path)` override the effective paths at
  runtime (used heavily by the test suite for isolation).

---

## How Files Are Discovered (`storage_metrics.py`)

`src/utils/storage_metrics.py` calculates disk usage by discovering the
database and index files:

### SQLite databases

`get_sqlite_db_paths()` collects:

1. `src.db.corpus_db.get_corpus_db_path()` — the configured corpus DB path.
2. `src.db.auth._DB_PATH` — the auth DB path.
3. `src.db.incidents.DEFAULT_DB_PATH` — the incidents DB path.
4. Any `*.db` file in the repo root **or** the `data/` directory.

Paths are resolved and deduplicated before being returned.

### FAISS indexes

`get_faiss_index_paths()` collects:

1. `<repo>/corpus.index`
2. `<repo>/data/corpus.index`
3. Any `*.index` file in the repo root **or** the `data/` directory.

### Usage calculation

`calculate_storage_usage()` sums the sizes of the discovered files and returns
a dict with byte and megabyte values plus a formatted total, e.g.
`"1.25 MB"`. This powers the storage usage dashboard and the `/healthz`
endpoint (via `HEALTHZ_DB_PATHS`).

---

## Backup & Cleanup (`database_backup.py`)

`src/db/database_backup.py` provides snapshot, backup, restore, and cleanup
operations:

| Function | Purpose |
|----------|---------|
| `create_sqlite_snapshot(path)` | Transactionally consistent snapshot of a SQLite DB as bytes |
| `create_database_backup(path)` | Writes a timestamped, gzip-compressed `.db.gz` backup into `backups/` (or a custom `backup_dir`) |
| `create_password_protected_backup(path)` | AES-256 encrypted backup (requires `pyzipper`) |
| `restore_database_backup(...)` / `restore(...)` | Restores a backup with strict path & content validation |
| `cleanup_old_backups(...)` | Removes backups older than a retention window |
| `optimize_database(path)` | Runs `VACUUM` to reclaim space |
| `checkpoint_wal_log(path)` | Checkpoints the SQLite WAL |

Backups are written as `<source>.db.<timestamp>.db.gz` (or `.db` when
`compress_backup=False`) into `DEFAULT_BACKUP_DIRECTORY` (the `backups/`
folder at the repo root).

### Security validation

`restore()` validates the backup **before** applying it:

- The source path must resolve inside the designated backup directory
  (`BackupRestoreSecurityError` otherwise).
- The source must be a regular file (no symlink tricks).
- On non-Windows systems, a world-writable backup file is rejected.
- The file must carry the SQLite header magic (`SQLite format 3\0`).

Backups are never applied from an unvalidated path, and cleanup keeps only
backups within the configured retention window.

---

## Related Documentation

- [DATABASE.md](./DATABASE.md) — schema details for `users.db` and `corpus.db`.
- [FAISS_MAPPING.md](./FAISS_MAPPING.md) — how embeddings map to the FAISS index.
- [DB_SCHEMA.md](./DB_SCHEMA.md) — full table definitions.
- [BACKUP & RESTORE security notes](./SECURITY_CHECKLIST.md) — security model.
