"""Public SQLite migration API."""

from .auth import AUTH_MIGRATIONS, AUTH_SCHEMA_VERSION, migrate_auth_database
from .common import (column_exists, delete_all_if_table_exists,
                     get_user_version, index_exists, run_migrations,
                     table_exists)
from .corpus import (CORPUS_MIGRATIONS, CORPUS_SCHEMA_VERSION,
                     migrate_corpus_database)

__all__ = [
    "AUTH_MIGRATIONS",
    "AUTH_SCHEMA_VERSION",
    "CORPUS_MIGRATIONS",
    "CORPUS_SCHEMA_VERSION",
    "column_exists",
    "delete_all_if_table_exists",
    "get_user_version",
    "index_exists",
    "migrate_auth_database",
    "migrate_corpus_database",
    "run_migrations",
    "table_exists",
]
