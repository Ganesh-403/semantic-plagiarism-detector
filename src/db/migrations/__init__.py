# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Public SQLite migration API."""

from .auth import (
    AUTH_DOWN_MIGRATIONS,
    AUTH_MIGRATIONS,
    AUTH_SCHEMA_VERSION,
    migrate_auth_database,
)
from .common import (
    check_table_exists,
    column_exists,
    delete_all_if_table_exists,
    ensure_migration_history_table,
    get_latest_applied_migration,
    get_migration_status,
    get_user_version,
    index_exists,
    rollback_migration,
    run_migrations,
    table_exists,
)
from .corpus import (
    CORPUS_DOWN_MIGRATIONS,
    CORPUS_MIGRATIONS,
    CORPUS_SCHEMA_VERSION,
    migrate_corpus_database,
)

__all__ = [
    "AUTH_DOWN_MIGRATIONS",
    "AUTH_MIGRATIONS",
    "AUTH_SCHEMA_VERSION",
    "CORPUS_DOWN_MIGRATIONS",
    "CORPUS_MIGRATIONS",
    "CORPUS_SCHEMA_VERSION",
    "column_exists",
    "delete_all_if_table_exists",
    "ensure_migration_history_table",
    "get_latest_applied_migration",
    "get_migration_status",
    "get_user_version",
    "index_exists",
    "migrate_auth_database",
    "migrate_corpus_database",
    "rollback_migration",
    "run_migrations",
    "table_exists",
    "check_table_exists",
]
