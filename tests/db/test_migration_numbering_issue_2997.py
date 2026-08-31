"""Unit tests for sequential migration numbering and discovery in auth and corpus schema (Issue #2997)."""

import sqlite3
import pytest

from src.db.migrations.auth import (
    AUTH_MIGRATIONS,
    AUTH_SCHEMA_VERSION,
    AUTH_DOWN_MIGRATIONS,
    migrate_auth_database,
)
from src.db.migrations.corpus import CORPUS_MIGRATIONS, CORPUS_SCHEMA_VERSION


def test_auth_migration_numbering_sequential():
    """Verify auth migration dict keys are strictly 1..N with matching function names."""
    assert len(AUTH_MIGRATIONS) == AUTH_SCHEMA_VERSION
    assert sorted(AUTH_MIGRATIONS.keys()) == list(range(1, AUTH_SCHEMA_VERSION + 1))

    for idx, func in AUTH_MIGRATIONS.items():
        expected_prefix = f"migration_{idx:03d}_"
        assert func.__name__.startswith(expected_prefix), (
            f"Migration {idx} function name {func.__name__} does not start with {expected_prefix}"
        )


def test_auth_down_migrations_sequential():
    """Verify auth down migration dict keys match up migration keys and function names."""
    assert len(AUTH_DOWN_MIGRATIONS) == AUTH_SCHEMA_VERSION
    assert sorted(AUTH_DOWN_MIGRATIONS.keys()) == list(range(1, AUTH_SCHEMA_VERSION + 1))

    for idx, func in AUTH_DOWN_MIGRATIONS.items():
        expected_prefix = f"down_{idx:03d}_"
        assert func.__name__.startswith(expected_prefix), (
            f"Down migration {idx} function name {func.__name__} does not start with {expected_prefix}"
        )


def test_auth_migration_execution(tmp_path):
    """Verify migrate_auth_database runs cleanly to latest schema version without errors."""
    db_file = tmp_path / "test_auth_seq.db"
    with sqlite3.connect(db_file) as conn:
        final_ver = migrate_auth_database(conn)
        assert final_ver == AUTH_SCHEMA_VERSION
