"""
tests/utils/test_migrate_redis_keys.py
---------------------------------------
Unit tests for the Redis key migration script and CacheNamespace legacy prefix removal (Issue #2803).
"""

from unittest.mock import Mock

import pytest

from scripts.migrate_redis_keys import map_legacy_key, migrate_redis_keys
from src.utils.redis_cache import CacheNamespace
from src.version import APP_VERSION


class TestCacheNamespaceEnum:
    """Verify that CacheNamespace contains only modern spd:v1:* namespaces and no LEGACY_* attributes."""

    def test_legacy_prefixes_removed(self):
        """Ensure all LEGACY_* attributes are removed from CacheNamespace."""
        enum_names = [e.name for e in CacheNamespace]
        legacy_names = [name for name in enum_names if name.startswith("LEGACY_")]
        assert legacy_names == [], (
            f"Found legacy attributes in CacheNamespace: {legacy_names}"
        )

    def test_modern_namespaces_present(self):
        """Ensure standard modern namespaces are present."""
        assert CacheNamespace.SESSION.value == "spd:v1:session"
        assert CacheNamespace.FAISS.value == "spd:v1:faiss"
        assert CacheNamespace.ANALYSIS.value == "spd:v1:analysis"
        assert CacheNamespace.LOGIN_ATTEMPTS.value == "spd:v1:login_attempts"
        assert CacheNamespace.UPLOADS.value == "spd:v1:uploads"


class TestMapLegacyKey:
    """Test mapping of legacy key formats to modern spd:v1:* format."""

    @pytest.mark.parametrize(
        "legacy_key, expected_key",
        [
            ("login_attempts:user123", f"spd:v1:login_attempts:{APP_VERSION}:user123"),
            ("upload_count:alice", f"spd:v1:uploads:{APP_VERSION}:alice"),
            ("similarity:abc123hash", f"spd:v1:analysis:{APP_VERSION}:abc123hash"),
            ("analysis:def456hash", f"spd:v1:analysis:{APP_VERSION}:def456hash"),
            ("doc:1042", f"spd:v1:analysis:{APP_VERSION}:doc:1042"),
            ("faiss_index", f"spd:v1:faiss:{APP_VERSION}:index:corpus_index"),
            ("faiss_index:custom_index", f"spd:v1:faiss:{APP_VERSION}:index:custom_index"),
            ("session:sess_abc:user", f"spd:v1:session:{APP_VERSION}:sess_abc:user"),
        ],
    )
    def test_map_legacy_keys(self, legacy_key: str, expected_key: str):
        """Verify proper transformation of legacy key patterns."""
        assert map_legacy_key(legacy_key) == expected_key

    @pytest.mark.parametrize(
        "modern_or_unrelated_key",
        [
            "spd:v1:session:sess1:auth",
            "spd:v1:faiss:index:corpus_index",
            "spd:v1:analysis:some_hash",
            "spd:v1:login_attempts:id",
            "spd:v1:uploads:username",
            "unrelated_key_prefix:data",
            "other:1234",
        ],
    )
    def test_map_non_legacy_keys_returns_none(self, modern_or_unrelated_key: str):
        """Verify modern or unrecognized keys return None and are not modified."""
        assert map_legacy_key(modern_or_unrelated_key) is None


class TestMigrateRedisKeys:
    """Test migration execution logic (live and dry-run)."""

    def test_migrate_dry_run(self):
        """Test dry run scans keys but does not call rename."""
        mock_client = Mock()
        mock_client.scan_iter.return_value = [
            b"login_attempts:user1",
            b"upload_count:john",
            b"spd:v1:session:sess123:key",
            b"faiss_index",
        ]

        stats = migrate_redis_keys(mock_client, dry_run=True)

        assert stats["scanned"] == 4
        assert stats["migrated"] == 3
        assert stats["skipped"] == 1
        assert stats["errors"] == 0
        assert len(stats["renames"]) == 3
        mock_client.rename.assert_not_called()

    def test_migrate_live(self):
        """Test live migration calls rename for legacy keys."""
        mock_client = Mock()
        mock_client.scan_iter.return_value = [
            b"similarity:query123",
            b"doc:99",
            b"analysis:result1",
        ]

        stats = migrate_redis_keys(mock_client, dry_run=False)

        assert stats["scanned"] == 3
        assert stats["migrated"] == 3
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert mock_client.rename.call_count == 3
        mock_client.rename.assert_any_call(
            b"similarity:query123", f"spd:v1:analysis:{APP_VERSION}:query123".encode()
        )
        mock_client.rename.assert_any_call(b"doc:99", f"spd:v1:analysis:{APP_VERSION}:doc:99".encode())
        mock_client.rename.assert_any_call(
            b"analysis:result1", f"spd:v1:analysis:{APP_VERSION}:result1".encode()
        )

    def test_migrate_handles_rename_error(self):
        """Test error handling when Redis rename raises an exception."""
        mock_client = Mock()
        mock_client.scan_iter.return_value = [b"login_attempts:bad_key"]
        mock_client.rename.side_effect = Exception("Redis error during rename")

        stats = migrate_redis_keys(mock_client, dry_run=False)

        assert stats["scanned"] == 1
        assert stats["migrated"] == 0
        assert stats["errors"] == 1
