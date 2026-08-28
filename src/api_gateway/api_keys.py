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

"""API Key Management Service."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from src.api_gateway.models import APIKeyRecord


def hash_api_key(api_key: str) -> str:
    """Compute SHA-256 hash of raw API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class APIKeyService:
    """Service managing API key generation, validation, rotation, and revocation."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKeyRecord] = {}  # key_id -> record
        self._hash_index: dict[str, str] = {}  # key_hash -> key_id

    def create_key(
        self,
        name: str,
        expires_in_days: int | None = None,
        rate_limit: int = 100,
    ) -> tuple[APIKeyRecord, str]:
        """Generate a new secure API key.

        Returns:
            Tuple of (APIKeyRecord, raw_api_key)
        """
        raw_key = f"spd_live_{secrets.token_urlsafe(32)}"
        key_hash = hash_api_key(raw_key)

        key_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        expires_at = (
            created_at + timedelta(days=expires_in_days)
            if expires_in_days is not None
            else None
        )

        record = APIKeyRecord(
            id=key_id,
            name=name,
            key_hash=key_hash,
            created_at=created_at,
            expires_at=expires_at,
            rate_limit=rate_limit,
        )

        self._keys[key_id] = record
        self._hash_index[key_hash] = key_id

        return record, raw_key

    def validate_key(self, raw_key: str) -> APIKeyRecord | None:
        """Validate raw API key. Update last_used_at on success."""
        if not raw_key:
            return None

        key_hash = hash_api_key(raw_key)
        key_id = self._hash_index.get(key_hash)
        if not key_id:
            return None

        record = self._keys.get(key_id)
        if record is None or not record.is_active:
            return None

        record.last_used_at = datetime.now(timezone.utc)
        return record

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an active API key by ID."""
        record = self._keys.get(key_id)
        if record is None or record.revoked_at is not None:
            return False

        record.revoked_at = datetime.now(timezone.utc)
        return True

    def rotate_key(
        self, key_id: str, expires_in_days: int | None = None
    ) -> tuple[APIKeyRecord, str] | None:
        """Rotate an existing key by revoking the old one and creating a new key with the same name."""
        old_record = self._keys.get(key_id)
        if old_record is None or old_record.revoked_at is not None:
            return None

        # Revoke old key
        self.revoke_key(key_id)

        # Create new key with same name and rate_limit
        new_record, new_raw_key = self.create_key(
            name=f"{old_record.name} (rotated)",
            expires_in_days=expires_in_days,
            rate_limit=old_record.rate_limit,
        )

        return new_record, new_raw_key

    def get_key_by_id(self, key_id: str) -> APIKeyRecord | None:
        """Retrieve API key metadata record by key ID."""
        return self._keys.get(key_id)

    def list_keys(self) -> list[APIKeyRecord]:
        """List metadata for all API keys."""
        return list(self._keys.values())
