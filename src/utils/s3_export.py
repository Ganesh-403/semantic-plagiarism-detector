"""src/utils/s3_export.py - Export plagiarism reports to AWS S3 for institutional archival (Issue #3464).

Provides ``upload_to_s3()`` so generated reports can be pushed straight to a
customer-managed S3 bucket instead of (or in addition to) local storage.

Configuration is read from environment variables:

- ``AWS_ACCESS_KEY_ID``: Static access key used to sign requests.
- ``AWS_SECRET_ACCESS_KEY``: Secret key paired with the access key.
- ``AWS_S3_BUCKET``: Default bucket used when no bucket is passed explicitly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"

ENV_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
ENV_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
ENV_S3_BUCKET = "AWS_S3_BUCKET"


def get_s3_config() -> Dict[str, Optional[str]]:
    """Read AWS S3 export settings from the environment.

    Returns:
        Dict[str, Optional[str]]: Mapping with ``access_key_id``,
        ``secret_access_key`` and ``bucket_name`` entries. Values are
        ``None`` when the corresponding variable is unset. Secret values
        are returned for client construction only and must never be logged.
    """
    return {
        "access_key_id": os.getenv(ENV_ACCESS_KEY_ID),
        "secret_access_key": os.getenv(ENV_SECRET_ACCESS_KEY),
        "bucket_name": os.getenv(ENV_S3_BUCKET),
    }


def _build_client(region: str):
    """Create a boto3 S3 client honoring explicit credentials when available.

    When both credential variables are configured they are passed to boto3
    explicitly; when neither is set, boto3's standard credential chain
    (instance profiles, SSO, cached sessions, ...) applies instead. Supplying
    only one of the two variables is treated as a configuration error.

    Args:
        region: AWS region to create the client in.

    Returns:
        A connected boto3 S3 client.

    Raises:
        ImportError: If the optional ``boto3`` package is not installed.
        ValueError: If exactly one of the two credential variables is set.
    """
    try:
        import boto3  # Optional dependency; imported lazily on first upload.
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 report export. "
            "Install it with: pip install boto3"
        ) from exc

    config = get_s3_config()
    access_key_id = config["access_key_id"]
    secret_access_key = config["secret_access_key"]

    if bool(access_key_id) != bool(secret_access_key):
        raise ValueError(
            f"AWS S3 export requires both {ENV_ACCESS_KEY_ID} and "
            f"{ENV_SECRET_ACCESS_KEY} to be set together."
        )

    if access_key_id and secret_access_key:
        logger.debug("Using AWS credentials from environment for S3 export.")
        return boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    logger.debug(
        "No static AWS credentials configured; falling back to the "
        "default boto3 credential chain."
    )
    return boto3.client("s3", region_name=region)


def upload_to_s3(
    file_bytes: bytes,
    bucket_name: Optional[str] = None,
    object_key: str = "",
    region: str = DEFAULT_REGION,
) -> Dict[str, str]:
    """Upload report bytes to an AWS S3 bucket for institutional archival.

    Args:
        file_bytes: Report content to store (e.g., PDF/JSON report bytes).
        bucket_name: Target bucket. Falls back to the ``AWS_S3_BUCKET``
            environment variable when omitted.
        object_key: Key (path) under which the object is stored, e.g.
            ``reports/2026/invoice_batch.pdf``.
        region: AWS region of the bucket. Defaults to ``us-east-1``.

    Returns:
        Dict[str, str]: Confirmation mapping with ``bucket``, ``key``,
        ``region``, and the stored object's ``etag``.

    Raises:
        ImportError: If ``boto3`` is not installed.
        ValueError: If arguments are missing/invalid or the AWS credential
            configuration is incomplete.
        Exception: Propagates boto3/network failures after logging context.
    """
    if not isinstance(file_bytes, (bytes, bytearray)) or len(file_bytes) == 0:
        raise ValueError("file_bytes must be a non-empty bytes payload.")

    if not isinstance(object_key, str) or not object_key.strip():
        raise ValueError("object_key must be a non-empty string.")

    resolved_bucket = bucket_name or get_s3_config()["bucket_name"]
    if not resolved_bucket:
        raise ValueError(
            f"No S3 bucket specified: pass bucket_name or set {ENV_S3_BUCKET}."
        )

    client = _build_client(region)

    logger.info(
        "Uploading %d byte(s) to s3://%s/%s (region=%s)",
        len(file_bytes),
        resolved_bucket,
        object_key,
        region,
    )
    response: Dict[str, Any] = client.put_object(
        Bucket=resolved_bucket,
        Key=object_key,
        Body=bytes(file_bytes),
    )

    etag = str(response.get("ETag", "")).strip('"')
    logger.info("S3 upload complete: etag=%s", etag)

    return {
        "bucket": resolved_bucket,
        "key": object_key,
        "region": region,
        "etag": etag,
    }
