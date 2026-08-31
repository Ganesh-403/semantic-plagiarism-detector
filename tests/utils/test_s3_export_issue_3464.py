"""
tests/utils/test_s3_export_issue_3464.py
----------------------------------------
Tests for AWS S3 report export integration (Issue #3464).

Verifies that upload_to_s3():
* validates its inputs (payload, object key, bucket resolution),
* reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_S3_BUCKET from the
  environment and passes credentials to the boto3 client,
* raises a helpful ImportError when boto3 is not installed,
* uploads via put_object() with the expected Bucket/Key/Body.
"""

import sys
from unittest.mock import MagicMock

import pytest

from src.utils.s3_export import (
    DEFAULT_REGION,
    ENV_S3_BUCKET,
    get_s3_config,
    upload_to_s3,
)


@pytest.fixture
def mock_boto3(monkeypatch):
    """Inject a mocked boto3 module into sys.modules for the duration of a test."""
    boto3_mock = MagicMock()
    saved = sys.modules.get("boto3")
    monkeypatch.setitem(sys.modules, "boto3", boto3_mock)
    yield boto3_mock
    if saved is not None:
        sys.modules["boto3"] = saved


@pytest.fixture(autouse=True)
def clean_aws_env(monkeypatch):
    """Remove AWS variables so tests are isolated from the host environment."""
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", ENV_S3_BUCKET):
        monkeypatch.delenv(var, raising=False)


def test_get_s3_config_reads_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv(ENV_S3_BUCKET, "archive-bucket")

    config = get_s3_config()

    assert config == {
        "access_key_id": "AKIA_TEST",
        "secret_access_key": "secret",
        "bucket_name": "archive-bucket",
    }


def test_get_s3_config_missing_variables_return_none():
    assert get_s3_config() == {
        "access_key_id": None,
        "secret_access_key": None,
        "bucket_name": None,
    }


def test_upload_rejects_empty_payload(mock_boto3):
    with pytest.raises(ValueError, match="non-empty bytes"):
        upload_to_s3(b"", bucket_name="bucket", object_key="r.pdf")


def test_upload_rejects_non_bytes_payload(mock_boto3):
    with pytest.raises(ValueError, match="non-empty bytes"):
        upload_to_s3("text report", bucket_name="bucket", object_key="r.pdf")


def test_upload_rejects_empty_object_key(mock_boto3):
    with pytest.raises(ValueError, match="object_key"):
        upload_to_s3(b"report", bucket_name="bucket", object_key="   ")


def test_upload_requires_bucket_argument_or_env(mock_boto3):
    with pytest.raises(ValueError, match=ENV_S3_BUCKET):
        upload_to_s3(b"report", bucket_name=None, object_key="r.pdf")


def test_upload_requires_paired_credentials(monkeypatch, mock_boto3):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_ONLY")
    with pytest.raises(ValueError, match="together"):
        upload_to_s3(b"report", bucket_name="bucket", object_key="r.pdf")


def test_upload_without_boto3_installed_raises_import_error(
    monkeypatch,
):
    """A missing boto3 package must surface a clear, actionable error."""
    monkeypatch.setitem(sys.modules, "boto3", None)  # Forces `import boto3` to fail.

    with pytest.raises(ImportError, match="pip install boto3"):
        upload_to_s3(b"report", bucket_name="bucket", object_key="r.pdf")


def test_upload_uses_env_bucket_and_explicit_credentials(monkeypatch, mock_boto3):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv(ENV_S3_BUCKET, "archive-bucket")

    client_mock = mock_boto3.client.return_value
    client_mock.put_object.return_value = {"ETag": '"abc123"'}

    result = upload_to_s3(b"report-bytes", object_key="reports/2026/r.pdf")

    mock_boto3.client.assert_called_once_with(
        "s3",
        region_name=DEFAULT_REGION,
        aws_access_key_id="AKIA_TEST",
        aws_secret_access_key="secret",
    )
    client_mock.put_object.assert_called_once_with(
        Bucket="archive-bucket",
        Key="reports/2026/r.pdf",
        Body=b"report-bytes",
    )
    assert result == {
        "bucket": "archive-bucket",
        "key": "reports/2026/r.pdf",
        "region": DEFAULT_REGION,
        "etag": "abc123",
    }


def test_upload_prefers_explicit_bucket_and_region(monkeypatch, mock_boto3):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    client_mock = mock_boto3.client.return_value
    client_mock.put_object.return_value = {"ETag": '"etag-xyz"'}

    result = upload_to_s3(
        b"report-bytes",
        bucket_name="explicit-bucket",
        object_key="r.json",
        region="eu-west-1",
    )

    mock_boto3.client.assert_called_once_with(
        "s3",
        region_name="eu-west-1",
        aws_access_key_id="AKIA_TEST",
        aws_secret_access_key="secret",
    )
    client_mock.put_object.assert_called_once_with(
        Bucket="explicit-bucket",
        Key="r.json",
        Body=b"report-bytes",
    )
    assert result["bucket"] == "explicit-bucket"
    assert result["region"] == "eu-west-1"
    assert result["etag"] == "etag-xyz"


def test_upload_falls_back_to_default_credential_chain(mock_boto3):
    """With no static keys configured, boto3's default chain is used as-is."""
    client_mock = mock_boto3.client.return_value
    client_mock.put_object.return_value = {}

    result = upload_to_s3(b"data", bucket_name="bucket", object_key="k.txt")

    mock_boto3.client.assert_called_once_with("s3", region_name=DEFAULT_REGION)
    assert result["etag"] == ""
