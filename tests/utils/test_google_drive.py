import os
from unittest.mock import Mock, patch

import pytest

from src.utils.google_drive import (
    bulk_download_drive_folder,
    check_folder_access,
    download_file_bytes,
    extract_google_drive_folder_id,
    get_drive_service,
    list_files_in_folder,
    validate_service_account_key,
)

def test_extract_google_drive_folder_id_valid_id():
    valid_id = "1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7"
    assert len(valid_id) == 33
    assert extract_google_drive_folder_id(valid_id) == valid_id

def test_extract_google_drive_folder_id_valid_url():
    valid_id = "1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7"
    url = f"https://drive.google.com/drive/folders/{valid_id}"
    assert extract_google_drive_folder_id(url) == valid_id

def test_extract_google_drive_folder_id_valid_url_with_query():
    valid_id = "1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7"
    url = f"https://drive.google.com/drive/folders/{valid_id}?usp=sharing"
    assert extract_google_drive_folder_id(url) == valid_id

def test_extract_google_drive_folder_id_malformed_url():
    url = "https://drive.google.com/drive/folders/shortid"
    assert extract_google_drive_folder_id(url) is None

def test_extract_google_drive_folder_id_empty_string():
    assert extract_google_drive_folder_id("") is None

def test_extract_google_drive_folder_id_random_string():
    assert extract_google_drive_folder_id("random_garbage_string_not_an_id") is None

def test_extract_google_drive_folder_id_unsupported_url():
    url = "https://google.com"
    assert extract_google_drive_folder_id(url) is None

def test_extract_google_drive_folder_id_whitespace():
    valid_id = "1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7"
    assert extract_google_drive_folder_id(f"  {valid_id}  ") == valid_id
    url = f"  https://drive.google.com/drive/folders/{valid_id}?usp=sharing  "
    assert extract_google_drive_folder_id(url) == valid_id

def test_extract_google_drive_folder_id_invalid_type():
    assert extract_google_drive_folder_id(None) is None
    assert extract_google_drive_folder_id(12345) is None


# ── get_drive_service tests ──────────────────────────────────────────────


@patch("src.utils.google_drive.build")
def test_get_drive_service_with_api_key(mock_build):
    service = get_drive_service(api_key="test-api-key")
    mock_build.assert_called_once_with(
        "drive", "v3", developerKey="test-api-key"
    )
    assert service == mock_build.return_value


@patch("src.utils.google_drive.build")
@patch("src.utils.google_drive.service_account")
def test_get_drive_service_with_service_account(mock_sa, mock_build):
    mock_creds = Mock()
    mock_sa.Credentials.from_service_account_info.return_value = mock_creds
    sa_info = {"type": "service_account"}

    service = get_drive_service(service_account_info=sa_info)

    mock_sa.Credentials.from_service_account_info.assert_called_once_with(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    mock_build.assert_called_once_with("drive", "v3", credentials=mock_creds)
    assert service == mock_build.return_value


@patch("src.utils.google_drive.build")
def test_get_drive_service_with_env_key(mock_build):
    with patch.dict(os.environ, {"GOOGLE_DRIVE_API_KEY": "env-key"}, clear=False):
        service = get_drive_service()
    mock_build.assert_called_once_with("drive", "v3", developerKey="env-key")
    assert service == mock_build.return_value


@patch("src.utils.google_drive.build")
def test_get_drive_service_no_credentials(mock_build):
    with patch.dict(os.environ, {"GOOGLE_DRIVE_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError, match="No API Key or Service Account"):
            get_drive_service()


# ── list_files_in_folder tests ───────────────────────────────────────────


def _mock_service_for_list(files):
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {
        "files": files
    }
    return service


def test_list_files_in_folder_returns_supported():
    files = [
        {"id": "1", "name": "report.pdf", "mimeType": "application/pdf"},
        {"id": "2", "name": "essay.docx", "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        {"id": "3", "name": "notes.txt", "mimeType": "text/plain"},
        {"id": "4", "name": "script.exe", "mimeType": "application/x-msdownload"},
    ]
    service = _mock_service_for_list(files)
    result = list_files_in_folder(service, "folder123")

    assert len(result) == 3
    assert result[0]["name"] == "report.pdf"
    assert result[1]["name"] == "essay.docx"
    assert result[2]["name"] == "notes.txt"

    service.files.return_value.list.assert_called_once_with(
        q="'folder123' in parents and trashed = false",
        pageSize=100,
        fields="nextPageToken, files(id, name, mimeType, size)",
    )


def test_list_files_in_folder_empty():
    service = _mock_service_for_list([])
    result = list_files_in_folder(service, "folder123")
    assert result == []


def test_list_files_in_folder_no_supported_extensions():
    files = [
        {"id": "1", "name": "script.exe", "mimeType": "application/x-msdownload"},
        {"id": "2", "name": "image.png", "mimeType": "image/png"},
    ]
    service = _mock_service_for_list(files)
    result = list_files_in_folder(service, "folder123")
    assert result == []


def test_list_files_in_folder_handles_api_error():
    service = Mock()
    service.files.return_value.list.return_value.execute.side_effect = \
        Exception("403 Forbidden")

    with pytest.raises(Exception, match="403 Forbidden"):
        list_files_in_folder(service, "folder123")


def test_list_files_in_folder_handles_not_found():
    service = Mock()
    service.files.return_value.list.return_value.execute.side_effect = \
        Exception("404 Not Found")

    with pytest.raises(Exception, match="404 Not Found"):
        list_files_in_folder(service, "folder123")


# ── download_file_bytes tests ────────────────────────────────────────────


@patch("src.utils.google_drive.MediaIoBaseDownload")
def test_download_file_bytes(mock_downloader_cls):
    mock_downloader = Mock()
    mock_downloader.next_chunk.side_effect = [(None, False), (None, True)]
    mock_downloader_cls.return_value = mock_downloader

    service = Mock()
    mock_request = Mock()
    service.files.return_value.get_media.return_value = mock_request

    result = download_file_bytes(service, "file123")

    service.files.return_value.get_media.assert_called_once_with(fileId="file123")
    mock_downloader_cls.assert_called_once()
    assert isinstance(result, bytes)


@patch("src.utils.google_drive.MediaIoBaseDownload")
def test_download_file_bytes_calls_progress_callback(mock_downloader_cls):
    status1 = Mock(resumable_progress=50, total_size=100)
    status2 = Mock(resumable_progress=100, total_size=100)
    mock_downloader = Mock()
    mock_downloader.next_chunk.side_effect = [(status1, False), (status2, True)]
    mock_downloader_cls.return_value = mock_downloader

    service = Mock()
    service.files.return_value.get_media.return_value = Mock()

    calls = []
    download_file_bytes(service, "file123", progress_callback=lambda d, t: calls.append((d, t)))

    # One callback per chunk, plus a guaranteed final 100% callback.
    assert calls[0] == (50, 100)
    assert calls[1] == (100, 100)
    assert calls[-1][0] == calls[-1][1]


@patch("src.utils.google_drive.MediaIoBaseDownload")
def test_download_file_bytes_progress_callback_optional(mock_downloader_cls):
    mock_downloader = Mock()
    mock_downloader.next_chunk.side_effect = [(None, False), (None, True)]
    mock_downloader_cls.return_value = mock_downloader

    service = Mock()
    service.files.return_value.get_media.return_value = Mock()

    # Should not raise when no progress_callback is supplied.
    result = download_file_bytes(service, "file123")
    assert isinstance(result, bytes)


@patch("src.utils.google_drive.MediaIoBaseDownload")
def test_download_file_bytes_handles_api_error(mock_downloader_cls):
    mock_downloader = Mock()
    mock_downloader.next_chunk.side_effect = Exception("403 Forbidden")
    mock_downloader_cls.return_value = mock_downloader

    service = Mock()
    service.files.return_value.get_media.return_value = Mock()

    with pytest.raises(Exception, match="403 Forbidden"):
        download_file_bytes(service, "file456")


# ── bulk_download_drive_folder tests ─────────────────────────────────────


@patch("src.utils.google_drive.download_file_bytes")
@patch("src.utils.google_drive.list_files_in_folder")
@patch("src.utils.google_drive.get_drive_service")
def test_bulk_download_drive_folder(mock_get_service, mock_list, mock_download):
    mock_list.return_value = [
        {"id": "f1", "name": "doc1.pdf"},
        {"id": "f2", "name": "doc2.docx"},
    ]
    mock_download.side_effect = [b"content1", b"content2"]

    result, names = bulk_download_drive_folder(
        "https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7",
        api_key="key",
    )

    assert len(result) == 2
    assert result["doc1.pdf"] == b"content1"
    assert result["doc2.docx"] == b"content2"
    assert names == ["doc1.pdf", "doc2.docx"]


@patch("src.utils.google_drive.download_file_bytes")
@patch("src.utils.google_drive.list_files_in_folder")
@patch("src.utils.google_drive.get_drive_service")
def test_bulk_download_drive_folder_handles_download_error(
    mock_get_service, mock_list, mock_download
):
    mock_list.return_value = [{"id": "f1", "name": "doc1.pdf"}]
    mock_download.side_effect = Exception("404 Not Found")

    with pytest.raises(Exception, match="404 Not Found"):
        bulk_download_drive_folder(
            "https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7",
            api_key="key",
        )


@patch("src.utils.google_drive.download_file_bytes")
@patch("src.utils.google_drive.list_files_in_folder")
@patch("src.utils.google_drive.get_drive_service")
def test_bulk_download_drive_folder_reports_aggregate_progress(
    mock_get_service, mock_list, mock_download
):
    mock_list.return_value = [
        {"id": "f1", "name": "doc1.pdf", "size": "100"},
        {"id": "f2", "name": "doc2.docx", "size": "200"},
    ]

    def fake_download(service, file_id, progress_callback=None):
        if progress_callback:
            if file_id == "f1":
                progress_callback(100, 100)
            else:
                progress_callback(200, 200)
        return b"x" * (100 if file_id == "f1" else 200)

    mock_download.side_effect = fake_download

    calls = []
    bulk_download_drive_folder(
        "https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7",
        api_key="key",
        progress_callback=lambda d, t: calls.append((d, t)),
    )

    # Progress accumulates across files against the batch total (300 bytes),
    # and the final call always reaches 100%.
    assert (0, 300) in calls
    assert (100, 300) in calls
    assert (300, 300) in calls
    assert calls[-1] == (300, 300)


def test_bulk_download_drive_folder_invalid_folder():
    with pytest.raises(ValueError, match="Invalid Google Drive Folder"):
        bulk_download_drive_folder("https://example.com/path")


@patch("src.utils.google_drive.download_file_bytes")
@patch("src.utils.google_drive.list_files_in_folder")
@patch("src.utils.google_drive.get_drive_service")
def test_bulk_download_drive_folder_handles_list_error(
    mock_get_service, mock_list, mock_download
):
    mock_list.side_effect = Exception("403 Forbidden")

    with pytest.raises(Exception, match="403 Forbidden"):
        bulk_download_drive_folder(
            "https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7",
            api_key="key",
        )


@patch("src.utils.google_drive.requests.head")
def test_check_folder_access_accessible(mock_head):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_head.return_value = mock_response

    result = check_folder_access("valid-folder-id")

    assert result is True
    mock_head.assert_called_once_with(
        "https://drive.google.com/drive/folders/valid-folder-id",
        allow_redirects=False,
        timeout=10,
    )


@patch("src.utils.google_drive.requests.head")
def test_check_folder_access_not_accessible(mock_head):
    mock_response = Mock()
    mock_response.status_code = 302
    mock_head.return_value = mock_response

    result = check_folder_access("private-folder-id")
    assert result is False

    mock_response.status_code = 404
    result = check_folder_access("nonexistent-folder-id")
    assert result is False


@patch("src.utils.google_drive.requests.head")
def test_check_folder_access_request_exception(mock_head):
    from requests import RequestException
    mock_head.side_effect = RequestException("Timeout")

    result = check_folder_access("error-folder-id")
    assert result is False


def test_check_folder_access_invalid_folder_id():
    assert check_folder_access("") is False
    assert check_folder_access("folder/with/slashes") is False


def test_validate_service_account_key_valid():
    key_dict = {
        "type": "service_account",
        "project_id": "my-project",
        "private_key": "some-private-key",
        "client_email": "service-account@my-project.iam.gserviceaccount.com"
    }
    assert validate_service_account_key(key_dict) is True


def test_validate_service_account_key_missing_fields(caplog):
    key_dict = {
        "type": "service_account",
        "project_id": "my-project",
        "private_key": "some-private-key",
    }
    with caplog.at_level("WARNING"):
        assert validate_service_account_key(key_dict) is False
        assert "Google Drive service account key is missing or empty" in caplog.text


def test_validate_service_account_key_empty_fields(caplog):
    key_dict = {
        "type": "service_account",
        "project_id": "",
        "private_key": "some-private-key",
        "client_email": "service-account@my-project.iam.gserviceaccount.com"
    }
    with caplog.at_level("WARNING"):
        assert validate_service_account_key(key_dict) is False
        assert "Google Drive service account key is missing or empty" in caplog.text


def test_validate_service_account_key_invalid_type(caplog):
    with caplog.at_level("WARNING"):
        assert validate_service_account_key(None) is False
        assert "Invalid key type: expected a dictionary" in caplog.text
        assert validate_service_account_key("string-key") is False

