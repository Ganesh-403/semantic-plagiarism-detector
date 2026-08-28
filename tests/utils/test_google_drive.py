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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.utils.google_drive import (
    LARGE_FILE_THRESHOLD_BYTES,
    RESUMABLE_UPLOAD_CHUNK_SIZE,
    build_resumable_media,
    bulk_download_drive_folder,
    bulk_upload_files_resumable,
    check_folder_access,
    download_file_bytes,
    extract_folder_id,
    extract_google_drive_folder_id,
    get_drive_service,
    list_files_in_folder,
    should_use_resumable_upload,
    upload_file_resumable,
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


def test_extract_folder_id_valid_id():
    valid_id = "1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7"
    assert extract_folder_id(valid_id) == valid_id


def test_extract_folder_id_valid_url():
    valid_id = "1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7"
    url = f"https://drive.google.com/drive/folders/{valid_id}"
    assert extract_folder_id(url) == valid_id


def test_extract_folder_id_too_short_returns_none():
    assert extract_folder_id("short_id_123") is None


def test_extract_folder_id_non_string_returns_none():
    assert extract_folder_id(None) is None


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


# â”€â”€ get_drive_service tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@patch("src.utils.google_drive.build")
def test_get_drive_service_with_api_key(mock_build):
    service = get_drive_service(api_key="test-api-key")
    mock_build.assert_called_once_with("drive", "v3", developerKey="test-api-key")
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


# â”€â”€ list_files_in_folder tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _mock_service_for_list(files):
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {"files": files}
    return service


def test_list_files_in_folder_returns_supported():
    files = [
        {"id": "1", "name": "report.pdf", "mimeType": "application/pdf"},
        {
            "id": "2",
            "name": "essay.docx",
            "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
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
    service.files.return_value.list.return_value.execute.side_effect = Exception(
        "403 Forbidden"
    )

    with pytest.raises(Exception, match="403 Forbidden"):
        list_files_in_folder(service, "folder123")


def test_list_files_in_folder_handles_not_found():
    service = Mock()
    service.files.return_value.list.return_value.execute.side_effect = Exception(
        "404 Not Found"
    )

    with pytest.raises(Exception, match="404 Not Found"):
        list_files_in_folder(service, "folder123")


# â”€â”€ download_file_bytes tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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
    download_file_bytes(
        service, "file123", progress_callback=lambda d, t: calls.append((d, t))
    )

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


# â”€â”€ bulk_download_drive_folder tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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
        "client_email": "service-account@my-project.iam.gserviceaccount.com",
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
        "client_email": "service-account@my-project.iam.gserviceaccount.com",
    }
    with caplog.at_level("WARNING"):
        assert validate_service_account_key(key_dict) is False
        assert "Google Drive service account key is missing or empty" in caplog.text


def test_validate_service_account_key_invalid_type(caplog):
    with caplog.at_level("WARNING"):
        assert validate_service_account_key(None) is False
        assert "Invalid key type: expected a dictionary" in caplog.text
        assert validate_service_account_key("string-key") is False


# ── Resumable / chunked upload support (#3462) ───────────────────────────────


def _make_upload_service(chunks, final_response, total_size=1024):
    """Build a mock Drive service whose files().create() yields the given chunks."""
    statuses = [
        SimpleNamespace(resumable_progress=b, total_size=total_size) for b in chunks
    ]
    sequence = [(s, None) for s in statuses] + [(None, final_response)]
    request = MagicMock(name="HttpRequest")
    request.next_chunk.side_effect = sequence
    service = MagicMock(name="DriveService")
    service.files.return_value.create.return_value = request
    return service, request


@patch("src.utils.google_drive.MediaFileUpload")
def test_build_resumable_media_uses_256kb_chunks(mock_media, tmp_path):
    build_resumable_media(str(tmp_path / "f.pdf"), mime_type="application/pdf")
    mock_media.assert_called_once_with(
        str(tmp_path / "f.pdf"),
        mimetype="application/pdf",
        resumable=True,
        chunksize=RESUMABLE_UPLOAD_CHUNK_SIZE,
    )


@patch("src.utils.google_drive.MediaFileUpload")
def test_upload_file_resumable_creates_file_with_metadata(mock_media, tmp_path):
    mock_media.return_value.size.return_value = 512
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"x" * 512)
    service, request = _make_upload_service(
        [0, 256], {"id": "abc123", "name": "report.pdf"}
    )

    response = upload_file_resumable(service, str(file_path), folder_id="folder1")

    assert response == {"id": "abc123", "name": "report.pdf"}
    _, kwargs = service.files.return_value.create.call_args
    assert kwargs["body"] == {"name": "report.pdf", "parents": ["folder1"]}
    assert "id" in kwargs["fields"]
    request.next_chunk.assert_called_with(num_retries=3)


@patch("src.utils.google_drive.MediaFileUpload")
def test_upload_file_resumable_defaults_name_to_basename(mock_media, tmp_path):
    mock_media.return_value.size.return_value = 4
    file_path = tmp_path / "essay.docx"
    file_path.write_bytes(b"data")
    service, _ = _make_upload_service([0], {"id": "id1", "name": "essay.docx"})

    upload_file_resumable(service, str(file_path))

    body = service.files.return_value.create.call_args.kwargs["body"]
    assert body["name"] == "essay.docx"
    assert "parents" not in body


@patch("src.utils.google_drive.MediaFileUpload")
def test_upload_file_resumable_reports_progress_and_final_callback(
    mock_media, tmp_path
):
    mock_media.return_value.size.return_value = 1024
    file_path = tmp_path / "big.bin"
    file_path.write_bytes(b"z" * 10)
    service, _ = _make_upload_service([0, 4, 8], {"id": "big1"})
    calls = []

    def record(done, total):
        calls.append((done, total))

    upload_file_resumable(service, str(file_path), progress_callback=record)

    assert (0, 1024) in calls
    assert (8, 1024) in calls
    # Final guaranteed 100% callback.
    assert calls[-1] == (1024, 1024)


@patch("src.utils.google_drive.MediaFileUpload")
def test_upload_file_resumable_passes_num_retries(mock_media, tmp_path):
    mock_media.return_value.size.return_value = 2
    file_path = tmp_path / "a.txt"
    file_path.write_text("hi")
    service, request = _make_upload_service([], {"id": "n1"})

    upload_file_resumable(service, str(file_path), num_retries=5)

    request.next_chunk.assert_called_with(num_retries=5)


@patch("src.utils.google_drive.MediaFileUpload")
def test_upload_file_resumable_missing_file_raises(mock_media):
    service, _ = _make_upload_service([], {})
    with pytest.raises(FileNotFoundError):
        upload_file_resumable(service, "does/not/exist.pdf")


@pytest.mark.parametrize("bad_path", ["", "   ", None])
@patch("src.utils.google_drive.MediaFileUpload")
def test_upload_file_resumable_invalid_path_raises(mock_media, bad_path):
    service, _ = _make_upload_service([], {})
    with pytest.raises(ValueError):
        upload_file_resumable(service, bad_path)


def test_should_use_resumable_upload_thresholds():
    assert should_use_resumable_upload(LARGE_FILE_THRESHOLD_BYTES + 1) is True
    assert should_use_resumable_upload(LARGE_FILE_THRESHOLD_BYTES) is False
    assert should_use_resumable_upload(1024) is False
    assert should_use_resumable_upload(None) is False
    assert should_use_resumable_upload(0) is False


@patch("src.utils.google_drive.MediaFileUpload")
def test_bulk_upload_files_resumable_aggregates_progress(mock_media, tmp_path):
    paths = []
    for name in ("one.pdf", "two.pdf"):
        p = tmp_path / name
        p.write_bytes(b"d")
        paths.append(str(p))
    created = []

    def create_side_effect(**kwargs):
        created.append(kwargs)
        return _make_upload_service([0], {"id": "id%d" % len(created)})[1]

    service = MagicMock(name="DriveService")
    service.files.return_value.create.side_effect = create_side_effect
    calls = []

    results = bulk_upload_files_resumable(
        service, paths, progress_callback=lambda d, t: calls.append((d, t))
    )

    assert len(results) == 2
    assert len(created) == 2
    assert calls[-1] == (2, 2)  # Both 1-byte files fully uploaded.
