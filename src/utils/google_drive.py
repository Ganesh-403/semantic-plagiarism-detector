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

"""
src/utils/google_drive.py
-------------------------
Utilities for authenticating with Google Drive API, listing folder contents,
and bulk downloading supported assignment files (.pdf, .docx, .txt).
"""

import io
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from src.utils.filename import unique_filename

logger = logging.getLogger(__name__)

# Supported extensions for the plagiarism detection pipeline
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".doc", ".txt")

#: Chunk size (256 KB) used for resumable uploads so large files (>10 MB)
#: survive flaky/slow connections instead of failing in a single request (#3462).
RESUMABLE_UPLOAD_CHUNK_SIZE = 256 * 1024

#: Files larger than this size (10 MB) should use the resumable upload path.
LARGE_FILE_THRESHOLD_BYTES = 10 * 1024 * 1024


def validate_service_account_key(key_dict: dict) -> bool:
    """
    Validate that the service account JSON key dictionary contains required fields.
    """
    if not isinstance(key_dict, dict):
        logger.warning("Invalid key type: expected a dictionary.")
        return False

    required_keys = ["type", "project_id", "private_key", "client_email"]
    for key in required_keys:
        if key not in key_dict or not key_dict[key]:
            logger.warning(
                f"Google Drive service account key is missing or empty for required field: {key}"
            )
            return False

    return True


def get_supported_file_extensions() -> list[str]:
    """
    Return the list of supported file extensions for the plagiarism detection pipeline.

    Returns:
        A sorted list of supported file extensions (e.g., [".docx", ".pdf", ".txt"]).
    """
    return sorted(SUPPORTED_EXTENSIONS)


_DRIVE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{33}$")
_DRIVE_URL_RE = re.compile(r"folders/([a-zA-Z0-9_-]{33})(?:[/?]|$)")


def extract_google_drive_folder_id(url_or_id: str) -> str | None:
    """
    Extracts the Google Drive Folder ID from a full Drive URL or raw ID string.
    Validates a 33-character ID consisting of letters, numbers, '_' and '-'.
    """
    if not isinstance(url_or_id, str):
        return None

    cleaned = url_or_id.strip()
    if not cleaned:
        return None

    if _DRIVE_ID_RE.match(cleaned):
        return cleaned

    match = _DRIVE_URL_RE.search(cleaned)
    if match:
        return match.group(1)

    return None


_FOLDER_ID_PATTERN = re.compile(r"[\w-]{25,}")


def extract_folder_id(url_or_id: str) -> str | None:
    """
    Extracts a Google Drive folder ID from a full Drive URL or raw ID string.

    Uses a permissive regex to match any run of word characters and hyphens
    that is at least 25 characters long, so it accepts both full folder
    URLs (e.g. "https://drive.google.com/drive/folders/1A2B3C...") and a
    bare folder ID pasted on its own.

    Args:
        url_or_id: A Google Drive folder URL or a raw folder ID string.

    Returns:
        The extracted folder ID string, or None if no valid ID is found.
    """
    if not isinstance(url_or_id, str):
        return None

    match = _FOLDER_ID_PATTERN.search(url_or_id)
    if match:
        return match.group(0)

    return None


def get_drive_service(
    api_key: Optional[str] = None, service_account_info: Optional[dict] = None
):
    """
    Builds and returns a Google Drive API service instance using an API key or Service Account.
    """
    if service_account_info:
        creds = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        return build("drive", "v3", credentials=creds)
    elif api_key:
        return build("drive", "v3", developerKey=api_key)
    else:
        # Fallback to environment variable key if present
        env_key = os.getenv("GOOGLE_DRIVE_API_KEY")
        if env_key:
            return build("drive", "v3", developerKey=env_key)
        raise ValueError("No API Key or Service Account credentials provided.")


def list_files_in_folder(service, folder_id: str) -> list[dict[str, str]]:
    """
    Lists all supported assignment files (.pdf, .docx, .txt) within a specified Google Drive folder.
    """
    query = f"'{folder_id}' in parents and trashed = false"
    results = (
        service.files()
        .list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, size)",
        )
        .execute()
    )

    files = results.get("files", [])
    supported_files = [
        f for f in files if f["name"].lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    return supported_files


def download_file_bytes(
    service,
    file_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """
    Downloads a binary file from Google Drive into a BytesIO stream and returns bytes.

    Args:
        service: Authenticated Google Drive API service instance.
        file_id: The Drive file ID to download.
        progress_callback: Optional callable invoked as ``callback(bytes_downloaded,
            total_bytes)`` after every chunk is streamed, so callers (e.g. a Streamlit
            UI) can render live download progress. ``total_bytes`` may be ``0`` if the
            API does not report a size for the file being downloaded.
    """
    request = service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if progress_callback is not None and status is not None:
            bytes_downloaded = getattr(status, "resumable_progress", 0) or 0
            total_bytes = getattr(status, "total_size", 0) or 0
            progress_callback(bytes_downloaded, total_bytes)
    if progress_callback is not None:
        # Guarantee a final 100%-complete callback even if the API never sent a
        # status object with total_size populated (small/empty files).
        final_size = len(file_stream.getvalue())
        progress_callback(final_size, final_size)
    return file_stream.getvalue()


def bulk_download_drive_folder(
    folder_url_or_id: str,
    api_key: Optional[str] = None,
    service_account_info: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[dict[str, bytes], list[str]]:
    """
    Main helper: Extracts folder ID, lists supported files, downloads them into memory,
    and returns a dictionary mapping filename -> raw bytes.

    Args:
        folder_url_or_id: A Drive folder URL or a raw folder ID.
        api_key: Optional Drive API key for authentication.
        service_account_info: Optional service-account credentials dict.
        progress_callback: Optional callable invoked as ``callback(bytes_downloaded,
            total_bytes)`` after every chunk of every file is streamed. Progress is
            aggregated across the whole batch, so ``bytes_downloaded`` accumulates
            across files and ``total_bytes`` is the sum of all file sizes in the
            folder (as reported by the Drive API). Pass this straight to a Streamlit
            ``st.progress`` bar to show live batch-download progress.

    Returns:
        Tuple[Dict[str, bytes], List[str]]: (file_bytes_dict, list_of_downloaded_filenames)
    """
    folder_id = extract_google_drive_folder_id(folder_url_or_id)
    if not folder_id:
        raise ValueError("Invalid Google Drive Folder URL or ID.")

    service = get_drive_service(
        api_key=api_key, service_account_info=service_account_info
    )
    files_to_download = list_files_in_folder(service, folder_id)

    # Pre-compute total batch size (in bytes) so the callback can report overall
    # progress rather than just per-file progress. Files without a reported size
    # (rare, but the Drive API allows it) simply don't contribute to the total.
    batch_total_bytes = sum(int(f["size"]) for f in files_to_download if f.get("size"))

    downloaded_files_dict = {}
    downloaded_names = []
    bytes_done_before_current_file = 0

    if progress_callback is not None:
        progress_callback(0, batch_total_bytes)

    for file_record in files_to_download:
        file_progress_cb = None
        if progress_callback is not None:

            def file_progress_cb(
                file_bytes_downloaded,
                _file_total_bytes,
                _base=bytes_done_before_current_file,
            ):
                progress_callback(_base + file_bytes_downloaded, batch_total_bytes)

        file_bytes = download_file_bytes(
            service, file_record["id"], progress_callback=file_progress_cb
        )
        safe_name = unique_filename(
            file_record["name"],
            downloaded_files_dict,
        )
        downloaded_files_dict[safe_name] = file_bytes
        downloaded_names.append(safe_name)
        bytes_done_before_current_file += int(
            file_record.get("size") or len(file_bytes)
        )
        if progress_callback is not None:
            progress_callback(bytes_done_before_current_file, batch_total_bytes)

    if progress_callback is not None:
        # Final callback guarantees the bar always reaches 100% even if reported
        # file sizes didn't perfectly match the bytes actually streamed.
        progress_callback(batch_total_bytes, batch_total_bytes)

    return downloaded_files_dict, downloaded_names


def check_folder_access(folder_id: str) -> bool:
    """Checks if a Google Drive folder is publicly accessible via a lightweight HTTP HEAD request.

    Args:
        folder_id: The Google Drive folder ID to verify.

    Returns:
        bool: True if the folder is publicly accessible (returns 200 OK), False otherwise.
    """
    if not folder_id or not re.match(r"^[a-zA-Z0-9_-]+$", folder_id):
        return False
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        response = requests.head(url, allow_redirects=False, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def should_use_resumable_upload(file_size: Optional[int]) -> bool:
    """
    Return True when a file is large enough to require the resumable upload path.

    Args:
        file_size: File size in bytes (``None`` or ``0`` when unknown).

    Returns:
        True for files larger than :data:`LARGE_FILE_THRESHOLD_BYTES` (10 MB).
    """
    return bool(file_size and file_size > LARGE_FILE_THRESHOLD_BYTES)


def build_resumable_media(
    file_path: str,
    mime_type: str = "application/octet-stream",
) -> MediaFileUpload:
    """
    Build a resumable ``MediaFileUpload`` handle with 256 KB chunks.

    Args:
        file_path: Path of the local file to upload.
        mime_type: MIME type reported to the Drive API.

    Returns:
        A ``googleapiclient.http.MediaFileUpload`` configured with
        ``resumable=True`` and ``chunksize=256*1024`` (#3462).
    """
    return MediaFileUpload(
        file_path,
        mimetype=mime_type,
        resumable=True,
        chunksize=RESUMABLE_UPLOAD_CHUNK_SIZE,
    )


def upload_file_resumable(
    service: Any,
    file_path: str,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None,
    mime_type: str = "application/octet-stream",
    num_retries: int = 3,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Upload a local file to Google Drive using a resumable, chunked transfer.

    Files larger than ~10 MB frequently fail on slow connections when sent as
    a single request. This uploads the file in 256 KB chunks via
    ``MediaFileUpload(resumable=True)`` so interrupted transfers resume from
    the last acknowledged chunk instead of starting over.

    Args:
        service: Authenticated Google Drive API service instance.
        file_path: Path of the local file to upload.
        folder_id: Optional parent folder ID; omit to upload to "My Drive".
        file_name: Optional Drive file name; defaults to the basename.
        mime_type: MIME type reported to the Drive API.
        num_retries: Retries per individual chunk on transient HTTP errors.
        progress_callback: Optional callable invoked as ``callback(bytes_uploaded,
            total_bytes)`` after every chunk, mirroring the download callback
            convention used in this module. Always receives a final call at
            100% so Streamlit progress bars complete deterministically.

    Returns:
        The created Drive file resource dict (at minimum ``id`` and ``name``).

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If ``file_path`` points to an empty path.
        googleapiclient.errors.HttpError: If the Drive API rejects a chunk.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path must be a non-empty string.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata: Dict[str, Any] = {"name": file_name or os.path.basename(file_path)}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = build_resumable_media(file_path, mime_type=mime_type)
    request = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, name, mimeType, size",
    )

    total_bytes = media.size() or 0
    logger.info(
        "Starting resumable upload of %s (%d bytes, %d-byte chunks)",
        file_path,
        total_bytes,
        RESUMABLE_UPLOAD_CHUNK_SIZE,
    )

    response: Optional[Dict[str, Any]] = None
    bytes_uploaded = 0
    while response is None:
        status, response = request.next_chunk(num_retries=num_retries)
        if status is not None:
            bytes_uploaded = getattr(status, "resumable_progress", 0) or 0
            total_bytes = getattr(status, "total_size", 0) or total_bytes
        if progress_callback is not None:
            progress_callback(bytes_uploaded, total_bytes)

    # Guarantee a final 100%-complete callback even if the API never sent a
    # status object carrying the full size (e.g. tiny files uploaded in one chunk).
    if progress_callback is not None:
        progress_callback(total_bytes, total_bytes)

    logger.info(
        "Resumable upload finished for '%s' (id=%s)",
        metadata["name"],
        response.get("id"),
    )
    return response


def bulk_upload_files_resumable(
    service: Any,
    file_paths: List[str],
    folder_id: Optional[str] = None,
    mime_type: str = "application/octet-stream",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Upload multiple files resumably, aggregating progress across the batch.

    Args:
        service: Authenticated Google Drive API service instance.
        file_paths: Local paths to upload, in order.
        folder_id: Optional destination folder ID shared by all files.
        mime_type: MIME type applied to every file.
        progress_callback: Optional callable invoked as ``callback(bytes_uploaded,
            batch_total_bytes)`` where totals accumulate across all files.

    Returns:
        The list of created Drive file resource dicts, one per input path.
    """
    sizes = []
    for path in file_paths:
        try:
            sizes.append(os.path.getsize(path))
        except OSError:
            sizes.append(0)
    batch_total_bytes = sum(sizes)

    results: List[Dict[str, Any]] = []
    bytes_done_before_current_file = 0

    def _batch_progress(bytes_uploaded: int, _total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                bytes_done_before_current_file + bytes_uploaded, batch_total_bytes
            )

    for index, path in enumerate(file_paths):
        response = upload_file_resumable(
            service,
            path,
            folder_id=folder_id,
            mime_type=mime_type,
            progress_callback=_batch_progress,
        )
        results.append(response)
        bytes_done_before_current_file += sizes[index]
        if progress_callback is not None:
            progress_callback(bytes_done_before_current_file, batch_total_bytes)

    if progress_callback is not None:
        # Final callback guarantees the bar reaches 100% even if reported
        # sizes didn't perfectly match the bytes actually streamed.
        progress_callback(batch_total_bytes, batch_total_bytes)

    return results
