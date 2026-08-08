"""
src/utils/google_drive.py
-------------------------
Utilities for authenticating with Google Drive API, listing folder contents,
and bulk downloading supported assignment files (.pdf, .docx, .txt).
"""

import io
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import requests

import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.utils.filename import unique_filename

logger = logging.getLogger(__name__)

# Supported extensions for the plagiarism detection pipeline
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".doc", ".txt")


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
            logger.warning(f"Google Drive service account key is missing or empty for required field: {key}")
            return False

    return True


def get_supported_file_extensions() -> List[str]:
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


def list_files_in_folder(service, folder_id: str) -> List[Dict[str, str]]:
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
) -> Tuple[Dict[str, bytes], List[str]]:
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
    batch_total_bytes = sum(
        int(f["size"]) for f in files_to_download if f.get("size")
    )

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
                progress_callback(
                    _base + file_bytes_downloaded, batch_total_bytes
                )

        file_bytes = download_file_bytes(
            service, file_record["id"], progress_callback=file_progress_cb
        )
        safe_name = unique_filename(
            file_record["name"],
            downloaded_files_dict,
        )
        downloaded_files_dict[safe_name] = file_bytes
        downloaded_names.append(safe_name)
        bytes_done_before_current_file += int(file_record.get("size") or len(file_bytes))
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
