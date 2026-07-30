"""
errors.py
---------
Centralized constant definitions for all user-facing error and warning messages.
"""

# Authentication Errors
AUTH_USERNAME_EMPTY = "Username cannot be empty."
AUTH_PASSWORD_TOO_SHORT = "Password must be at least 6 characters long."
AUTH_INVALID_ROLE = "Role must be one of: {roles}"
AUTH_USER_EXISTS = "User '{username}' already exists."
AUTH_USER_NOT_FOUND = "User not found."
AUTH_INVALID_CREDENTIALS = "Invalid username or password."
AUTH_BLANK_CREDENTIALS = "Please enter both username and password."
AUTH_ROLE_UNDETERMINED = "Unable to determine the user role."
AUTH_INVALID_2FA_CODE = "Invalid verification code. Please try again."
AUTH_CONFIG_2FA_ERROR = "2FA configuration error. Please contact admin."
AUTH_INVALID_2FA_DISABLE = "Invalid verification code. 2FA remains enabled."

# ZIP File Processing Errors
ZIP_EMPTY = "ZIP archive is empty."
ZIP_SINGLE_FILE_LIMIT = (
    "Entry '{filename}' exceeds single file decompression safety limit of {limit_mb}MB."
)
ZIP_TOTAL_SIZE_LIMIT = (
    "ZIP archive total decompressed size exceeds safety limit of {limit_mb}MB."
)
ZIP_ENCRYPTED = "Password-protected or encrypted ZIP files are not supported."
ZIP_ENTRY_CORRUPTED = "Corrupted or protected entry: {filename}"
ZIP_INVALID = "Invalid or corrupted ZIP archive."
ZIP_NO_SUPPORTED_DOCS = (
    "⚠️ ZIP file '{filename}' contains no supported documents (.pdf, .docx, .txt)."
)
ZIP_FAILED_TO_PROCESS = "⚠️ Failed to process ZIP archive '{filename}': {error}"

# Google Drive Errors
DRIVE_NO_CREDENTIALS = "No API Key or Service Account credentials provided."
DRIVE_INVALID_URL_OR_ID = "Invalid Google Drive Folder URL or ID."
DRIVE_IMPORT_FAILED = "Failed to import from Google Drive: {error}"
DRIVE_ENTER_VALID_LINK = "Please enter a valid Google Drive folder link or ID."

# OCR & Document Parser Errors
OCR_DPI_INVALID = "OCR DPI must be an integer between 150 and 400."
OCR_DPI_OUT_OF_RANGE = "OCR DPI must be between {min_dpi} and {max_dpi}."
OCR_LANGUAGE_UNSUPPORTED = (
    "Unsupported OCR language '{language}'. Supported values: {supported}."
)
OCR_DEPENDENCIES_MISSING = "OCR dependencies are missing. Install pytesseract, PyMuPDF and Pillow using: python -m pip install pytesseract pymupdf pillow"
OCR_TESSERACT_NOT_FOUND = "Tesseract OCR was not found. Install Tesseract and either add it to PATH or set TESSERACT_CMD to tesseract.exe."
BADGE_PIL_REQUIRED = "PIL/Pillow is required for PNG badge generation"

# Similarity & FAISS Errors
SIM_BATCH_SIZE_INVALID = "batch_size must be an integer"


def sim_weight_out_of_range(w):
    return f"Weight w must be between 0.0 and 1.0, got {w}"


SIM_SHAPE_MISMATCH = "Semantic and lexical matrices must have the same shape"
SIM_INDEX_MISMATCH = (
    "Semantic and lexical matrices must have the same index and columns"
)
FAISS_STORED_EMB_DIM_INVALID = "Stored embeddings must be two-dimensional."


def faiss_emb_registry_mismatch(emb_count, reg_count):
    return f"Corpus embedding count does not match chunk registry count: {emb_count} != {reg_count}"


# Incident Database Errors
INCIDENT_DB_INIT_FAILED = "Failed to initialize incident database: {error}"
INCIDENT_SYNC_FAILED = "Failed to synchronize incidents: {error}"
INCIDENT_INVALID_REVIEW_STATUS = "review_status must be one of {valid_statuses}"
INCIDENT_UPDATE_STATUS_FAILED = "Failed to update review status: {error}"

# SSRF / Webhook Security Errors
SSRF_WEBHOOK_URL_EMPTY = "Webhook URL cannot be empty."
SSRF_INSECURE_SCHEME = "Insecure scheme '{scheme}'. Webhooks must use 'https'."
SSRF_MISSING_HOSTNAME = "Invalid URL: missing hostname."
SSRF_INVALID_IP_FORMAT = "Resolved invalid IP address format: {error}"
SSRF_BLOCKED_PRIVATE_SUBNET = "Blocked private IPv4 subnet IP: {ip} ({subnet})"
SSRF_BLOCKED_LOOPBACK = "Blocked loopback IP: {ip}"
SSRF_BLOCKED_PRIVATE_NETWORK = "Blocked private network IP: {ip}"
SSRF_BLOCKED_LINK_LOCAL = "Blocked link-local IP: {ip}"
SSRF_BLOCKED_MULTICAST = "Blocked multicast IP: {ip}"
SSRF_BLOCKED_UNSPECIFIED = "Blocked unspecified IP: {ip}"
SSRF_DNS_NO_ADDRESSES = "No addresses found for hostname '{hostname}'"
SSRF_DNS_RESOLUTION_FAILED = "DNS resolution failed for hostname '{hostname}': {error}"

# API ErrorsAPI_UNAUTHORIZED = "Invalid or missing authentication token."
API_FILENAME_MISSING = "Filename must be provided."
API_FILE_EMPTY = "Uploaded file is empty."
API_TEXT_EXTRACTION_FAILED = "Failed to extract readable text from the uploaded file."
API_FORBIDDEN_CLEAR = (
    "Forbidden: Only administrators are authorized to clear all documents."
)
API_CLEAR_CORPUS_FAILED = "An error occurred while clearing the corpus: {error}"

# UI/Dashboard Errors
UI_SESSION_EXPIRED = (
    "⏱️ Your session has expired due to 15 minutes of inactivity. Please log in again."
)


def ui_index_load_failed(error):
    return f"Error loading index: {error}"


UI_PDF_PREVIEW_FAILED = "Unable to render PDF preview: {error}"
UI_PDF_PREVIEW_RESTRICTED = "PDF Preview is only available for uploaded `.pdf` files."
UI_UPLOAD_MIN_FILES = "Upload at least 2 files to begin analysis."
UI_UPLOAD_MIN_DOCS = "Please upload or import from Drive at least 2 PDF, DOCX, or TXT assignments to begin."
UI_UPLOAD_MIN_DOCS_ANALYSIS = (
    "Please upload at least 2 PDF, DOCX, or TXT assignments to begin analysis."
)
UI_COMPUTE_SIMILARITY_MIN_DOCS = (
    "Ensure at least 2 documents are uploaded to compute similarities."
)
UI_NO_DOCUMENTS_INDEXED = (
    "No documents are currently indexed. Please contact your administrator."
)
UI_REUPLOAD_REQUIRED_MATRIX = "⚠️ Full similarity matrix requires re-uploading files. FAISS search is available with existing index."
UI_NO_NEW_FILES = (
    "No new files to upload. All uploaded files are already in the database."
)
UI_SIMILARITY_MATRIX_REUPLOAD = "⚠️ Similarity matrix requires re-uploading files. FAISS search is available with existing index."
UI_COULD_NOT_EXTRACT_TEXT = "⚠️ **Could not extract text from:** {docs}. These might be scanned images or password-protected PDFs."
UI_NEED_MIN_DOCUMENTS = "Need at least 2 documents."
UI_PDF_REPORT_GEN_FAILED = "Error generating PDF report: {error}"

# CLI Errors
CLI_FOLDER_NOT_FOUND = "Error: Folder '{folder_path}' does not exist.\n"
CLI_PATH_NOT_DIR = "Error: Path '{folder_path}' is not a directory.\n"
CLI_READ_FOLDER_FAILED = "Error reading folder contents: {error}\n"
CLI_EXTRACTED_TEXT_EMPTY = "Warning: Extracted text from '{filename}' is empty.\n"
CLI_PARSE_FILE_FAILED = "Warning: Failed to parse '{filename}': {error}\n"
CLI_PIPELINE_FAILED = "Error during plagiarism detection pipeline: {error}\n"
CLI_THRESHOLD_INVALID = "Error: Threshold must be a float between 0.0 and 1.0.\n"
CLI_INVALID_COMMAND = "Error: Invalid command '{command}'.\n"


# SSRF Protection Errors
SSRF_EMPTY_URL = "Webhook URL cannot be empty."
SSRF_INSECURE_SCHEME = "Insecure scheme '{scheme}'. Webhooks must use 'https'."
SSRF_MISSING_HOSTNAME = "Invalid URL: missing hostname."
SSRF_NO_ADDRESSES = "No addresses found for hostname '{hostname}'"
SSRF_DNS_RESOLUTION_FAILED = "DNS resolution failed for hostname '{hostname}': {error}"
SSRF_INVALID_IP = "Resolved invalid IP address format: {error}"
SSRF_BLOCKED_LOOPBACK = "Blocked loopback IP: {ip}"
SSRF_BLOCKED_PRIVATE = "Blocked private network IP: {ip}"
SSRF_BLOCKED_LINK_LOCAL = "Blocked link-local IP: {ip}"
SSRF_BLOCKED_MULTICAST = "Blocked multicast IP: {ip}"
SSRF_BLOCKED_UNSPECIFIED = "Blocked unspecified IP: {ip}"


class ExportFailedError(RuntimeError):
    """Raised when an export cannot be generated or written safely."""


EXPORT_WRITE_FAILED = (
    "Unable to write the {format_name} export to '{destination}'. "
    "Check the destination permissions and available disk space, then try again."
)

EXPORT_GENERATION_IO_FAILED = (
    "Unable to generate the {format_name} export because an I/O operation failed. "
    "Please try again."
)
