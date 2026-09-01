"""
errors.py
---------
Centralized constant definitions for all user-facing error and warning messages.
"""

__all__ = [
    "AUTH_USERNAME_EMPTY",
    "AUTH_PASSWORD_TOO_SHORT",
    "AUTH_INVALID_ROLE",
    "AUTH_USER_EXISTS",
    "AUTH_USER_NOT_FOUND",
    "AUTH_INVALID_CREDENTIALS",
    "AUTH_BLANK_CREDENTIALS",
    "AUTH_ROLE_UNDETERMINED",
    "AUTH_INVALID_2FA_CODE",
    "AUTH_CONFIG_2FA_ERROR",
    "AUTH_INVALID_2FA_DISABLE",
    "ZIP_EMPTY",
    "ZIP_SINGLE_FILE_LIMIT",
    "ZIP_TOTAL_SIZE_LIMIT",
    "ZIP_ENCRYPTED",
    "ZIP_ENTRY_CORRUPTED",
    "ZIP_INVALID",
    "ZIP_NO_SUPPORTED_DOCS",
    "ZIP_FAILED_TO_PROCESS",
    "DRIVE_NO_CREDENTIALS",
    "DRIVE_INVALID_URL_OR_ID",
    "DRIVE_IMPORT_FAILED",
    "DRIVE_ENTER_VALID_LINK",
    "OCR_DPI_INVALID",
    "OCR_DPI_OUT_OF_RANGE",
    "OCR_LANGUAGE_UNSUPPORTED",
    "OCR_DEPENDENCIES_MISSING",
    "OCR_TESSERACT_NOT_FOUND",
    "BADGE_PIL_REQUIRED",
    "PARSER_BATCH_LIMIT_EXCEEDED",
    "SIM_BATCH_SIZE_INVALID",
    "SIM_SHAPE_MISMATCH",
    "SIM_INDEX_MISMATCH",
    "SIM_WEIGHT_OUT_OF_RANGE",
    "FAISS_STORED_EMB_DIM_INVALID",
    "FAISS_EMB_REGISTRY_MISMATCH",
    "INCIDENT_DB_INIT_FAILED",
    "INCIDENT_SYNC_FAILED",
    "INCIDENT_INVALID_REVIEW_STATUS",
    "INCIDENT_UPDATE_STATUS_FAILED",
    "SSRF_WEBHOOK_URL_EMPTY",
    "SSRF_INSECURE_SCHEME",
    "SSRF_MISSING_HOSTNAME",
    "SSRF_INVALID_HOSTNAME",
    "SSRF_INVALID_IP_FORMAT",
    "SSRF_BLOCKED_PRIVATE_SUBNET",
    "SSRF_BLOCKED_LOOPBACK",
    "SSRF_BLOCKED_PRIVATE_NETWORK",
    "SSRF_BLOCKED_PRIVATE",
    "SSRF_BLOCKED_LINK_LOCAL",
    "SSRF_BLOCKED_MULTICAST",
    "SSRF_BLOCKED_UNSPECIFIED",
    "SSRF_DNS_NO_ADDRESSES",
    "SSRF_DNS_RESOLUTION_FAILED",
    "SSRF_DOMAIN_NOT_ALLOWED",
    "SSRF_MAX_REDIRECTS_EXCEEDED",
    "SSRF_CIRCULAR_REDIRECT_LOOP",
    "API_UNAUTHORIZED",
    "API_FILENAME_MISSING",
    "API_FILE_EMPTY",
    "API_TEXT_EXTRACTION_FAILED",
    "API_FORBIDDEN_CLEAR",
    "API_CLEAR_CORPUS_FAILED",
    "UI_SESSION_EXPIRED",
    "UI_INDEX_LOAD_FAILED",
    "UI_PDF_PREVIEW_FAILED",
    "UI_PDF_PREVIEW_RESTRICTED",
    "UI_UPLOAD_MIN_FILES",
    "UI_UPLOAD_MIN_DOCS",
    "UI_UPLOAD_MIN_DOCS_ANALYSIS",
    "UI_COMPUTE_SIMILARITY_MIN_DOCS",
    "UI_NO_DOCUMENTS_INDEXED",
    "UI_REUPLOAD_REQUIRED_MATRIX",
    "UI_NO_NEW_FILES",
    "UI_SIMILARITY_MATRIX_REUPLOAD",
    "UI_COULD_NOT_EXTRACT_TEXT",
    "UI_NEED_MIN_DOCUMENTS",
    "UI_PDF_REPORT_GEN_FAILED",
    "CLI_FOLDER_NOT_FOUND",
    "CLI_PATH_NOT_DIR",
    "CLI_READ_FOLDER_FAILED",
    "CLI_EXTRACTED_TEXT_EMPTY",
    "CLI_PARSE_FILE_FAILED",
    "CLI_PIPELINE_FAILED",
    "CLI_THRESHOLD_INVALID",
    "CLI_INVALID_COMMAND",
    "EXPORT_GENERATION_IO_FAILED",
    "PDFEncryptedError",
]

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
PARSER_BATCH_LIMIT_EXCEEDED = "Batch size exceeds maximum limit of {limit} files."

# Similarity & FAISS Errors
SIM_BATCH_SIZE_INVALID = "batch_size must be an integer"


SIM_WEIGHT_OUT_OF_RANGE = "Weight w must be between 0.0 and 1.0, got {w}"


SIM_SHAPE_MISMATCH = "Semantic and lexical matrices must have the same shape"
SIM_INDEX_MISMATCH = (
    "Semantic and lexical matrices must have the same index and columns"
)
FAISS_STORED_EMB_DIM_INVALID = "Stored embeddings must be two-dimensional."


FAISS_EMB_REGISTRY_MISMATCH = "Corpus embedding count does not match chunk registry count: {emb_count} != {reg_count}"


# Incident Database Errors
INCIDENT_DB_INIT_FAILED = "Failed to initialize incident database: {error}"
INCIDENT_SYNC_FAILED = "Failed to synchronize incidents: {error}"
INCIDENT_INVALID_REVIEW_STATUS = "review_status must be one of {valid_statuses}"
INCIDENT_UPDATE_STATUS_FAILED = "Failed to update review status: {error}"

# SSRF / Webhook Security Errors
SSRF_WEBHOOK_URL_EMPTY = "Webhook URL cannot be empty."
SSRF_INSECURE_SCHEME = "Insecure scheme '{scheme}'. Webhooks must use 'https'."
SSRF_MISSING_HOSTNAME = "Invalid URL: missing hostname."
SSRF_INVALID_HOSTNAME = (
    "Invalid URL: hostname cannot be encoded as an internationalised domain name."
)
SSRF_INVALID_IP_FORMAT = "Resolved invalid IP address format: {error}"
SSRF_BLOCKED_PRIVATE_SUBNET = "Blocked private IPv4 subnet IP: {ip} ({subnet})"
SSRF_BLOCKED_LOOPBACK = "Blocked loopback IP: {ip}"
SSRF_BLOCKED_PRIVATE_NETWORK = "Blocked private network IP: {ip}"
SSRF_BLOCKED_PRIVATE = "Blocked private network IP: {ip}"
SSRF_BLOCKED_LINK_LOCAL = "Blocked link-local IP: {ip}"
SSRF_BLOCKED_MULTICAST = "Blocked multicast IP: {ip}"
SSRF_BLOCKED_UNSPECIFIED = "Blocked unspecified IP: {ip}"
SSRF_DNS_NO_ADDRESSES = "No addresses found for hostname '{hostname}'"
SSRF_DNS_RESOLUTION_FAILED = "DNS resolution failed for hostname '{hostname}': {error}"
SSRF_DOMAIN_NOT_ALLOWED = (
    "Webhook domain '{hostname}' is not in ALLOWED_WEBHOOK_DOMAINS."
)
SSRF_PORT_NOT_ALLOWED = (
    "Unauthorized port {port} in webhook URL. Allowed ports: {allowed_ports}."
)
SSRF_MAX_REDIRECTS_EXCEEDED = "Maximum HTTP redirect depth exceeded"
SSRF_CIRCULAR_REDIRECT_LOOP = "Circular HTTP redirect loop detected"

# API Errors
API_UNAUTHORIZED = "Invalid or missing authentication token."
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
UI_INDEX_LOAD_FAILED = "Error loading index: {error}"
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


EXPORT_WRITE_FAILED = (
    "Unable to write the {format_name} export to '{destination}'. "
    "Check the destination permissions and available disk space, then try again."
)

EXPORT_GENERATION_IO_FAILED = (
    "Unable to generate the {format_name} export because an I/O operation failed. "
    "Please try again."
)


class EmptyDocumentError(ValueError):
    """Raised when a document contains no extractable or readable text.

    This specific exception allows the UI and CLI to differentiate between
    a file that failed to parse due to corruption/format issues and a file
    that is simply blank or contains only images without OCR text.

    Attributes:
        filename: The name of the file that was empty.
        message: Explanation of the error.
    """

    def __init__(self, filename: str, message: str | None = None):
        self.filename = filename
        if message is None:
            self.message = f"The document '{filename}' contains no readable text."
        else:
            self.message = message

        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


# Event / Webhook Errors
EVENT_MALFORMED_PAYLOAD = "Malformed event payload: {error}"
EVENT_MISSING_FIELD = "Missing required event field: {field}"
EVENT_UNKNOWN_TYPE = "Unknown webhook event type: {event_type}"


class EventSchemaError(ValueError):
    """Raised when a webhook event payload violates the schema definition."""

    pass


class SSOConfigurationError(ValueError):
    """Raised when required SSO provider environment configuration (e.g. client ID or secret) is missing.

    Acceptance Criteria (Issue #2583):
    Subclasses ValueError so existing exception handlers and tests work seamlessly while providing
    a dedicated exception type for UI layers to catch and display as a graceful Streamlit error.
    """

    pass


class PDFEncryptedError(ValueError):
    """Raised when a PDF file is encrypted and password authentication fails or is not provided."""

    def __init__(
        self,
        message: str = "PDF is encrypted and password was not provided or invalid.",
    ):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

