"""src/core/parsers/ocr_parser.py - Optical Character Recognition (OCR) strategies."""

import io
import logging
import os
from PIL import Image

from src.core.parsers.common import DEFAULT_OCR_DPI, DEFAULT_OCR_LANGUAGE, PDFInput

logger = logging.getLogger(__name__)


class OCRDependencyError(RuntimeError):
    """Raised when Tesseract or system dependencies required for OCR are missing."""

    pass


def _configure_tesseract(pytesseract_module) -> None:
    """Configure Tesseract binary location dynamically from system environment variables.

    Supports custom `TESSERACT_CMD` environment settings as well as standard Windows/Linux default paths
    (e.g., `C:\\Program Files\\Tesseract-OCR\\tesseract.exe` or `/usr/bin/tesseract`).
    """
    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    if configured_path:
        pytesseract_module.pytesseract.tesseract_cmd = configured_path
    elif os.name == "nt":
        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win_path):
            pytesseract_module.pytesseract.tesseract_cmd = default_win_path


def check_ocr_dependencies() -> None:
    """Check that required OCR Python packages and Tesseract executable are available.

    Raises:
        OCRDependencyError: If required Python packages (pytesseract, PyMuPDF, Pillow)
            or Tesseract binary are missing/unavailable.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        from src.errors import OCR_DEPENDENCIES_MISSING

        raise OCRDependencyError(OCR_DEPENDENCIES_MISSING) from exc

    _configure_tesseract(pytesseract)

    try:
        pytesseract.get_tesseract_version()
    except (pytesseract.TesseractNotFoundError, EnvironmentError, Exception) as exc:
        from src.errors import OCR_TESSERACT_NOT_FOUND

        raise OCRDependencyError(OCR_TESSERACT_NOT_FOUND) from exc



def _is_blank_scanned_page(
    pdf_bytes: bytes,
    page_index: int,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    variance_threshold: float = 5.0,
) -> bool:
    """Return True if a rendered page looks blank (very low pixel variance)."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError:
        return False

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page = document.load_page(page_index)
            scale = dpi / 72
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            ).convert("L")

        histogram = image.histogram()
        pixel_count = image.width * image.height
        if pixel_count == 0:
            return True

        mean = sum(i * count for i, count in enumerate(histogram)) / pixel_count
        variance = (
            sum(count * ((i - mean) ** 2) for i, count in enumerate(histogram))
            / pixel_count
        )
        return variance < variance_threshold
    except Exception as exc:
        logger.error(f"[document_parser] Error checking blank page {page_index}: {exc}")
        return False


def _ocr_pdf_page(
    pdf_bytes: bytes,
    page_index: int,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    language: str = DEFAULT_OCR_LANGUAGE,
) -> str:
    """Render one PDF page and extract text with Tesseract."""
    check_ocr_dependencies()

    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    from src.core.metrics import ocr_invocations_total
    from src.utils.temp_manager import managed_ocr_temp_dir

    ocr_invocations_total.labels(status="started").inc()
    try:
        with managed_ocr_temp_dir(prefix=f"ocr_pdf_p{page_index}_") as tmp_dir:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                page = document.load_page(page_index)
                scale = dpi / 72
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                extracted = pytesseract.image_to_string(
                    image,
                    lang=language,
                    config="--oem 3 --psm 3",
                ).strip()
                ocr_invocations_total.labels(status="success").inc()
                return extracted
    except Exception as exc:
        ocr_invocations_total.labels(status="failure").inc()
        logger.error(f"[document_parser] OCR page extraction failed: {exc}")
        return ""


def preprocess_image_for_ocr(image):
    """Preprocess standalone images (contrast enhancement, binarization, noise reduction) prior to OCR.

    Parameters
    ----------
    image : PIL.Image.Image
        Input PIL Image object.

    Returns
    -------
    PIL.Image.Image
        Preprocessed image ready for pytesseract.image_to_string.
    """
    try:
        from PIL import ImageEnhance, ImageFilter

        # Convert palette/RGBA modes to RGB for uniform channel processing
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            bg = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            bg.paste(image, mask=image.split()[-1])
            image = bg
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to grayscale for OCR optimization
        gray = image.convert("L")

        # Contrast enhancement to sharpen scanned text against background noise
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)

        # Subtle median filter for noise suppression
        filtered = enhanced.filter(ImageFilter.MedianFilter(size=3))
        return filtered
    except Exception as exc:
        logger.debug(f"[ocr_parser] Image preprocessing fallback: {exc}")
        return image


def extract_text_from_image(
    file: PDFInput,
    *,
    ocr_language: str = DEFAULT_OCR_LANGUAGE,
    preprocess: bool = True,
) -> str:
    """Extract native text directly from standalone image files (.png, .jpg, .jpeg) using Tesseract OCR.

    Feature Issue #2720:
    --------------------
    Extends plagiarism detection pipeline capabilities beyond PDF embedded images to natively ingest and
    process standalone screenshot uploads, essay photos, and scanned page images.

    Preprocessing Pipeline:
    - Normalizes RGBA/palette images onto a solid white background.
    - Applies grayscale conversion, contrast enhancement (1.8x factor), and median filtering.
    - Routes preprocessed image data to `pytesseract.image_to_string` with OEM engine 3 and PSM mode 3.

    Parameters
    ----------
    file : PDFInput
        Bytes, file path, or buffer containing raw PNG/JPG image data.
    ocr_language : str
        Tesseract language code (default: 'eng').
    preprocess : bool
        If True, applies contrast enhancement and noise reduction prior to OCR.

    Returns
    -------
    str
        Extracted text string stripped of leading/trailing whitespace.
    """
    check_ocr_dependencies()

    import pytesseract
    from PIL import Image

    from src.core.metrics import ocr_invocations_total
    from src.core.parsers.pdf_parser import _read_pdf_bytes
    from src.utils.temp_manager import managed_ocr_temp_dir

    file_bytes = _read_pdf_bytes(file)
    ocr_invocations_total.labels(status="started").inc()
    try:
        with managed_ocr_temp_dir(prefix="ocr_image_") as tmp_dir:
            image = Image.open(io.BytesIO(file_bytes))
            if preprocess:
                processed_img = preprocess_image_for_ocr(image)
            else:
                processed_img = image

            try:
                extracted_text = pytesseract.image_to_string(
                    processed_img,
                    lang=ocr_language,
                    config="--oem 3 --psm 3",
                ).strip()
                ocr_invocations_total.labels(status="success").inc()
                return extracted_text
            except (MemoryError, Exception) as exc:
                ocr_invocations_total.labels(status="failure").inc()
                if isinstance(exc, MemoryError):
                    logger.warning(
                        f"[document_parser] OCR image extraction failed due to memory exhaustion: {exc}"
                    )
                else:
                    logger.warning(f"[document_parser] OCR image extraction failed: {exc}")
                return "[OCR extraction failed for the file]"
    except pytesseract.TesseractNotFoundError as exc:
        ocr_invocations_total.labels(status="failure").inc()
        from src.errors import OCR_TESSERACT_NOT_FOUND

        raise OCRDependencyError(OCR_TESSERACT_NOT_FOUND) from exc
    except Exception as exc:
        ocr_invocations_total.labels(status="failure").inc()
        logger.error(f"[document_parser] Error reading standalone image: {exc}")
        return ""
