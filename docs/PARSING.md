# Document Parsing Guidelines

## Supported Formats & File Size Limits

| Extension | Parser Library | OCR Support | Limitations / Max Size |
|---|---|---|---|
| `.pdf` | PyMuPDF / pdfplumber | Yes (Tesseract) | Max 10 MB; scanned pages require OCR |
| `.docx` | python-docx | No | Max 10 MB |
| `.txt` | Native Python | No | Max 10 MB; plain text only |
| `.rtf` | striprtf / native | No | Max 10 MB |
| `.epub` | ebooklib | No | Max 10 MB |
## Extraction Options

* **Text Parsing:** Extract raw text from standard documents.
* **OCR Fallback:** Automated OCR fallback for scanned PDF documents.

## OCR DPI Settings

* **Default DPI:** 250 DPI* **Recommended Range:** 150 - 400 DPI for optimal accuracy vs performance.
