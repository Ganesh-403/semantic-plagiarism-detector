# Parser Guide

## Overview

The document parsing system extracts plain text from supported document formats using dedicated parser functions. Each supported file type has its own parser implementation, while the `extract_text()` function acts as the central dispatcher that selects the appropriate parser based on the uploaded file's extension.

The parser module currently supports the following formats:

- PDF (`.pdf`)
- Microsoft Word (`.doc`, `.docx`)
- Plain Text (`.txt`)
- Rich Text Format (`.rtf`)
- Markdown (`.md`, `.markdown`, `.mdown`)
- EPUB (`.epub`)
- HTML (`.html`)
- CSV (`.csv`)

The extracted text is then processed using helper functions such as text cleaning, bibliography removal, zero-width character sanitization, and language detection before being used by the application.

---

# Parser Flow

When a document is uploaded, the parsing pipeline works as follows:

1. `extract_text()` receives the uploaded file and its filename.
2. The file extension is determined from the filename.
3. The appropriate parser function is selected based on the extension.
4. The parser extracts readable plain text from the document.
5. The extracted text is cleaned and normalized using helper functions.
6. The processed text is returned to the caller.

---

# Parser Interface

Every custom parser should follow the same interface used throughout `document_parser.py`.

### Function Signature

```python
def extract_text_from_<format>(file: PDFInput) -> str:
```

### Requirements

A parser should:

- Accept a `PDFInput` object (file path, bytes, or file-like object).
- Extract readable plain text from the document.
- Return the extracted text as a `str`.
- Return an empty string (`""`) when extraction cannot be completed but the error is recoverable.
- Raise project-specific exceptions only when the caller must handle the failure.
- Avoid modifying the original input.
- Avoid printing directly to the console.

---

# Creating a New Parser

To add support for a new document format:

### Step 1

Create a parser function following the existing naming convention.

```python
def extract_text_from_tex(file: PDFInput) -> str:
    """
    Extract text from a TEX document.
    """

    # Parse document

    return extracted_text
```

### Step 2

Implement the parsing logic.

The parser should:

- Read the document.
- Extract readable text.
- Return plain text only.
- Handle parsing failures consistently.

### Step 3

Reuse existing helper functions whenever appropriate, including:

- `clean_text()`
- `strip_bibliography()`
- `sanitize_zero_width_characters()`

This ensures consistent output across all supported formats.

---

# Integrating the Parser into `document_parser.py`

After implementing the parser, register it inside the `extract_text()` dispatcher by adding support for the new file extension.

Example:

```python
elif extension == "tex":
    raw = extract_text_from_tex(file)
```

If the format should be accepted by the application, also add the extension to the `ALLOWED_EXTENSIONS` set.

Example:

```python
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".tex",
}
```

After registration, verify that the new parser is selected correctly when a file with the new extension is uploaded.

---

# Error Handling Conventions

All parser implementations should follow the project's existing error handling conventions.

Recommended practices include:

- Catch parser-specific exceptions.
- Log unexpected errors using the project logger.
- Return an empty string for recoverable parsing failures.
- Raise custom exceptions when the caller must handle the failure.
- Avoid silently ignoring parsing errors.

Examples of project-specific exceptions include:

- `OCRDependencyError`
- `CorruptedArchiveError`

---

# Writing Tests

Every new parser should include unit tests.

Recommended test cases:

- Valid document
- Empty document
- Corrupted document
- Unsupported file extension
- Invalid document format
- Unicode characters
- Large documents
- Invalid encoding

Tests should verify that:

- The parser extracts the expected text.
- Invalid input is handled safely.
- Errors do not crash the parser.
- The parser follows the expected interface.
- The parser is correctly integrated with `extract_text()`.

---

# Best Practices

- Keep parser functions focused on text extraction.
- Return plain text only.
- Reuse existing helper functions whenever possible.
- Follow the project's logging and error handling conventions.
- Register every new parser inside `extract_text()`.
- Update `ALLOWED_EXTENSIONS` when introducing a new supported format.
- Add unit tests for every new parser.
- Document any additional third-party dependencies required by the parser.

---

# Example Workflow

To add support for a new document type:

1. Create `extract_text_from_<format>()`.
2. Implement the parsing logic.
3. Handle parser-specific errors.
4. Register the parser inside `extract_text()`.
5. Add the extension to `ALLOWED_EXTENSIONS`.
6. Create unit tests covering normal and error cases.
7. Verify that documents using the new extension are parsed successfully.
