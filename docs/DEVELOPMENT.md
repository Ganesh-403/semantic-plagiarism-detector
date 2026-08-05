# Local Development Setup Guide

Welcome! This guide will walk you through setting up your local environment for developing and testing the Semantic Plagiarism Detection System.

---

## 🛠️ Step 1: Create a Virtual Environment

It is recommended to use a Python virtual environment to manage dependencies.

### Windows (PowerShell)
```powershell
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### macOS / Linux (Terminal)
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

---

## 📦 Step 2: Install Python Dependencies

After activating your virtual environment, upgrade pip and install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest-cov
```

---

## ⚙️ Step 3: Install Native C Dependencies

The application relies on `Tesseract` (for OCR parsing of images/scanned PDFs) and `Poppler` (for PDF rendering/highlighting). Follow the setup instructions for your operating system:

### 🪟 Windows
1. **Tesseract OCR**:
   - Install via Winget in PowerShell:
     ```powershell
     winget install UB.TesseractOCR
     ```
   - Alternatively, download and run the installer from [UB Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki).
   - Ensure the installation directory (usually `C:\Program Files\Tesseract-OCR`) is added to your system's `PATH` environment variable.
2. **Poppler**:
   - Download the latest pre-compiled Poppler binaries for Windows from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases/).
   - Extract the ZIP archive (e.g., to `C:\Program Files\poppler`).
   - Add the `bin/` subdirectory (e.g., `C:\Program Files\poppler\Library\bin`) to your system's `PATH` environment variable.

### 🍎 macOS
Install both dependencies easily using [Homebrew](https://brew.sh/):
```bash
brew install tesseract poppler
```

### 🐧 Linux (Ubuntu / Debian)
Install dependencies using `apt`:
```bash
sudo apt update
sudo apt install -y tesseract-ocr poppler-utils libtesseract-dev
```

---

## 🧪 Step 4: Running Tests

The test suite uses `pytest` for unit and integration testing.

### Run All Tests
```bash
pytest
```

### Run Tests in a Specific File
```bash
pytest tests/db/test_database_backup.py
```

### Run Tests Bypassing Default Coverage (Local Quick Run)
If you have config options causing coverage failures locally:
```bash
pytest -o addopts="" tests/db/test_database_backup.py
```

### Generate and View the HTML Coverage Report
Run the test suite with HTML coverage and open `htmlcov/index.html` automatically in your default browser:
```bash
python scripts/coverage_report.py
```
To generate the report without opening the browser, add `--no-open`:
```bash
python scripts/coverage_report.py --no-open
```

---

## 🎨 Step 5: Formatting & Linting (Pre-commit)

This project uses `pre-commit` to automatically check formatting, linting, and validation rules before commits are finalized.

### Install and Register Git Hooks
```bash
pip install pre-commit
pre-commit install
```

### Run Checks Manually
To run checks on all files:
```bash
pre-commit run --all-files
```

To run checks only on staged files:
```bash
pre-commit run
```

---

## 📥 Step 6: Generating Seed Data

To populate your local SQLite and FAISS database with mock student submissions and plagiarism test cases:

```bash
python scripts/generate_seed_data.py --reset-db --include-plagiarism
```

For more details on optional parameters, refer to the seed generation guide in `scripts/generate_seed_data.py`.
