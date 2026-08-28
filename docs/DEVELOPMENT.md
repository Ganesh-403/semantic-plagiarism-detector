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
pip install -r requirements-dev.txt
pip install pytest-cov
python -m nltk.downloader punkt_tab
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
pip install -r requirements-dev.txt
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

---

## 🚀 Production Deployment & PyTorch Worker Memory

The embedding pipeline uses a Sentence Transformers / PyTorch model loaded by
`EmbeddingModelManager`. The manager is a singleton **within one Python
process**; it is not a cross-process singleton.

When deploying behind a process-based server such as Gunicorn, independent
workers can each initialise their own model instance, multiplying model memory
usage.

### Recommended Gunicorn configuration

Prefer **one worker with multiple threads** when memory is constrained unless
you have measured that multiple model-owning processes fit the available budget.

```bash
gunicorn --workers 1 --threads 4 --bind 0.0.0.0:8501 <application-module>:<application>
```

Replace the application placeholder with the WSGI/ASGI entry point used by
your deployment. Do not use this command as a replacement for the Streamlit
entry point; Streamlit is normally launched with:

```bash
streamlit run app/streamlit_app.py
```

### Why worker count affects memory

Gunicorn workers are separate operating-system processes. Each process has its
own Python interpreter and its own `EmbeddingModelManager` instance.

| Gunicorn configuration | Model-owning processes | Expected model-memory impact |
|---|---:|---|
| `--workers 1 --threads 4` | 1 | One model copy |
| `--workers 2 --threads 4` | 2 | Approximately two model copies |
| `--workers 4 --threads 4` | 4 | Approximately four model copies |

Actual memory usage depends on the model, PyTorch runtime, allocator
behaviour, and CPU/GPU execution, so these figures are deployment guidance.

### PyTorch shared-memory multiprocessing

PyTorch multiprocessing and shared-memory primitives are not a drop-in
replacement for the current process-local singleton. Sharing model parameters
between worker processes requires deliberate process creation, tensor sharing,
lifecycle, and device management.

For the current architecture, limiting deployment to a single model-owning
worker is the simpler and more predictable memory-control strategy.

### Deployment checklist

1. Confirm how many Python processes will load `EmbeddingModelManager`.
2. Start with one worker and multiple threads when memory is constrained.
3. Measure resident memory after the embedding model loads.
4. Increase worker count only after confirming additional model copies fit the
   host or container memory limit.
5. If process-level model sharing is introduced later, add integration tests
   covering model initialisation, concurrent inference, shutdown, and the
   target CPU/GPU device.
