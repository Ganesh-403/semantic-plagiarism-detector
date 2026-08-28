# Troubleshooting

This guide covers common issues contributors may encounter when setting up the project locally.

---

## Tesseract OCR: Missing Binary Error

### Symptoms

Examples include:

```text
TesseractNotFoundError: tesseract is not installed or it's not in your PATH
```

or

```text
No executable found: tesseract
```

### Solution

1. Install Tesseract OCR.

- **Windows:** Download and install Tesseract, then add the installation directory (e.g. `C:\Program Files\Tesseract-OCR`) to your system `PATH`.
- **macOS:**

  ```bash
  brew install tesseract
  ```

- **Ubuntu/Debian:**

  ```bash
  sudo apt update
  sudo apt install tesseract-ocr
  ```

2. Verify the installation:

```bash
tesseract --version
```

If your application requires it, configure the path to the Tesseract executable according to the project's documentation.

---

## PyTorch Installation (CPU vs CUDA)

Choose the installation that matches your hardware.

### CPU-only

```bash
pip install torch torchvision torchaudio
```

### NVIDIA GPU (CUDA)

Install the CUDA-enabled version recommended on the official PyTorch website.

Visit:

<https://pytorch.org/get-started/locally/>

Verify your installation:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

If `torch.cuda.is_available()` returns `False`, verify that:

- NVIDIA drivers are installed.
- CUDA version matches your PyTorch installation.
- Your GPU is supported.

---

## SQLite Permission Errors

### Symptoms

```text
sqlite3.OperationalError: attempt to write a readonly database
```

or

```text
database is locked
```

### Solutions

- Ensure the database file is writable.
- Verify write permissions for the directory containing the database.
- Close any applications currently using the database.
- Avoid opening the same SQLite database from multiple write processes simultaneously.
- If needed, delete the local development database and allow the application to recreate it.

> **Note:** Do not delete `users.db` or `corpus.db` as a first troubleshooting step. The project uses versioned SQLite migrations to preserve existing data.

---

## ModuleNotFoundError or dependency installation errors

``` text
ModuleNotFoundError: No module named '<package>'
```

### Possible causes

- Project dependencies have not been installed.
- The virtual environment is not activated.
- The dependency installation was interrupted or failed.

### Solution

- Make sure your virtual environment is activated and install the required dependencies:

```bash
python -m venv venv
```

- Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

If the missing module is related to OCR support, install the additional OCR dependencies:

```bash
python -m pip install pytesseract pymupdf pillow
```

## Sentence Transformer model download or loading errors

The application may fail while loading paraphrase-multilingual-MiniLM-L12-v2, or the first startup may appear to hang while downloading the model.

### Possible causes

- The model has not been downloaded before.
- The machine has limited or unstable internet connectivity.
- There is not enough disk space for the model cache.

### Solution

- Make sure you have an active internet connection during the first run. The application downloads the paraphrase-multilingual-MiniLM-L12-v2 model (approximately 420 MB) and caches it locally.
- After the initial download, subsequent runs should use the cached model. If the model download fails, restart the application and try again. Also verify that sufficient disk space is available.

---

## Memory allocation or out-of-memory errors

The application may become slow, crash, or report memory allocation errors when processing a large number of documents or building FAISS indexes.

### Possible causes

- A large number of documents or text chunks are being processed at once.
- Embedding generation requires more RAM than is currently available.
- FAISS is indexing a large collection of vectors.
- The embedding batch size is too high for the available hardware.

### Solutions

Try the following:

- Close other applications to free system memory.
- Process fewer documents at a time.
- Reduce the embedding batch size in src/core/embedding_model.py.
- If using a GPU, make sure sufficient GPU memory is available.
- Restart the application to clear unused memory. For larger collections, the application automatically switches from IndexFlatIP to IndexIVFFlat when the number of vectors reaches 5,000 or more. If the problem persists, reduce the batch size and try processing the documents again.

## Still Having Issues?

If the problem persists:

- Review the project README.
- Search existing GitHub issues.
- Open a new issue if your problem has not already been reported.
