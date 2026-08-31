# Troubleshooting guide for Tesseract OCR dependencies

When users run the app on Windows or macOS without Tesseract installed, document parsing falls back to non-OCR mode with debug warnings like "Tesseract is not installed / OCR unavailable". To enable OCR functionality, you must install Tesseract OCR and its language packs (`tessdata`).

## Installation Guide

### 1. Ubuntu
Use the `apt` package manager to install Tesseract OCR and the necessary language packs.

```bash
sudo apt update
sudo apt install tesseract-ocr
# Install specific language packs (e.g., English)
sudo apt install tesseract-ocr-eng
# Install all language packs (optional)
sudo apt install tesseract-ocr-all
```

### 2. macOS
Use Homebrew to install Tesseract OCR.
`brew install tesseract` provides the base Tesseract formula and currently includes `eng`, `osd`, and `snum`.
`brew install tesseract-lang` provides additional languages.

```bash
brew install tesseract
# To install additional language packs:
brew install tesseract-lang
```

### 3. Windows
You can install Tesseract on Windows using Chocolatey or by downloading third-party binaries.

**Option A: Using Chocolatey (Recommended)**
Open your command prompt or PowerShell as an Administrator and run:
```powershell
choco install tesseract
```

**Option B: Third-Party Installers (UB Mannheim)**
The UB Mannheim project provides Windows installers for Tesseract.
1. Download the latest Windows installer from the [UB Mannheim wiki](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer. **Important:** During installation, select the language data you need if the installer provides that option.
3. Add the Tesseract installation directory (e.g., `C:\Program Files\Tesseract-OCR`) to your system's `PATH` environment variable.

## Verification

To verify that Tesseract is correctly installed and accessible on your system `PATH`, run:
```bash
tesseract --version
```

To list all available language packs (tessdata) installed on your system, run:
```bash
tesseract --list-langs
```

## Troubleshooting

- **"Tesseract is not installed / OCR unavailable" warning:**
  The application relies on Tesseract being available in your system's `PATH`. If you encounter this warning, ensure that the `tesseract` executable is in your system's `PATH`. You can verify this by running `tesseract --version` in a new terminal window. If it's not recognized, check your installation and `PATH` settings.

- **Missing language data (tessdata):**
  If OCR works for English but fails for other languages, ensure the correct language packs are installed. The location of the `tessdata` directory varies by distribution and version (e.g., package-managed locations on Linux or within the `C:\Program Files\Tesseract-OCR\tessdata` directory on Windows). You can manually download `.traineddata` files from the [tessdata repository](https://github.com/tesseract-ocr/tessdata) and place them in your system's appropriate `tessdata` directory.
