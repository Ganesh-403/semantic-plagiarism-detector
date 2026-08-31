# Support Policy

Thank you for using the **Semantic Plagiarism Detection System**!

## 🙋 Need Help?

Before opening a new issue, please check the following resources:

1. **[Documentation & README](README.md):** Architectural diagrams, installation guides, and configuration parameters.
2. **[GitHub Issues](https://github.com/Ganesh-403/semantic-plagiarism-detector/issues):** Search existing open and closed issues to see if your question or bug has already been addressed.
3. **[GitHub Discussions](https://github.com/Ganesh-403/semantic-plagiarism-detector/discussions):** For Q&A, usage suggestions, and general ideas.

## 🐛 Found a Bug?

If you have encountered a bug or unexpected behavior:
- Check existing issues first.
- Open a new issue using the **[Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md)**.
- Provide step-by-step reproduction steps, error logs, and environment details (Python version, OS, Streamlit version).

## ✨ Requesting Features

Have an idea to improve the application?
- Open a new issue using the **[Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md)**.
- Describe the problem your feature solves and your proposed user experience.

## 🔒 Security Vulnerabilities

If you discover a security flaw or vulnerability (such as SSRF, SQL Injection, remote code execution, or authentication bypasses), **please do NOT open a public GitHub issue**. Publicly disclosing security flaws can expose application instances to risk before a patch is available.

Instead, please report security vulnerabilities privately:
- **Email Alias:** [security@domain.com](mailto:security@domain.com)
- **Detailed Policy:** Refer to our **[Security Policy (SECURITY.md)](SECURITY.md)** for our full vulnerability disclosure guidelines, response timelines, and security best practices.

---

## 📋 Support Response Times

This section outlines our Support Level Agreements (SLAs) and expected response timelines for different types of community support requests. These guidelines help us prioritize issues and set expectations for response times.

### Response Time Matrix

| Issue Type | Priority | Initial Response | Resolution Target | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Critical Security Vulnerability** | Critical | 24 hours | 48 hours | Reported via private security email. Immediate triage and patch development. |
| **Bug Reports** | High | 48-72 hours | 1-2 weeks | Reproducible bugs with clear steps and logs. |
| **Feature Requests** | Medium | 1 week | Case-by-case | Evaluated based on community demand and roadmap alignment. |
| **General Questions** | Low | 1-2 weeks | Best-effort | GitHub Discussions recommended for Q&A. |
| **Documentation Clarifications** | Low | 1-2 weeks | Best-effort | May be addressed in documentation updates. |

### Support Level Details

#### 🔴 Critical Issues
- **Definition:** Security vulnerabilities, production outages, data loss risks.
- **Response:** Best-effort within 24 hours via email.
- **Channel:** Private email to [security@domain.com](mailto:security@domain.com)
- **SLA:** Target patch delivery within 48 hours of confirmation.

#### 🟠 High Priority Issues
- **Definition:** Reproducible bugs affecting core functionality, blocking workflows.
- **Response:** Initial response within 48-72 hours.
- **Channel:** GitHub Issues with bug report template.
- **Resolution Target:** 1-2 weeks, depending on severity and complexity.

#### 🟡 Medium Priority Issues
- **Definition:** Feature requests, enhancements, non-critical bugs.
- **Response:** Initial response within 1 week.
- **Channel:** GitHub Issues or GitHub Discussions.
- **Resolution Target:** Evaluated based on community interest and resource availability.

#### 🟢 Low Priority Issues
- **Definition:** General questions, documentation clarifications, usage guidance.
- **Response:** Best-effort within 1-2 weeks.
- **Channel:** GitHub Discussions recommended.
- **Resolution Target:** Best-effort, may be addressed in batch documentation updates.

### Community Support Expectations

- **Best-Effort Basis:** Community support is provided on a best-effort basis. Response times are targets, not guarantees, and may vary based on team availability and issue complexity.
- **Volunteer Contributions:** This project is maintained by volunteer contributors. Please be patient and respectful in your interactions.
- **Escalation Path:** For urgent business-critical issues, consider opening a GitHub Sponsor inquiry or contacting [security@domain.com](mailto:security@domain.com) for guidance.

---
## ⚡ Troubleshooting Batch Uploads

When processing large batches of PDF/DOCX assignments or generating high-dimensional vector embeddings, memory allocation spikes can occur. Below are common FAQs, memory management tips, and recommended environment settings to ensure smooth throughput and prevent Out-Of-Memory (OOM) errors.

### ❓ Frequently Asked Questions (FAQs)

#### 1. Why does memory consumption spike during PDF batch uploads?
PDF extraction (especially with OCR via Tesseract or PyMuPDF) loads raw page buffers and uncompressed image streams into memory. When multiple multi-page PDFs are ingested simultaneously, memory usage accumulates in PyTorch/Transformer model buffers and un-Garbage-Collected file descriptors.

#### 2. How can I optimize vector index generation for thousands of documents?
Vector embedding models convert document text chunks into dense vector embeddings. To prevent memory exhaustion:
- Process document embeddings in fixed mini-batches rather than holding full corpus tensors in RAM.
- Use `FAISS` index types tailored for memory efficiency (`IndexFlatIP` for small-to-medium corpora, or `IndexIVFFlat` for large scale corpora).
- Trigger manual Python `gc.collect()` calls after batch vector generation.

---

### 💡 Memory Management Tips

1. **Adjust Chunking Parameters**: Decrease `CHUNK_SIZE` (e.g., 250–500 tokens) and `CHUNK_OVERLAP` to prevent excessively long chunk strings from consuming extra memory during transformer inference.
2. **Limit Concurrent Worker Threads**: Restrict background thread pool workers when parsing large document sets to avoid resource contention on systems with limited CPU RAM.
3. **Opt-in OCR Scanning**: Enable OCR only when scanned image PDFs are detected. Set `DEFAULT_OCR_DPI` to `150` or `200` instead of high-DPI settings (300+) to drastically reduce image extraction buffers.
4. **Periodic FAISS Index Persistence**: Flush and save FAISS index states to disk (`save_index`) periodically during large batch ingestion instead of storing all vector matrices in memory.

---

### ⚙️ Recommended Environment Settings

To optimize performance and memory footprint in production or Docker environments, set the following environment variables in your `.env` file or deployment specification:

| Environment Variable | Recommended Value | Purpose / Description |
| :--- | :--- | :--- |
| `OMP_NUM_THREADS` | `4` | Controls OpenMP CPU threads for FAISS and PyTorch to prevent thread over-subscription. |
| `MKL_NUM_THREADS` | `4` | Limits Intel MKL CPU thread allocation. |
| `MALLOC_TRIM_THRESHOLD_` | `65536` | Forces glibc memory allocator to release freed memory back to the OS aggressively (Linux). |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces CUDA/CPU memory fragmentation during large vector embedding inference. |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | `200` | Sets maximum upload file size (in MB) per request in Streamlit. |

#### Example `.env` Configuration for High-Volume Deployments
```env
# Thread & Memory Controls
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
MALLOC_TRIM_THRESHOLD_=65536
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Streamlit App Limits
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

---

## ❓ Frequently Asked Questions (Top Support Queries)

This troubleshooting FAQ provides direct solutions for the top 5 common operational and setup questions:

### 1. How do I install Tesseract OCR for scanned document parsing?
Document OCR parsing relies on system-level Tesseract installations. Install it via your platform package manager:
- **Ubuntu / Debian:** `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-eng`
- **macOS:** `brew install tesseract tesseract-lang`
- **Windows:** Run `choco install tesseract` or download installer binaries from [UB-Mannheim Wiki](https://github.com/UB-Mannheim/tesseract/wiki) and append `C:\Program Files\Tesseract-OCR` to system `PATH`.
- **Verification:** Run `tesseract --version` and `tesseract --list-langs` in a terminal window.

For step-by-step instructions and multi-language pack configuration, refer to the [Tesseract OCR Setup Guide](docs/ocr_setup.md) and [Troubleshooting Guide](docs/TROUBLESHOOTING.md#tesseract-ocr-missing-binary-error).

### 2. How do I reset an administrator or user password?
Password resets can be performed via CLI management tools or administrative Python modules:
```bash
python -m src.cli.manage_users reset-password --username admin --new-password "YourNewSecretPassword123!"
```
Administrators can also manage password rotation policies and unlock locked accounts via the Web UI Settings panel.

For complete password policies, security options, and token revocation, see the [Authentication Guide](docs/AUTHENTICATION.md) and [CLI Guide](docs/CLI_GUIDE.md).

### 3. How do I run the application using Docker and Docker Compose?
You can spin up the full service stack (Streamlit Frontend, FastAPI Backend, Redis Cache, and FAISS vector worker) with a single command:
```bash
docker compose up --build -d
```
- **Streamlit Web UI:** `http://localhost:8501`
- **FastAPI REST API:** `http://localhost:8000`
- **Stopping Services:** `docker compose down`

For container configuration parameters and production image optimization, refer to the [Docker Deployment Guide](README.md#docker-deployment-recommended-for-quick-setup) and [Deployment Documentation](docs/deployment.md).

### 4. How do I configure Redis for session caching and rate limiting?
Set the Redis connection parameters in your `.env` configuration file:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password
```
Test the connection using `redis-cli -h localhost -p 6379 ping`. If Redis is offline, the application automatically degrades to in-memory caching.

For setup details, fallback behavior, and cluster tuning, refer to the [Redis Setup Guide](docs/REDIS_SETUP.md) and [Redis Performance Guide](docs/REDIS_PERFORMANCE.md).

### 5. How do I resolve Out-Of-Memory (OOM) errors during high-volume document uploads?
spikes during large batch embedding generation can be prevented with the following configuration:
1. **Thread Limits:** Add OpenMP/MKL flags to your `.env` file to prevent CPU over-subscription:
   ```env
   OMP_NUM_THREADS=4
   MKL_NUM_THREADS=4
   MALLOC_TRIM_THRESHOLD_=65536
   ```
2. **Chunk Size:** Decrease `CHUNK_SIZE` (e.g. 250-500 tokens) in [Chunking Strategies](docs/CHUNKING_STRATEGIES.md).
3. **FAISS Index Persistence:** Enable periodic disk flushes (`save_index`) to reduce vector memory footprint.

For comprehensive memory management tips, check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md#memory-allocation-or-out-of-memory-errors).

