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
