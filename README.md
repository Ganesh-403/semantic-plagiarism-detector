# 🔍 Semantic Plagiarism Detection System

![Semantic Plagiarism Detector Banner](assets/hero_banner.png)

> **[▶ Live Demo](https://semantic-plagiarism-detector.streamlit.app/)**

A production-ready NLP application that detects **semantic plagiarism** in student
assignments—even when text has been paraphrased—using Sentence Transformers, cosine
similarity, and **FAISS vector search**.

---

## 📸 Screenshots

### Dashboard

![Dashboard](screenshots/screenshot_1_dashboard.png)

### Plagiarism Warnings

![Warnings](screenshots/screenshot_2_warnings.png)

### Similarity Heatmap

![Heatmap](screenshots/screenshot_3_heatmap.png)

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Semantic understanding** | Detects paraphrased plagiarism, not just copy-paste |
| **Transformer embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual, accurate) |
| **FAISS vector search** | Adaptive indexing (Flat / IVF) — scales to thousands of assignments |
| **Paragraph chunking** | Detects localised section-level plagiarism |
| **Similarity matrix** | Full N×N pairwise document comparison; downloadable as CSV or Excel |
| **Interactive heatmap** | Plotly heatmap with hover tooltips; toggle to static Seaborn view |
| **Pair drill-down** | See exactly which paragraphs match |
| **Custom text query** | Paste any snippet to search against all uploaded assignments |
| **Authentication** | Login system with role-based access (admin / teacher) |
| **User management** | Admin can create, reset passwords, and delete users |
| **Streamlit dashboard** | Clean, teacher-friendly web interface |
| **Configurable threshold** | Adjustable via sidebar slider (default 0.59) |

---

## 🏗️ System Architecture

```
                   ┌─────────────────────────────────────────────────┐
                   │              Streamlit Dashboard                │
                   │                (app/streamlit_app.py)           │
                   └────────────────────┬────────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │                  Processing Pipeline                │
              │                                                     │
              │  PDF Upload → Text Extraction → Paragraph Chunking  │
              │    → Embedding → FAISS Index → Similarity → Flags   │
              └─────────────────────────────────────────────────────┘
                    │         │          │         │        │       │
              ┌─────▼──┐ ┌───▼────┐ ┌───▼────┐ ┌──▼────┐ ┌▼─────┐ ┌▼──────┐
              │document│ │text_   │ │embed-  │ │faiss_ │ │simi- │ │heat-  │
              │_parser │ │chunking│ │ding_   │ │index  │ │larity│ │map.py │
              │.py     │ │.py     │ │model.py│ │.py    │ │.py   │ │       │
              └────────┘ └────────┘ └────────┘ └───────┘ └──────┘ └───────┘
```

> For a detailed explanation of the system components and data flow, see the [Architecture Guide](docs/ARCHITECTURE.md). To understand domain-specific terms (FAISS, Cosine Similarity, SSRF, WAL, TTR, etc.), reference the [Glossary](docs/GLOSSARY.md).

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/core/document_parser.py` | Extract raw text from PDF, DOCX, and TXT files |
| `src/core/text_chunking.py` | Split text into paragraph chunks (20–200 words) |
| `src/core/embedding_model.py` | Generate L2-normalised embeddings via SentenceTransformers |
| `src/core/faiss_index.py` | Build FAISS index (Flat/IVF); chunk-level search across all documents |
| `src/core/similarity.py` | Compute cosine similarity matrices; flag plagiarism |
| `src/core/translator.py` | Translate non-English matching paragraphs to English |
| `src/db/auth.py` | SQLite-backed authentication with bcrypt password hashing |
| `src/db/corpus_db.py` | SQLite database manager for metadata, text chunks, and embedding vectors |
| `src/visualization/heatmap.py` | Render Seaborn/Plotly heatmaps (document-level & chunk-level) |
| `src/visualization/network_graph.py` | Render interactive Plotly plagiarism networks using spring layout |
| `app/streamlit_app.py` | Streamlit UI: login, upload, warnings, FAISS search, heatmap, drill-down |

---

## 📁 Project Structure

```
semantic_plagiarism_detector/
├── .github/                  # CI/CD workflows and issue templates
│   ├── ISSUE_TEMPLATE/       # Bug report and feature request forms
│   └── workflows/            # GitHub Actions CI and lint workflows
├── app/                      # Streamlit application interface
│   ├── components/           # Incident export and UI helper components
│   ├── streamlit_app.py      # Main Streamlit dashboard entrypoint
│   └── theme.py              # Visual design system and CSS injection
├── assets/                   # Project visual assets & AI header graphics
├── src/                      # Core backend source package
│   ├── core/                 # Parsing, chunking, embedding, FAISS & similarity
│   ├── db/                   # SQLite authentication, corpus & incident databases
│   ├── utils/                # PDF reports, warning lists, badges & caching
│   └── visualization/        # Seaborn/Plotly heatmaps and network graphs
├── tests/                    # Comprehensive unit and integration test suite
│   ├── app/                  # UI and dashboard smoke tests
│   ├── core/                 # Core NLP, translation, and indexing tests
│   ├── db/                   # Database authentication and corpus tests
│   ├── utils/                # PDF reports, email, and cache tests
│   └── visualization/        # Network graph and heatmap tests
├── docs/                     # Detailed setup guides and integration docs
├── evaluation/               # Benchmark dataset and evaluation harness
├── screenshots/              # Dashboard UI preview images
├── CHANGELOG.md              # Version release history
├── CODE_OF_CONDUCT.md        # Contributor Covenant v2.1
├── CONTRIBUTING.md           # Developer setup and contribution guidelines
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── SECURITY.md               # Vulnerability reporting policy
├── SUPPORT.md                # Help channels and FAQ
├── pytest.ini                # Pytest configuration
└── requirements.txt          # Python dependencies
```

---

## 🚀 Setup & Running

For a detailed local setup guide detailing virtual environment creation, native C dependencies (Tesseract and Poppler) installation, and running pytest, refer to the [Developer Setup Guide](docs/DEVELOPMENT.md).

### 1. Clone / download the project

```bash
git clone https://github.com/your-org/semantic-plagiarism-detector.git
cd semantic-plagiarism-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install pytest-cov  # Required for coverage reporting
python -m nltk.downloader punkt_tab  # Pre-download NLTK corpus to avoid runtime delays
```

> **Note:** The first run will download the `paraphrase-multilingual-MiniLM-L12-v2` model (~420 MB).
> Subsequent runs use the local cache.

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

The app opens at **<http://localhost:8501>**.

### 5. Pre-populated Seed Data (Optional for Contributors)

To quickly test dashboard UI/CSS changes or verify logic without manually registering accounts or uploading documents, you can load pre-populated seed data:

```bash
# Load seed databases (users.db, corpus.db) and FAISS index (corpus.index)
make load-seed   # Or: python scripts/manage_seed.py load
```

After loading the seed data, launch the Streamlit dashboard and log in with the pre-configured contributor accounts:

* **Admin**: `admin` / `admin123`
* **Teacher**: `teacher` / `teacher123`

### Docker Deployment (recommended for quick setup)

One-command local deployment using Docker and Docker Compose. This builds a slim
Python 3.11 image with all dependencies and spins up the Streamlit dashboard plus
an optional Redis cache.

**Prerequisites:**
* Docker Engine 20.10+
* Docker Compose v2+

**Start the app:**

```bash
docker compose up --build
```

The dashboard is available at **<http://localhost:8501>**.

**Optional services:**
* **Redis** is included in `docker-compose.yml` for session caching and rate-limiting.
  The app runs without Redis and falls back to local in-memory state, so you can
  comment out the `redis` service if you only need the Streamlit UI.

**Environment variables:**

Customize behavior via a `.env` file in the project root or inline in
`docker-compose.yml`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `APP_BASE_URL` | `http://localhost:8501` | Base URL used in notifications |
| `SMTP_SERVER` | `smtp.gmail.com` | SMTP server for daily summary emails |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USERNAME` | | SMTP username |
| `SMTP_PASSWORD` | | SMTP password |
| `API_BEARER_TOKEN` | | Bearer token for REST API |
| `BACKUP_IDLE_TIMEOUT_MINUTES` | `30` | Duration of zero user activity (in minutes) before automated DB backup runs |

See `.env.example` for the full list.

**Securing Redis with TLS:**

By default, `docker-compose.yml` and the table above use a plaintext
`redis://` connection, which is fine for local development but should not be
used in production. [`SECURITY.md`](SECURITY.md#redis-security--access-control)
recommends encrypting Redis traffic in transit with TLS (`rediss://`). The
app doesn't need any code changes to support this — `REDIS_URL` is passed
straight to `redis-py`'s `redis.from_url()`, which natively understands the
`rediss://` scheme and standard `ssl_*` query parameters.

1. **Generate or obtain TLS certificates** for your Redis server (CA
   certificate, and optionally a client certificate/key pair if you're using
   `tls-auth-clients yes` as described in `SECURITY.md`). For local testing
   you can generate a self-signed CA with `openssl`; for production, use
   certificates issued by your organization's CA.

2. **Mount the certificates into both containers.** Add a volume mount to
   the `redis` and app services in `docker-compose.yml`:

   ```yaml
   services:
     app:
       volumes:
         - ./certs:/app/certs:ro   # add alongside the existing volumes

     redis:
       image: redis:7-alpine
       command: >
         redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
         --tls-port 6380 --port 0
         --tls-cert-file /certs/redis.crt
         --tls-key-file /certs/redis.key
         --tls-ca-cert-file /certs/ca.crt
       volumes:
         - ./certs:/certs:ro
         - redis_data:/data
       ports:
         - "6380:6380"
   ```

   Setting `--port 0` disables Redis's plaintext port entirely, so all
   connections must use TLS on `--tls-port`.

3. **Point `REDIS_URL` at the TLS port using the `rediss://` scheme,** and
   pass the CA certificate (and client cert/key, if configured) as query
   parameters:

   ```bash
   REDIS_URL=rediss://redis:6380/0?ssl_ca_certs=/app/certs/ca.crt&ssl_cert_reqs=required
   ```

   If you've enabled client certificate verification (`tls-auth-clients yes`),
   also include your client certificate and key:

   ```bash
   REDIS_URL=rediss://redis:6380/0?ssl_ca_certs=/app/certs/ca.crt&ssl_certfile=/app/certs/client.crt&ssl_keyfile=/app/certs/client.key&ssl_cert_reqs=required
   ```

4. **Restart the stack** so both containers pick up the new configuration:

   ```bash
   docker compose down
   docker compose up --build
   ```

For the full set of Redis hardening recommendations — strong passwords,
`requirepass`, and least-privilege ACLs — see the
[Redis Security & Access Control](SECURITY.md#redis-security--access-control)
section of `SECURITY.md`.

**Rebuild after dependency changes:**

```bash
docker compose build --no-cache
docker compose up
```

```markdown
**Stop the app:**

```bash
docker compose down
```

### Default credentials

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Admin — full access + user management |

Additional users can be created from the **User Management** page (admin only).

## ⚠️ Data Persistence & Docker Volumes

The app persists two SQLite databases plus the FAISS index. All three
live in the container filesystem and are wiped on `docker compose down -v`
**unless** they are mounted on named volumes. As of issue #3025, the
`docker-compose.yml` mounts three named volumes by default so the
data survives `down` / `up` cycles.

### What is persisted

| Volume name            | Container path | Holds                                           | Wiped by `down -v`? |
|------------------------|----------------|-------------------------------------------------|----------------------|
| `plagiarism_data`      | `/app/data`    | `corpus.db`, `corpus.index`, `backups/`         | ✅ Yes |
| `plagiarism_users`     | `/app`         | `users.db` (auth, roles, password hashes)        | ✅ Yes |
| `redis_data`           | `/data`        | Redis dump (session cache, rate-limit counters)  | ✅ Yes |

### Safe operations

```bash
# Stop the app — data is preserved.
docker compose down

# Restart — data is back, no migration needed.
docker compose up

---

## 🛠️ Troubleshooting

If you encounter issues while setting up the project locally, refer to the
[Troubleshooting Guide](docs/TROUBLESHOOTING.md) for solutions to common problems including:

- Tesseract OCR installation
- PyTorch CPU vs CUDA installation
- SQLite permission issues

---


## ⚓ Pre-commit Hooks

To maintain code quality and styling standards, we use client-side Git hooks managed by `pre-commit`. The hooks execute automatically before every commit to format and check code.

### Installation

1. Install the `pre-commit` utility:
   ```bash
   pip install pre-commit
   ```

2. Install the Git hooks:

   ```bash
   pre-commit install
   ```

After installation, the following checks run automatically on every staged file:
* **`black`**: Formats Python code.
* **`isort`**: Sorts import lines.
* **`ruff`**: Checks for lint warnings and errors.
* **`pre-commit-hooks`**: Performs basic validation (trailing whitespace, end-of-file fixer, check-yaml, check-added-large-files).

### Run Hooks Manually

You can manually trigger all hooks on all files in the repository at any time:

```bash
pre-commit run --all-files
```

---

## 💾 Database Backups

The system includes an automated background backup daemon that safely creates snapshots of the SQLite corpus database (`data/corpus.db`) during periods of inactivity.

### Idle Trigger & Daemon Semantics
* **Background Daemon:** A background thread polls every 30 seconds to monitor user session activity.
* **Idle Threshold:** When all user sessions are idle and no active user requests occur for the configured duration (default: **30 minutes** of zero activity), the daemon creates a timestamped database snapshot.
* **Rotation & Retention:** Automated backup rotation keeps only the **10 most recent backups** and automatically deletes backups older than **30 days** to prevent disk space exhaustion.

### Configuration Keys (`.env`)

| Key | Default | Description |
|---|---|---|
| `BACKUP_IDLE_TIMEOUT_MINUTES` | `30` | Duration of zero user activity (in minutes) required to trigger an automated database snapshot |

### Storage Location
* Automated backups are saved in the `data/backups/` directory (relative to the corpus database location).
* Backup files are timestamped using the naming convention `corpus_backup_YYYYMMDD_HHMMSS.db`.

---

## OCR support for scanned PDFs

Scanned and image-only PDFs are automatically detected page by page. Pages that
do not contain enough embedded text are rendered with PyMuPDF and processed
locally with Tesseract OCR. The extracted text then follows the same paragraph
chunking, embedding, FAISS, and similarity pipeline as regular PDFs.

### Python dependencies

```bash
python -m pip install pytesseract pymupdf pillow
```

### Tesseract system dependency

Tesseract must also be installed on the operating system.

On Windows, it is commonly installed at:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

When it is not available on PATH, set:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Verify the installation:

```powershell
tesseract --version
```

OCR is performed locally; uploaded documents are not sent to an external OCR
service.

---

## 🖥️ Dashboard — 5 Tabs

| Tab | What it shows |
|---|---|
| **Plagiarism Warnings** | All flagged pairs sorted by severity (High / Medium); downloadable CSV |
| **FAISS Chunk Search** | Chunk-level ANN search across all documents; custom text query box |
| **Similarity Matrix** | Full N×N similarity table; downloadable as CSV or Excel |
| **Heatmap** | Interactive Plotly heatmap (hover values) or static Seaborn view; downloadable PNG |
| **Pair Drill-Down** | Select any two docs to see which specific paragraphs match |

---

## ⚙️ Configuration

| Setting | Default | Description |
|---|---|---|
| Plagiarism threshold | `0.59` | Pairs above this score are flagged |
| FAISS matches per chunk | `5` | Nearest neighbours retrieved per chunk |
| Chunk min words | `20` | Paragraphs shorter than this are discarded |
| Chunk max words | `200` | Longer paragraphs are sub-split at sentence boundaries |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | Change in `src/core/embedding_model.py` or set `SEMANTIC_PLAGIARISM_MODEL` |
| Batch size | `64` | Tune for GPU/CPU in `src/core/embedding_model.py` |

---

## 🧠 How It Works

![Semantic Vector Search & AI NLP Architecture](assets/vector_search_concept.png)

### Step 1 – Text Extraction

PyPDF2 reads each PDF page and concatenates the text.

### Step 2 – Paragraph Chunking

Text is split on blank lines into chunks of 20–200 words.
Short chunks (headers, captions) are discarded; long chunks are sub-split at sentence boundaries.

### Step 3 – Embedding

Each chunk is passed through `paraphrase-multilingual-MiniLM-L12-v2`:
* Output: 384-dimensional, L2-normalised vector
* L2 normalisation means cosine similarity = dot product (fast)

### Step 4 – FAISS Index

All chunk vectors are added to a FAISS index. The system automatically selects the
best index type based on collection size:
* **< 5 000 vectors → `IndexFlatIP`** (exact inner-product search, O(N) per query)
* **≥ 5 000 vectors → `IndexIVFFlat`** (inverted-file approximate search, sub-linear per query)

Since embeddings are L2-normalised, inner product equals cosine similarity.

### Step 5 – Similarity Computation
* **Document-level:** mean-pooled chunk embeddings → cosine similarity matrix
* **Chunk-level:** FAISS ANN search → max similarity per chunk pair

### Step 6 – Flagging

Pairs with similarity >= threshold are flagged:
* **High**: >= 0.90
* **Medium**: >= 0.75 (default)

### Why semantic similarity catches paraphrasing

The model encodes **meaning**, not surface words:
> "The quick brown fox jumped over the lazy dog."
> "A nimble auburn canine leapt above a lethargic hound."

Both sentences produce nearly identical embeddings because the semantic content is the same.

---

## 📊 Performance

| Scenario | Expected time |
|---|---|
| First load (model download) | ~30–60 s (once only) |
| 5 documents, CPU | ~10–15 s |
| 10 documents, CPU | ~20–30 s |
| 10 documents, GPU | ~5–8 s |
| 1000 documents, FAISS | Feasible — auto-switches to IVF index |

Results are **cached by Streamlit** — re-uploading the same files is instant.

---

## 🔒 Privacy & Ethics

* All processing runs **locally**; no data leaves your machine.
* This tool is an **aid** for academic review, not a final verdict.
* A high similarity score should prompt **manual review**, not automatic sanctions.
* Consider informing students that submitted work will be checked.

---

## 🌐 REST API for External LMS Integrations

Expose a secure FastAPI endpoint for Learning Management Systems (Canvas, Moodle, Blackboard) to scan student submissions programmatically.

### Start the REST API Server

```bash
uvicorn src.api.app:app --reload --port 8000
```

### Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | `GET` | None | API health and readiness check |
| `/api/v1/scan` | `POST` | Bearer Token | Scan a document (`.pdf`, `.docx`, `.txt`) against the indexed corpus |

### Example Request (`curl`)

```bash
curl -X POST "http://localhost:8000/api/v1/scan?threshold=0.59" \
  -H "Authorization: Bearer dev-bearer-token" \
  -F "file=@student_essay.pdf"
```

### Example Response (`JSON`)

```json
{
  "filename": "student_essay.pdf",
  "word_count": 480,
  "chunk_count": 5,
  "plagiarism_flagged": true,
  "threshold_used": 0.59,
  "overall_document_similarity": 0.8523,
  "max_chunk_similarity": 0.9125,
  "matched_documents_count": 1,
  "matched_documents": [
    {
      "filename": "course_source_material.pdf",
      "document_similarity_score": 0.8523,
      "max_chunk_similarity_score": 0.9125,
      "severity": "🔴 High",
      "flagged_chunks": [
        {
          "uploaded_chunk": "Artificial Intelligence is rapidly reshaping higher education...",
          "matched_chunk": "AI models are transforming modern academic institutions...",
          "similarity_score": 0.9125
        }
      ]
    }
  ]
}
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `sentence-transformers` | Pre-trained transformer embeddings |
| `faiss-cpu` | Vector search (exact / approximate nearest-neighbour) |
| `PyPDF2` | PDF text extraction |
| `streamlit` | Web dashboard |
| `bcrypt` | Password hashing for authentication |
| `python-dotenv` | Load environment variables from `.env` |
| `numpy` | Numerical operations |
| `pandas` | Similarity DataFrame |
| `scikit-learn` | `cosine_similarity` utility |
| `plotly` | Interactive heatmap with hover tooltips |
| `seaborn` | Static heatmap styling |
| `matplotlib` | Figure rendering |
| `openpyxl` | Excel export for similarity matrix |

---

## 📊 Evaluation & Benchmarks

The system is evaluated on a **25-pair benchmark dataset** covering heavy paraphrases,
light paraphrases, same-topic originals, and different-topic negatives.

Run the evaluation yourself:

```bash
python -m evaluation.evaluate
```

For benchmark schema, contributor guidance, threshold sweeps, and output details,
see the [Evaluation and Benchmark Dataset Guide](evaluation/README.md).

Results are saved to `evaluation/results/` and include:

| Output | Description |
|---|---|
| `metrics.json` | Precision, recall, F1, ROC-AUC at optimal threshold |
| `threshold_sweep_semantic.csv` | Metrics at every threshold (0.30 – 0.95) |
| `roc_curve.png` | ROC curve — Semantic vs TF-IDF baseline |
| `pr_curve.png` | Precision-Recall curve |
| `similarity_distribution.png` | Score histograms by label |

### Benchmark Results

Evaluated on 25 text pairs (10 plagiarized, 15 not plagiarized):

| Metric | Sentence Transformers | TF-IDF Baseline | Δ |
|---|---|---|---|
| **ROC-AUC** | **1.000** | 0.973 | +0.027 |
| **Best F1** | **1.000** | 0.667 | +0.333 |
| Precision | 1.000 | 1.000 | — |
| Recall | **1.000** | 0.500 | +0.500 |
| Accuracy | **1.000** | 0.800 | +0.200 |
| Optimal Threshold | 0.59 | 0.30 | — |

**Key finding:** TF-IDF misses **all 5 heavy paraphrases** (scoring 0.18–0.27) while
Sentence Transformers correctly flags them (scoring 0.60–0.82). Light paraphrases are
detected by both, but the semantic model provides much stronger signal separation.

### Why semantic beats lexical

The TF-IDF baseline relies on exact word overlap — it fails when students paraphrase.
Sentence Transformers encode **meaning**, catching paraphrases that surface-level
methods miss entirely.

## Similarity threshold and severity configuration

All plagiarism and severity boundaries are defined in
`src/core/config.py`.

| Rule | Default |
|---|---:|
| Pair is flagged as plagiarism | `>= 0.59` |
| Medium severity | `>= 0.75` |
| High severity | `>= 0.90` |

The required ordering is:

```text
0.0 <= plagiarism <= medium <= high <= 1.0
```

The administrator slider controls which pairs are flagged. It does not redefine
the Medium or High severity bands.

Scores outside `[0.0, 1.0]` are clamped for consistent presentation. Invalid
non-numeric, NaN, or infinite values are rejected.

## Versioned SQLite schema migrations

`users.db` and `corpus.db` are upgraded automatically using SQLite
`PRAGMA user_version`.

Migration definitions live in:

```text
src/db/migrations/auth.py
src/db/migrations/corpus.py
src/db/migrations/common.py
```

Each upgrade:

1. reads the current schema version,
2. applies every missing migration in order,
3. runs inside a rollback-safe savepoint,
4. updates `PRAGMA user_version` only after all migrations succeed,
5. preserves existing users, documents, chunks, embeddings, and incidents.

Existing database files should not be deleted during an application upgrade.

---

## Linting

Before submitting a pull request, run the linting checks to ensure the code follows the project's formatting and type-checking standards.

Run all lint checks with:

```bash
make lint
```

## Documentation

* [Architecture](docs/ARCHITECTURE.md)
* [API Reference](docs/API.md)
* [Document Parsing & Formats](docs/PARSING.md)
* [NLP Architecture & Similarity Algorithm Guide](docs/ALGORITHMS.md)
* [Single Sign-On (SSO) Setup](docs/SSO_SETUP.md)

* [Bulk Export Formats & Data Fields](docs/EXPORTS.md)

* [UI Customization and Theme Guide](docs/THEMING.md)

---

## 📄 License

MIT License. Free for academic and educational use.

## Webhook retry behaviour

Plagiarism webhook delivery automatically retries temporary failures up to
three times with exponential backoff.

Retries apply to:

* connection failures and request timeouts,
* HTTP 408, 425, and 429,
* HTTP 500, 502, 503, and 504.

Permanent client errors such as HTTP 400 and 401 are not retried. Webhook SSRF
validation runs before dispatch and is never bypassed or retried.
