# 🔍 Semantic Plagiarism Detection System

**[▶ Live Demo](https://semantic-plagiarism-detector.streamlit.app/)**

A production-ready NLP application that detects **semantic plagiarism** in student assignments, even when the text has been paraphrased. It uses **Sentence Transformers**, **cosine similarity**, and **FAISS vector search** to compare documents beyond simple copy-paste matching.

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
| **Transformer embeddings** | Uses `all-MiniLM-L6-v2` for fast and accurate 384-dimensional embeddings |
| **FAISS vector search** | Supports adaptive indexing using Flat / IVF for scalable document comparison |
| **Paragraph chunking** | Detects localized section-level plagiarism |
| **Similarity matrix** | Performs full N×N pairwise document comparison |
| **Heatmap visualization** | Displays Green-Red heatmaps with flagged-pair borders |
| **Pair drill-down** | Shows exactly which paragraphs are similar |
| **Custom text query** | Allows searching a pasted snippet against uploaded assignments |
| **Streamlit dashboard** | Provides a clean, teacher-friendly web interface |
| **Configurable threshold** | Adjustable similarity threshold using the sidebar slider |

---

## 🧱 System Architecture

```txt
                         ┌──────────────────────────────────────────┐
                         │          Streamlit Dashboard              │
                         │          app/streamlit_app.py             │
                         └─────────────────────┬────────────────────┘
                                               │
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │             PDF Text Extraction           │
                         │             utils/pdf_reader.py           │
                         └─────────────────────┬────────────────────┘
                                               │
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │           Paragraph Chunking              │
                         │           utils/text_chunking.py          │
                         └─────────────────────┬────────────────────┘
                                               │
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │        Sentence Transformer Embedding     │
                         │        utils/embedding_model.py           │
                         └─────────────────────┬────────────────────┘
                                               │
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │        FAISS Index + Similarity Search    │
                         │        utils/faiss_index.py               │
                         └─────────────────────┬────────────────────┘
                                               │
                                               ▼
                         ┌──────────────────────────────────────────┐
                         │       Similarity Matrix + Heatmap         │
                         │       utils/similarity.py, heatmap.py     │
                         └──────────────────────────────────────────┘
```

---

## 📦 Module Responsibilities

| Module | Responsibility |
|---|---|
| `utils/pdf_reader.py` | Extract text from uploaded PDF files |
| `utils/text_chunking.py` | Clean text and split it into paragraph-level chunks |
| `utils/embedding_model.py` | Generate Sentence Transformer embeddings |
| `utils/faiss_index.py` | Build FAISS index using Flat / IVF search |
| `utils/similarity.py` | Compute cosine similarity and flag plagiarism |
| `utils/heatmap.py` | Render Matplotlib / Seaborn heatmaps |
| `app/streamlit_app.py` | Streamlit UI for upload, warnings, search, heatmap, and drill-down |

---

## 📁 Project Structure

```txt
semantic_plagiarism_detector/
│
├── utils/
│   ├── __init__.py              # Package exports
│   ├── pdf_reader.py            # PDF text extraction
│   ├── text_chunking.py         # Paragraph-level chunking
│   ├── embedding_model.py       # Sentence Transformer wrapper
│   ├── faiss_index.py           # FAISS vector index
│   ├── similarity.py            # Cosine similarity and plagiarism flagging
│   └── heatmap.py               # Matplotlib / Seaborn visualisations
│
├── app/
│   └── streamlit_app.py         # Main web dashboard
│
├── evaluation/
│   ├── benchmark_dataset.json   # Labelled benchmark text pairs
│   ├── evaluate.py              # Evaluation script
│   └── results/                 # Generated plots and metrics
│
├── screenshots/                 # README screenshots
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Running

### 1. Clone / download the project

```bash
git clone https://github.com/Ganesh-403/semantic-plagiarism-detector.git
cd semantic-plagiarism-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate
```

```bash
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model.  
> Subsequent runs use the local cache.

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

After running the command, open the local Streamlit URL shown in the terminal.

---

## 🖥️ Dashboard Tabs

| Tab | What it shows |
|---|---|
| **Plagiarism Warnings** | All flagged pairs sorted by severity |
| **FAISS Chunk Search** | Chunk-level nearest-neighbour search across uploaded documents |
| **Similarity Matrix** | Full N×N similarity table with downloadable CSV |
| **Heatmap** | Visual color matrix with red borders on flagged pairs |
| **Pair Drill-Down** | Select two documents to inspect matching paragraphs |

---

## ⚙️ Configuration

| Setting | Default | Description |
|---|---|---|
| Similarity threshold | `0.75` | Pairs above this score are flagged |
| FAISS matches per chunk | `5` | Nearest neighbours retrieved per chunk |
| Chunk minimum words | `20` | Paragraphs shorter than this are discarded |
| Chunk maximum words | `200` | Longer paragraphs are split at sentence boundaries |
| Embedding model | `all-MiniLM-L6-v2` | Sentence Transformer model used for embeddings |
| Batch size | `64` | Batch size used during embedding generation |

---

## 🧪 How It Works

### Step 1 — Text Extraction

PDF files are uploaded through the Streamlit dashboard. The application extracts text from each page and combines it into a clean document string.

### Step 2 — Paragraph Chunking

Extracted text is split into paragraph-level chunks. Very short chunks are removed, while long paragraphs are split at sentence boundaries.

### Step 3 — Embedding Generation

Each paragraph chunk is passed through the `all-MiniLM-L6-v2` Sentence Transformer model.

- Output: 384-dimensional vector
- L2 normalization is used
- Cosine similarity is calculated using dot product

### Step 4 — FAISS Index

All chunk vectors are added to a FAISS index. The system automatically chooses a suitable index type based on collection size.

- **< 5,000 vectors** → `IndexFlatIP`
- **≥ 5,000 vectors** → `IndexIVFFlat`

Since embeddings are L2-normalised, inner product equals cosine similarity.

### Step 5 — Similarity Computation

The application computes similarity in two ways:

- **Document-level:** mean-pooled chunk embeddings → cosine similarity matrix
- **Chunk-level:** FAISS nearest-neighbour search → max similarity per chunk pair

### Step 6 — Flagging

Pairs with similarity greater than or equal to the selected threshold are flagged:

- **High:** `>= 0.90`
- **Medium:** `>= 0.75`

---

## 🧠 Why Semantic Similarity Catches Paraphrasing

The model encodes **meaning**, not just surface words.

Example:

> "The quick brown fox jumped over the lazy dog."  
> "A nimble auburn canine leapt above a lethargic hound."

Both sentences can produce nearly identical embeddings because the semantic meaning is similar, even though the wording is different.

---

## 📊 Performance

| Scenario | Expected time |
|---|---|
| First load | ~30–60 seconds because the model downloads once |
| 5 documents, CPU | ~10–15 seconds |
| 10 documents, CPU | ~20–30 seconds |
| 10 documents, GPU | ~5–8 seconds |
| 1000 documents, FAISS | Feasible with IVF indexing |

Results are **cached by Streamlit**, so re-uploading the same files is faster.

---

## 🔐 Privacy & Ethics

- All processing runs **locally** when the app is used on a local machine.
- This tool is an **aid for academic review**, not a final verdict.
- A high similarity score should trigger **manual review**, not automatic punishment.
- Students should be informed when submitted work is checked.

---

## 📚 Dependencies

| Library | Purpose |
|---|---|
| `sentence-transformers` | Pre-trained transformer embeddings |
| `faiss-cpu` | Vector search and nearest-neighbour retrieval |
| `PyPDF2` | PDF text extraction |
| `streamlit` | Web dashboard |
| `numpy` | Numerical operations |
| `pandas` | Similarity DataFrame handling |
| `scikit-learn` | `cosine_similarity` utility |
| `seaborn` | Heatmap styling |
| `matplotlib` | Figure rendering |

---

## 📈 Evaluation & Benchmarks

The system is evaluated on a **25-pair benchmark dataset** covering heavy paraphrases, light paraphrases, same-topic originals, and different-topic negatives.

Run the evaluation yourself:

```bash
python -m evaluation.evaluate
```

Results are saved to `evaluation/results/` and include:

| Output | Description |
|---|---|
| `metrics.json` | Precision, recall, F1, ROC-AUC at optimal threshold |
| `threshold_sweep_semantic.csv` | Metrics at every threshold |
| `roc_curve.png` | ROC curve comparing semantic model vs TF-IDF baseline |
| `pr_curve.png` | Precision-Recall curve |
| `similarity_distribution.png` | Score histograms by label |

### Benchmark Results

Evaluated on 25 text pairs:

| Metric | Sentence Transformers | TF-IDF Baseline | Δ |
|---|---:|---:|---:|
| **ROC-AUC** | **1.000** | 0.973 | +0.027 |
| **Best F1** | **1.000** | 0.667 | +0.333 |
| Precision | 1.000 | 1.000 | — |
| Recall | **1.000** | 0.500 | +0.500 |
| Accuracy | **1.000** | 0.800 | +0.200 |
| Optimal Threshold | 0.59 | 0.30 | — |

**Key finding:** TF-IDF misses heavy paraphrases because it relies mainly on exact word overlap. Sentence Transformers capture semantic meaning and can detect paraphrased content more effectively.

---

## 📝 License

MIT License. Free for academic and educational use.