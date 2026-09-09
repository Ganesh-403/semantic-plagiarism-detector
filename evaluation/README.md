# Evaluation and Benchmark Dataset Guide

## Overview

The evaluation toolkit measures how well the project's text-pair detector separates
semantically plagiarized passages from independently authored passages. For each
pair, it calculates:

- Sentence Transformer cosine similarity (the semantic approach).
- TF-IDF cosine similarity (a lexical baseline).
- A hybrid score: `0.7 * semantic + 0.3 * TF-IDF`.

The evaluator uses the labels in [`benchmark_dataset.json`](./benchmark_dataset.json)
as ground truth, applies thresholds to the scores, and reports precision, recall,
F1, accuracy, and ROC-AUC. The benchmark exists so contributors can compare model
or dataset changes using the same labelled examples.

This guide is for contributors adding records, maintainers comparing model changes,
and developers studying how a decision threshold changes semantic-plagiarism
classification. A score is continuous; the threshold turns it into a binary
decision. Changing the threshold changes the balance between missed plagiarism and
incorrect flags.

## Evaluation Directory Structure

```text
evaluation/
├── __init__.py
├── README.md
├── benchmark_dataset.json
├── evaluate.py
└── results/                 # generated and ignored by Git
```

- [`benchmark_dataset.json`](./benchmark_dataset.json) contains the labelled text
  pairs and their metadata.
- [`evaluate.py`](./evaluate.py) loads the benchmark, computes semantic, TF-IDF,
  and hybrid scores, calculates metrics, and writes plots and reports.
- `results/` is created by the evaluator when needed; it contains regenerated
  evaluation outputs and is ignored by the repository's `.gitignore`.

## Prerequisites

Run the evaluator from the repository root, the directory containing `evaluation/`
and `src/`.

The root README documents the repository's setup:

```bash
python -m venv venv
```

Activate it with `venv\Scripts\activate` on Windows or
`source venv/bin/activate` on macOS/Linux, then install the declared dependencies:

```bash
pip install -r requirements.txt
```

The evaluation imports NumPy, pandas, scikit-learn, matplotlib, and
Sentence Transformers. The first semantic run loads the configured Sentence
Transformer model and may download it through the model library. The evaluator's
embedding code uses `paraphrase-multilingual-MiniLM-L12-v2` by default and accepts
the `SEMANTIC_PLAGIARISM_MODEL` environment variable to select another model.

There is no evaluation-specific API key or external service configured in
`evaluate.py`. A working model download/cache and the Python dependencies are
required. The evaluator's status text identifies the model as
`all-MiniLM-L6-v2`, but the actual loader default is
`paraphrase-multilingual-MiniLM-L12-v2`; when reporting results, record the model
that was actually configured.

## Benchmark Dataset Format

### Dataset-Level Structure

The JSON root is an object with these fields:

| Field          | Type   |   Required by current file | Description                                  |
| -------------- | ------ | -------------------------: | -------------------------------------------- |
| `name`         | string | Yes in the current dataset | Human-readable benchmark name.               |
| `version`      | string | Yes in the current dataset | Dataset version string.                      |
| `description`  | string | Yes in the current dataset | Summary of the benchmark contents.           |
| `label_schema` | object | Yes in the current dataset | Maps each label string to its meaning.       |
| `pairs`        | array  |                        Yes | Text-pair records consumed by `evaluate.py`. |

The parser does not validate the metadata fields. Evaluation requires the root
object to contain `pairs`; the current evaluator then reads the fields described
below from every record.

### Record Schema

Each `pairs` item is an object. A minimal record with the current field names is:

```json
{
  "id": "HP-EXAMPLE",
  "category": "heavy_paraphrase",
  "label": "plagiarized",
  "text_a": "A short passage describing a process.",
  "text_b": "A short restatement of the same process.",
  "notes": "Why this pair belongs in the benchmark."
}
```

### Field Reference

| Field      | Type   |                           Required for evaluation | Description                                           |
| ---------- | ------ | ------------------------------------------------: | ----------------------------------------------------- |
| `id`       | string |                       Yes for the per-pair report | Unique record identifier, such as `HP-01`.            |
| `category` | string |                       Yes for the per-pair report | Current dataset category, such as `heavy_paraphrase`. |
| `label`    | string |                                               Yes | `plagiarized` or `not_plagiarized`.                   |
| `text_a`   | string |                                               Yes | First passage in the pair.                            |
| `text_b`   | string |                                               Yes | Second passage in the pair.                           |
| `notes`    | string | Present in current records; not read by evaluator | Human explanation of the example.                     |

The evaluator maps exactly `"plagiarized"` to positive class `1`. Any other label
value is treated as negative class `0` by the current code, so contributors must
use the repository's two label values rather than inventing alternatives.

The current `label_schema` defines `plagiarized` as two texts conveying the same
core ideas or arguments where one is derived from the other. `not_plagiarized`
means independently authored texts, even when they share a broad topic. Existing
categories are `heavy_paraphrase`, `light_paraphrase`, `same_topic_original`, and
`different_topic`; the evaluator does not enforce an allowed category list.

### Positive and Negative Examples

Positive (`plagiarized`) records in the current benchmark include heavy paraphrases,
light paraphrases, restructured sentences, and synonym substitutions that preserve
the same core ideas. Negative (`not_plagiarized`) records include same-topic texts
with different claims and passages from different topics.

### Validation Rules

There is no separate schema validator in the repository. Before running evaluation,
ensure that:

- The file is valid UTF-8 JSON and the root contains a `pairs` array.
- Every pair has string `id`, `category`, `label`, `text_a`, and `text_b` values.
- Labels are exactly `plagiarized` or `not_plagiarized`.
- IDs are unique and text fields are non-empty.
- The JSON structure remains an object containing the existing dataset metadata and
  records.

The evaluator opens the file as UTF-8 and assumes these fields are present; it does
not provide friendly validation errors for missing or malformed record fields.

## Adding New Benchmark Cases

### Step-by-Step Process

1. Open [`benchmark_dataset.json`](./benchmark_dataset.json) and review existing
   records and categories.
2. Create a unique string `id` following the existing naming style where practical.
3. Add one object to `pairs` using the exact schema.
4. Choose `plagiarized` or `not_plagiarized` based on the benchmark definitions.
5. Write a concise `notes` explanation for human reviewers.
6. Check the JSON syntax:

   ```bash
   python -m json.tool evaluation/benchmark_dataset.json
   ```

7. Run the evaluator and review the metric changes and per-pair scores.
8. Check that the new ID and pair are not unintended duplicates.

### Writing Good Positive Pairs

Use purpose-built examples where the second passage preserves the first passage's
central ideas or argument. Useful cases include close paraphrases, sentence
reordering, synonym substitutions, and condensed or expanded restatements. Vary
the difficulty: the current benchmark contains both heavy and light paraphrases.

### Writing Good Negative Pairs

Use independently authored passages that do not preserve the same claim. Good
negatives can share a general topic or common terminology, but should differ in
their actual meaning. Contradictory claims and unrelated passages with incidental
word overlap are also useful negatives.

The current benchmark does not define a separate code-plagiarism or multilingual
evaluation category. Do not add such examples as if they were an established
benchmark capability without maintainer agreement.

### Avoiding Dataset Leakage

Keep benchmark text short, purpose-built, and manually reviewable. Avoid personally
identifiable or sensitive information, long copyrighted passages, and known copies
of model training or demo data. Do not add identical duplicates unless the duplicate
tests a documented edge case. The repository currently has one benchmark file and
no separate development/final split; if splits are introduced, keep the same pair
out of both.

### Dataset Quality Checklist

- Include difficult positives and difficult negatives, not only obvious cases.
- Keep labels balanced where practical, while preserving meaningful coverage.
- Make every label defensible from the text and `notes`.
- Use unique IDs and avoid duplicate pairs.
- Re-run the evaluator after adding records and inspect false positives and false
  negatives.

## Running the Evaluation

### Basic Command

From the repository root:

```bash
python -m evaluation.evaluate
```

The script always loads `evaluation/benchmark_dataset.json`; it does not accept a
dataset path argument. It embeds both texts in every pair, computes semantic
cosine similarities, computes the TF-IDF baseline, computes the 0.7-weight hybrid
score, sweeps thresholds, reports the best-F1 row for each approach, prints
per-pair scores, and saves plots and reports under `evaluation/results/`.

### Available CLI Options

`evaluate.py` defines no command-line argument parser. Therefore there are no
supported `--threshold`, `--dataset`, or `--threshold-sweep` options. Thresholds,
the dataset path, and the hybrid weight are constants or function arguments in the
source, not CLI settings.

### Example Commands

```bash
# Run the complete evaluation, including its built-in sweep.
python -m evaluation.evaluate

# Validate the dataset before running it.
python -m json.tool evaluation/benchmark_dataset.json
```

## Threshold Sweep Evaluation

### Why Threshold Selection Matters

The model produces a similarity score. A pair is predicted as plagiarism when its
score is greater than or equal to the selected threshold. Lower thresholds usually
catch more positive pairs (higher recall) but can flag more negative pairs (lower
precision); higher thresholds usually do the reverse. The useful operating point
depends on whether a product prioritizes finding more possible plagiarism or
reducing false alarms.

### Running a Threshold Sweep

There is no separate sweep command. The basic command runs it automatically:

```bash
python -m evaluation.evaluate
```

For semantic, TF-IDF, and hybrid scores, `sweep_thresholds` evaluates thresholds
from `0.30` up to but not including `0.96`, using a `0.01` step. The generated
threshold values are rounded to three decimals. At each threshold it calculates
precision, recall, F1, accuracy, and TP/FP/FN/TN. The evaluator selects the row
with maximum F1 independently for each approach; this is a benchmark summary, not
a universal production threshold recommendation.

### Reading the Threshold Results

The three CSV files contain one row per threshold. Compare the precision and recall
columns to see the trade-off, and use `f1` to locate the row selected by the
evaluator. The plots generated by this implementation are ROC, precision-recall,
and similarity-distribution plots; it does not generate a separate precision/
recall/F1-versus-threshold curve.

## Understanding the Metrics

### True Positives, False Positives, True Negatives, False Negatives

At a chosen threshold:

- **True positive (TP):** a `plagiarized` pair correctly flagged.
- **False positive (FP):** a `not_plagiarized` pair incorrectly flagged.
- **True negative (TN):** a `not_plagiarized` pair correctly left unflagged.
- **False negative (FN):** a `plagiarized` pair missed by the detector.

The console confusion-matrix summary prints these four counts for the best-F1
semantic, TF-IDF, and hybrid rows. `metrics.json` stores the summary metrics but
does not store the confusion-matrix counts.

### Precision

`precision = TP / (TP + FP)`

Precision answers: of everything flagged as plagiarism, how much was actually
plagiarism? The implementation returns `1.0` when no pair is predicted positive.

### Recall

`recall = TP / (TP + FN)`

Recall answers: of all benchmark plagiarism cases, how many did the detector catch?
The implementation returns `0.0` when there are no positive cases.

### F1 Score

The evaluator generates F1:

`F1 = 2 × (precision × recall) / (precision + recall)`

It balances precision and recall and is the criterion used to choose each approach's
reported best threshold. It is `0.0` when both precision and recall are zero.

### Accuracy

`accuracy = (TP + TN) / (TP + FP + FN + TN)`

Accuracy is the fraction of all pairs classified correctly. It can be misleading
when positive and negative classes are imbalanced, so read it together with
precision and recall.

### ROC-AUC

The evaluator computes ROC-AUC from continuous scores for the semantic, TF-IDF,
and hybrid approaches. A ROC curve compares true-positive rate with false-positive
rate across thresholds; AUC summarizes score-ranking separation across those
thresholds. Higher AUC indicates better separation on this benchmark, but ROC-AUC
does not select the production threshold for you.

### Confusion Matrix

The terminal output prints `TP`, `FP`, `FN`, and `TN` at each approach's best-F1
threshold. There is no separate confusion-matrix image or CSV output.

### Threshold Curves

No threshold curves are plotted by the current script. Use the three threshold-sweep
CSVs to construct one if needed: plot `threshold` on the x-axis and one or more of
`precision`, `recall`, or `f1` on the y-axis. The CSVs also include `accuracy` and
the four confusion counts.

## Generated Output

Every run creates `evaluation/results/` if it does not exist and overwrites these
files:

| File                           | Format | Meaning                                                                                                                                              |
| ------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `metrics.json`                 | JSON   | Dataset counts; ROC-AUC, best threshold, best F1, precision, recall, and accuracy for semantic, TF-IDF, and hybrid approaches; plus per-pair scores. |
| `threshold_sweep_semantic.csv` | CSV    | Threshold metrics for semantic scores.                                                                                                               |
| `threshold_sweep_tfidf.csv`    | CSV    | Threshold metrics for TF-IDF scores.                                                                                                                 |
| `threshold_sweep_hybrid.csv`   | CSV    | Threshold metrics for hybrid scores.                                                                                                                 |
| `roc_curve.png`                | PNG    | ROC curves comparing Sentence Transformers and TF-IDF.                                                                                               |
| `pr_curve.png`                 | PNG    | Precision-recall curves comparing Sentence Transformers and TF-IDF.                                                                                  |
| `similarity_distribution.png`  | PNG    | Semantic and TF-IDF score histograms split by label.                                                                                                 |

The console additionally prints the overall report, confusion counts, and one row
of semantic/TF-IDF/hybrid scores for every pair. The output directory is ignored by
Git, so do not commit regenerated files unless a maintainer specifically requests it.

## Example Workflow

```bash
git switch -c add-benchmark-cases

# Edit evaluation/benchmark_dataset.json and add two or three reviewed records.
python -m json.tool evaluation/benchmark_dataset.json
python -m evaluation.evaluate

# The same command performs the built-in threshold sweep.
python -m evaluation.evaluate

git diff -- evaluation/benchmark_dataset.json evaluation/README.md
git status --short
git add evaluation/benchmark_dataset.json evaluation/README.md
git commit -m "docs: add evaluation contributor guide"
```

Before committing, inspect false positives and false negatives in the per-pair
scores and compare the metric changes with the intended labels. The command above
is intentionally repeated only to make the sweep step explicit; one run already
performs it.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment, install `requirements.txt`, and run the command
from the repository root. The evaluator imports Sentence Transformers and the
scientific Python/plotting dependencies even though the dataset itself is JSON.

### Dataset file not found

Run `python -m evaluation.evaluate` from the repository root. The script resolves
the dataset next to `evaluate.py` as `evaluation/benchmark_dataset.json`; it does
not accept an alternate path.

### Invalid JSON

Check for a missing comma, trailing comma, unescaped quotation mark, or mismatched
brackets. Validate with:

```bash
python -m json.tool evaluation/benchmark_dataset.json
```

### Missing model or configuration

The first run may need to download the default
`paraphrase-multilingual-MiniLM-L12-v2` model. Check network access and the local
model cache. If `SEMANTIC_PLAGIARISM_MODEL` is set, confirm that it names a model
available to Sentence Transformers. No evaluation API key is read by this script.

### ROC-AUC cannot be calculated

ROC-AUC needs benchmark labels from both classes. Confirm that the dataset contains
at least one `plagiarized` and one `not_plagiarized` record and that labels use the
exact strings expected by the evaluator.

### All predictions belong to one class

Inspect the similarity scores and selected threshold. A threshold that is too low
or too high can produce only positive or only negative predictions. Also check that
the texts are non-empty and that the configured model is the intended one.

### Unexpected metric changes

Check labels, duplicate examples, class balance, the configured model, and the
selected threshold. The evaluator has no random-seed option, but changing the model
or benchmark contents changes the scores and metrics.

## Contribution Checklist

- [ ] I followed the documented dataset schema.
- [ ] Every new record has the correct `plagiarized` or `not_plagiarized` label.
- [ ] I used a unique ID and avoided unintended duplicate pairs.
- [ ] The JSON file parses successfully.
- [ ] `python -m evaluation.evaluate` runs successfully.
- [ ] I reviewed precision, recall, F1, and accuracy changes.
- [ ] I reviewed the built-in threshold-sweep CSV results.
- [ ] I did not commit generated temporary files unless required.
