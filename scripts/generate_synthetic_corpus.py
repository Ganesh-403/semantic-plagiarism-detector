#!/usr/bin/env python3
from __future__ import annotations

"""
generate_synthetic_corpus.py
----------------------------
Automated synthetic corpus dataset generator for testing plagiarism
detection algorithms.

Generates a dataset of text documents with controlled similarity overlaps
by applying paragraph mutations, synonym substitutions, and word reordering
to a set of base original documents.

Usage:
    python scripts/generate_synthetic_corpus.py --num-docs 20 --plagiarism-ratio 0.4 --output-dir data/synthetic

Acceptance Criteria (Issue #1376):
- Create scripts/generate_synthetic_corpus.py.
- Support --num-docs=10, --plagiarism-ratio=0.3, and --output-dir=data/synthetic.
- Generate text files containing controlled paragraph mutations, synonym
  substitutions, and word reordering.
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for potential future imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Base Text Corpus ───────────────────────────────────────────────────────────
# A collection of original academic and technical paragraphs used as the
# foundation for generating synthetic documents.

_BASE_PARAGRAPHS = [
    "Plagiarism detection is a critical component of academic integrity. Modern systems utilize natural language processing and machine learning to identify semantic similarities between documents, going beyond simple exact string matching. This allows educators to detect paraphrased content and structural borrowing that traditional tools might miss.",
    "Vector embeddings represent text as high-dimensional numerical arrays. By mapping words and sentences into a continuous vector space, models can capture semantic meaning. Sentences with similar meanings will have vectors that are geometrically close to each other, typically measured using cosine similarity.",
    "The FAISS library, developed by Facebook AI Research, provides highly optimized routines for similarity search and clustering of dense vectors. It supports indexing millions of vectors and performing nearest-neighbor searches in milliseconds, making it ideal for large-scale plagiarism detection systems.",
    "Natural language processing has evolved significantly with the advent of transformer architectures. Models like BERT and its variants generate contextualized embeddings, where the representation of a word depends on its surrounding context. This contextual awareness dramatically improves the accuracy of semantic similarity tasks.",
    "Academic institutions face growing challenges with contract cheating and AI-generated text. While traditional plagiarism detectors compare student submissions against existing databases, detecting AI-generated content requires analyzing statistical properties of the text, such as perplexity and burstiness, which often differ from human writing patterns.",
    "Text chunking is a necessary preprocessing step when dealing with long documents. Since embedding models have maximum sequence length limitations, documents must be split into overlapping chunks. This ensures that context is preserved across chunk boundaries and no semantic information is lost during the embedding process.",
    "Cosine similarity measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. In text analysis, it is used to determine how similar two documents are irrespective of their length. A cosine similarity of 1 indicates identical semantic meaning, while 0 indicates orthogonality or no similarity.",
    "Database optimization is crucial for maintaining performance in production systems. SQLite databases can suffer from fragmentation over time as records are inserted and deleted. Running periodic VACUUM commands reclaims unused space and defragments the database file, ensuring consistent read and write performance.",
    "Webhook integrations allow systems to push real-time notifications to external services. When a high-severity plagiarism incident is detected, the system can automatically dispatch an alert to a Slack channel or learning management system, enabling instructors to take immediate action.",
    "Secure software development practices require careful validation of all external inputs. Server-Side Request Forgery (SSRF) vulnerabilities can occur if an application fetches a remote resource based on user-supplied URLs without proper validation. Implementing domain allowlists and blocking private IP ranges mitigates this risk.",
    "Machine learning models require careful management of computational resources. Processing large batches of documents can lead to memory exhaustion if not handled correctly. Implementing mini-batch processing and periodic garbage collection helps maintain stable memory usage during intensive embedding operations.",
    "The transition from lexical to semantic search represents a paradigm shift in information retrieval. While lexical search relies on keyword matching, semantic search understands the intent and contextual meaning behind queries. This results in more relevant search results, even when the exact keywords are not present in the target documents.",
]

# ── Synonym Dictionary ─────────────────────────────────────────────────────────
# A hardcoded dictionary of synonyms used for text mutation. This avoids
# introducing heavy NLP dependencies like WordNet or NLTK.

_SYNONYM_DICT = {
    "critical": ["crucial", "essential", "vital", "important"],
    "component": ["element", "part", "aspect", "feature"],
    "academic": ["scholarly", "educational", "institutional"],
    "integrity": ["honesty", "ethics", "probity"],
    "modern": ["contemporary", "current", "advanced", "recent"],
    "systems": ["platforms", "tools", "applications", "software"],
    "utilize": ["use", "employ", "leverage", "apply"],
    "natural": ["inherent", "innate"],
    "language": ["linguistic", "textual"],
    "processing": ["analysis", "computation", "handling"],
    "machine": ["automated", "computational"],
    "learning": ["training", "modeling"],
    "identify": ["detect", "recognize", "discover", "find"],
    "semantic": ["meaning-based", "conceptual"],
    "similarities": ["resemblances", "parallels", "overlaps", "likenesses"],
    "between": ["among", "across"],
    "documents": ["texts", "papers", "submissions", "files"],
    "beyond": ["past", "exceeding", "outside"],
    "simple": ["basic", "straightforward", "rudimentary"],
    "exact": ["precise", "strict", "literal"],
    "string": ["text", "character"],
    "matching": ["comparison", "alignment"],
    "allows": ["enables", "permits", "empowers"],
    "educators": ["teachers", "instructors", "professors", "faculty"],
    "detect": ["identify", "spot", "uncover", "find"],
    "paraphrased": ["reworded", "restated", "rephrased"],
    "content": ["material", "text", "information"],
    "structural": ["organizational", "architectural"],
    "borrowing": ["copying", "appropriation", "lifting"],
    "traditional": ["conventional", "standard", "classic"],
    "tools": ["utilities", "applications", "software"],
    "might": ["could", "may", "would"],
    "miss": ["overlook", "ignore", "fail to detect"],
    "vector": ["array", "tensor"],
    "embeddings": ["representations", "encodings", "vectors"],
    "represent": ["encode", "express", "capture"],
    "text": ["content", "words", "language"],
    "high-dimensional": ["multi-dimensional", "complex"],
    "numerical": ["quantitative", "mathematical"],
    "arrays": ["matrices", "structures"],
    "mapping": ["projecting", "transforming"],
    "words": ["terms", "vocabulary", "tokens"],
    "sentences": ["phrases", "statements", "utterances"],
    "continuous": ["unbroken", "smooth"],
    "space": ["domain", "field", "manifold"],
    "models": ["algorithms", "architectures", "networks"],
    "capture": ["extract", "encode", "represent"],
    "meaning": ["semantics", "concept", "sense"],
    "similar": ["alike", "comparable", "related"],
    "meanings": ["definitions", "concepts", "interpretations"],
    "vectors": ["arrays", "embeddings", "points"],
    "geometrically": ["spatially", "mathematically"],
    "close": ["near", "proximate", "adjacent"],
    "each": ["one another", "mutually"],
    "typically": ["usually", "generally", "commonly"],
    "measured": ["calculated", "computed", "evaluated"],
    "using": ["via", "with", "through"],
    "cosine": ["angular"],
    "similarity": ["resemblance", "likeness", "proximity"],
    "developed": ["created", "built", "engineered"],
    "provides": ["offers", "supplies", "delivers"],
    "highly": ["extremely", "very", "exceptionally"],
    "optimized": ["tuned", "efficient", "streamlined"],
    "routines": ["functions", "operations", "procedures"],
    "search": ["retrieval", "querying", "lookup"],
    "clustering": ["grouping", "categorization"],
    "dense": ["compact", "filled"],
    "supports": ["handles", "accommodates", "enables"],
    "indexing": ["cataloging", "organizing"],
    "millions": ["vast numbers", "multitudes"],
    "performing": ["executing", "running"],
    "nearest-neighbor": ["proximity", "closest-match"],
    "searches": ["queries", "lookups"],
    "milliseconds": ["fractions of a second", "brief moments"],
    "making": ["rendering", "causing"],
    "ideal": ["perfect", "optimal", "excellent"],
    "large-scale": ["massive", "extensive", "enterprise"],
    "evolved": ["advanced", "progressed", "developed"],
    "significantly": ["substantially", "greatly", "markedly"],
    "advent": ["introduction", "arrival", "emergence"],
    "transformer": ["attention-based"],
    "architectures": ["designs", "frameworks", "structures"],
    "generate": ["produce", "create", "yield"],
    "contextualized": ["context-aware", "situated"],
    "representation": ["encoding", "embedding"],
    "word": ["term", "token"],
    "depends": ["relies", "contingent"],
    "surrounding": ["adjacent", "neighboring", "contextual"],
    "context": ["environment", "setting", "background"],
    "contextual": ["situational", "environmental"],
    "awareness": ["understanding", "recognition"],
    "dramatically": ["significantly", "vastly", "greatly"],
    "improves": ["enhances", "boosts", "increases"],
    "accuracy": ["precision", "correctness", "exactness"],
    "tasks": ["operations", "jobs", "functions"],
    "institutions": ["organizations", "establishments", "universities"],
    "face": ["encounter", "confront", "experience"],
    "growing": ["increasing", "expanding", "rising"],
    "challenges": ["difficulties", "obstacles", "problems"],
    "contract": ["outsourced", "ghostwritten"],
    "cheating": ["fraud", "dishonesty", "deception"],
    "AI-generated": ["machine-written", "automated", "synthetic"],
    "compare": ["evaluate", "contrast", "match"],
    "student": ["learner", "pupil", "scholar"],
    "submissions": ["assignments", "papers", "entries"],
    "against": ["with", "versus"],
    "existing": ["current", "available", "established"],
    "databases": ["repositories", "archives", "records"],
    "detecting": ["identifying", "spotting", "uncovering"],
    "requires": ["needs", "demands", "necessitates"],
    "analyzing": ["examining", "evaluating", "inspecting"],
    "statistical": ["quantitative", "mathematical"],
    "properties": ["characteristics", "attributes", "features"],
    "perplexity": ["complexity", "unpredictability"],
    "burstiness": ["variability", "fluctuation"],
    "often": ["frequently", "commonly", "regularly"],
    "differ": ["vary", "deviate", "contrast"],
    "human": ["organic", "natural", "manual"],
    "writing": ["composition", "authorship", "drafting"],
    "patterns": ["trends", "structures", "styles"],
}


# ── Text Mutation Functions ────────────────────────────────────────────────────


def _tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using basic regex punctuation matching."""
    # Split on period, exclamation, or question mark followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _tokenize_words(sentence: str) -> List[str]:
    """Split a sentence into words while preserving punctuation."""
    # Find all words and punctuation marks
    tokens = re.findall(r"\b\w+\b|[^\w\s]", sentence)
    return tokens


def replace_synonyms(text: str, mutation_ratio: float = 0.3) -> str:
    """
    Replace words in the text with synonyms based on a predefined dictionary.

    Args:
        text: The input text to mutate.
        mutation_ratio: The probability (0.0 to 1.0) that an eligible word
                        will be replaced with a synonym.

    Returns:
        The mutated text with synonym substitutions.
    """
    sentences = _tokenize_sentences(text)
    mutated_sentences = []

    for sentence in sentences:
        tokens = _tokenize_words(sentence)
        new_tokens = []

        for token in tokens:
            lower_token = token.lower()
            # Check if token is in synonym dictionary and meets probability threshold
            if lower_token in _SYNONYM_DICT and random.random() < mutation_ratio:
                # Pick a random synonym
                synonym = random.choice(_SYNONYM_DICT[lower_token])
                # Preserve original capitalization
                if token.istitle():
                    synonym = synonym.capitalize()
                elif token.isupper():
                    synonym = synonym.upper()
                new_tokens.append(synonym)
            else:
                new_tokens.append(token)

        # Reconstruct sentence with proper spacing
        reconstructed = ""
        for i, token in enumerate(new_tokens):
            if i == 0:
                reconstructed += token
            elif re.match(r"^[^\w\s]$", token):
                reconstructed += token  # No space before punctuation
            else:
                reconstructed += " " + token

        mutated_sentences.append(reconstructed)

    return " ".join(mutated_sentences)


def reorder_words(text: str, mutation_ratio: float = 0.2) -> str:
    """
    Randomly reorder words within clauses or sentences to simulate paraphrasing.

    This function splits sentences into clauses (using commas and conjunctions)
    and randomly shuffles the order of clauses or words within them.

    Args:
        text: The input text to mutate.
        mutation_ratio: The probability that a sentence will undergo reordering.

    Returns:
        The mutated text with reordered words/clauses.
    """
    sentences = _tokenize_sentences(text)
    mutated_sentences = []

    for sentence in sentences:
        if random.random() < mutation_ratio:
            # Split by commas to get clauses
            clauses = [c.strip() for c in sentence.split(",") if c.strip()]

            if len(clauses) > 1:
                # Shuffle the clauses
                random.shuffle(clauses)
                mutated_sentence = ", ".join(clauses)

                # Ensure first letter is capitalized and ends with punctuation
                if mutated_sentence:
                    mutated_sentence = (
                        mutated_sentence[0].upper() + mutated_sentence[1:]
                    )
                    if not re.search(r"[.!?]$", mutated_sentence):
                        mutated_sentence += "."
                mutated_sentences.append(mutated_sentence)
            else:
                # If no clauses, shuffle words (excluding first and last for grammar)
                tokens = _tokenize_words(sentence)
                if len(tokens) > 4:
                    # Keep first word and last punctuation intact
                    first_word = tokens[0]
                    last_punct = tokens[-1] if re.match(r"[^\w\s]", tokens[-1]) else ""

                    middle_words = tokens[1:-1] if last_punct else tokens[1:]
                    random.shuffle(middle_words)

                    new_tokens = [first_word] + middle_words
                    if last_punct:
                        new_tokens.append(last_punct)

                    mutated_sentences.append(" ".join(new_tokens))
                else:
                    mutated_sentences.append(sentence)
        else:
            mutated_sentences.append(sentence)

    return " ".join(mutated_sentences)


def delete_sentences(text: str, mutation_ratio: float = 0.1) -> str:
    """
    Randomly delete sentences from the text to simulate summarization or omission.

    Args:
        text: The input text to mutate.
        mutation_ratio: The probability that any given sentence will be deleted.

    Returns:
        The mutated text with some sentences removed.
    """
    sentences = _tokenize_sentences(text)

    if len(sentences) <= 1:
        return text  # Don't delete if only one sentence

    kept_sentences = [s for s in sentences if random.random() >= mutation_ratio]

    # Ensure at least one sentence remains
    if not kept_sentences:
        kept_sentences = [random.choice(sentences)]

    return " ".join(kept_sentences)


def insert_noise(text: str, mutation_ratio: float = 0.1) -> str:
    """
    Insert random filler words or phrases to simulate human writing imperfections.

    Args:
        text: The input text to mutate.
        mutation_ratio: The probability of inserting noise into a sentence.

    Returns:
        The mutated text with inserted filler phrases.
    """
    filler_phrases = [
        "in other words,",
        "basically,",
        "to put it simply,",
        "essentially,",
        "as a matter of fact,",
        "it is worth noting that",
        "generally speaking,",
    ]

    sentences = _tokenize_sentences(text)
    mutated_sentences = []

    for sentence in sentences:
        if random.random() < mutation_ratio and len(sentence) > 20:
            filler = random.choice(filler_phrases)
            # Insert filler at the beginning or middle
            tokens = sentence.split()
            insert_pos = random.randint(0, min(3, len(tokens) - 1))
            tokens.insert(insert_pos, filler)
            mutated_sentences.append(" ".join(tokens))
        else:
            mutated_sentences.append(sentence)

    return " ".join(mutated_sentences)


def apply_mutations(text: str, intensity: float = 0.5) -> str:
    """
    Apply a combination of mutation strategies to a text.

    The intensity parameter controls the overall aggressiveness of the mutations.
    Higher intensity results in more synonym replacements, word reordering, and
    sentence deletions, creating a document that is semantically similar but
    lexically distinct from the original.

    Args:
        text: The original text to mutate.
        intensity: A value between 0.0 and 1.0 controlling mutation severity.

    Returns:
        The heavily mutated text.
    """
    # Scale mutation ratios based on intensity
    synonym_ratio = 0.1 + (intensity * 0.4)  # 0.1 to 0.5
    reorder_ratio = 0.05 + (intensity * 0.3)  # 0.05 to 0.35
    delete_ratio = intensity * 0.2  # 0.0 to 0.2
    noise_ratio = intensity * 0.15  # 0.0 to 0.15

    # Apply mutations in sequence
    mutated = replace_synonyms(text, mutation_ratio=synonym_ratio)
    mutated = reorder_words(mutated, mutation_ratio=reorder_ratio)
    mutated = insert_noise(mutated, mutation_ratio=noise_ratio)
    mutated = delete_sentences(mutated, mutation_ratio=delete_ratio)

    return mutated


# ── Corpus Generation Logic ────────────────────────────────────────────────────


def generate_corpus(
    num_docs: int,
    plagiarism_ratio: float,
    output_dir: Path,
) -> List[Dict[str, any]]:
    """
    Generate a synthetic corpus of original and plagiarized documents.

    Args:
        num_docs: Total number of documents to generate.
        plagiarism_ratio: Fraction of documents that should be plagiarized (0.0 to 1.0).
        output_dir: Directory where generated .txt files will be saved.

    Returns:
        A list of metadata dictionaries describing each generated document,
        including its source and mutation intensity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    num_plagiarized = int(num_docs * plagiarism_ratio)
    num_original = num_docs - num_plagiarized

    logger.info(
        f"Generating {num_original} original documents and {num_plagiarized} plagiarized documents."
    )

    # 1. Generate Original Documents
    original_docs = []
    for i in range(num_original):
        doc_id = f"orig_{i:04d}"
        filename = f"{doc_id}.txt"

        # Combine 2-4 random base paragraphs to form a complete document
        num_paragraphs = random.randint(2, 4)
        selected_paragraphs = random.choices(_BASE_PARAGRAPHS, k=num_paragraphs)
        doc_text = "\n\n".join(selected_paragraphs)

        filepath = output_dir / filename
        filepath.write_text(doc_text, encoding="utf-8")

        metadata = {
            "filename": filename,
            "type": "original",
            "source": None,
            "mutation_intensity": 0.0,
            "base_paragraphs": num_paragraphs,
        }
        manifest.append(metadata)
        original_docs.append((doc_id, doc_text))

    # 2. Generate Plagiarized Documents
    if not original_docs:
        logger.warning(
            "No original documents generated to plagiarize from. Using base paragraphs directly."
        )
        original_docs = [("base_0", random.choice(_BASE_PARAGRAPHS))]

    for i in range(num_plagiarized):
        doc_id = f"plag_{i:04d}"
        filename = f"{doc_id}.txt"

        # Pick a random original document to plagiarize
        source_id, source_text = random.choice(original_docs)

        # Determine mutation intensity (higher intensity = harder to detect)
        intensity = random.uniform(0.3, 0.8)

        # Apply mutations
        mutated_text = apply_mutations(source_text, intensity=intensity)

        filepath = output_dir / filename
        filepath.write_text(mutated_text, encoding="utf-8")

        metadata = {
            "filename": filename,
            "type": "plagiarized",
            "source": f"{source_id}.txt",
            "mutation_intensity": round(intensity, 2),
            "base_paragraphs": source_text.count("\n\n") + 1,
        }
        manifest.append(metadata)

    # 3. Save Manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Corpus generation complete. Manifest saved to {manifest_path}")
    return manifest


# ── CLI Argument Parsing ───────────────────────────────────────────────────────


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the corpus generator."""
    parser = argparse.ArgumentParser(
        description="Semantic Plagiarism Detection System - Synthetic Corpus Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Total number of documents to generate (original + plagiarized).",
    )
    parser.add_argument(
        "--plagiarism-ratio",
        type=float,
        default=0.3,
        help="Fraction of documents that should be plagiarized (0.0 to 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Directory where generated text files and manifest will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible corpus generation.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_docs <= 0:
        parser.error("--num-docs must be greater than 0.")
    if not (0.0 <= args.plagiarism_ratio <= 1.0):
        parser.error("--plagiarism-ratio must be between 0.0 and 1.0.")

    return args


# ── Main Execution ─────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for the synthetic corpus generator."""
    args = parse_arguments()

    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed} for reproducibility.")

    output_path = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("Synthetic Corpus Generator")
    logger.info("=" * 70)
    logger.info(
        f"Parameters: num_docs={args.num_docs}, plagiarism_ratio={args.plagiarism_ratio}"
    )
    logger.info(f"Output directory: {output_path.resolve()}")

    generate_corpus(
        num_docs=args.num_docs,
        plagiarism_ratio=args.plagiarism_ratio,
        output_dir=output_path,
    )

    logger.info("Execution complete.")


if __name__ == "__main__":
    main()
