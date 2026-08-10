"""examples/basic_plagiarism_check.py

A minimal quickstart example showing how to perform a semantic plagiarism
check between documents using the core NLP components.
"""

import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embedding_model import embed_chunks, get_document_embedding
from src.core.similarity import PLAGIARISM_THRESHOLD, chunk_max_similarity
from src.core.text_chunking import chunk_document


def main():
    # 1. Define sample documents
    doc_a = (
        "Artificial Intelligence (AI) and Machine Learning (ML) are rapidly "
        "transforming higher education. Universities are adopting AI tools "
        "to assist with automated grading, personalized feedback, and course design."
    )

    # Paraphrased version of Doc A
    doc_b = (
        "AI and machine learning technologies are quickly reshaping the college experience. "
        "Higher learning institutes leverage smart tools for grading, custom feedback, "
        "and developing curriculum."
    )

    # Unrelated document
    doc_c = (
        "Photosynthesis is a process used by plants and other organisms to "
        "convert light energy into chemical energy that can later be released "
        "to fuel the organisms' activities."
    )

    print("--- Document Definitions ---")
    print(f"Document A (Original):\n  '{doc_a}'\n")
    print(f"Document B (Paraphrased):\n  '{doc_b}'\n")
    print(f"Document C (Unrelated):\n  '{doc_c}'\n")

    # 2. Split documents into paragraph-level chunks
    print("--- Chunking Documents ---")
    chunks_a = chunk_document(doc_a)
    chunks_b = chunk_document(doc_b)
    chunks_c = chunk_document(doc_c)
    print(f"Doc A chunks: {len(chunks_a)}")
    print(f"Doc B chunks: {len(chunks_b)}")
    print(f"Doc C chunks: {len(chunks_c)}")

    # 3. Generate embeddings
    print("\n--- Generating Embeddings ---")
    emb_a = embed_chunks(chunks_a)
    emb_b = embed_chunks(chunks_b)
    emb_c = embed_chunks(chunks_c)

    # 4. Compute Document-level Similarity (mean pooled representation)
    print("\n--- Computing Document-level Similarity ---")
    doc_emb_a = get_document_embedding(emb_a).reshape(1, -1)
    doc_emb_b = get_document_embedding(emb_b).reshape(1, -1)
    doc_emb_c = get_document_embedding(emb_c).reshape(1, -1)

    sim_ab_doc = float(cosine_similarity(doc_emb_a, doc_emb_b)[0, 0])
    sim_ac_doc = float(cosine_similarity(doc_emb_a, doc_emb_c)[0, 0])

    print(f"Overall similarity (A <-> B): {sim_ab_doc:.4f}")
    print(f"Overall similarity (A <-> C): {sim_ac_doc:.4f}")

    # 5. Compute Chunk-level Max Similarity (local similarity check)
    print("\n--- Computing Chunk-level Max Similarity ---")
    sim_ab_chunk = chunk_max_similarity(emb_a, emb_b)
    sim_ac_chunk = chunk_max_similarity(emb_a, emb_c)

    print(f"Max chunk-level similarity (A <-> B): {sim_ab_chunk:.4f}")
    print(f"Max chunk-level similarity (A <-> C): {sim_ac_chunk:.4f}")

    # 6. Evaluate against Plagiarism Threshold
    print(f"\nPlagiarism Threshold is set to: {PLAGIARISM_THRESHOLD}")
    for label, score in [("A & B", sim_ab_chunk), ("A & C", sim_ac_chunk)]:
        flagged = score >= PLAGIARISM_THRESHOLD
        status = "🔴 PLAGIARISM FLAGGED" if flagged else "🟢 PASS (Clean)"
        print(f"Pair {label}: {status} (score: {score:.4f})")
