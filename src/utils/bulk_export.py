import io
import json
import logging
import zipfile
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from src.core.similarity import find_most_similar_chunks
from src.utils.pdf_report import generate_plagiarism_report

logger = logging.getLogger(__name__)


def _sanitise_filename(name: str) -> str:
    """Strip non-alphanumeric characters (except ``-``, ``_``) for safe filenames."""
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).rstrip() or "unnamed"


def generate_bulk_reports_zip(
    flags: List[Dict],
    chunked_docs: Optional[Dict[str, List[str]]] = None,
    embeddings: Optional[Dict[str, "np.ndarray"]] = None,
) -> bytes:
    """Generate a ZIP file containing PDF reports for all flagged pairs.

    When *chunked_docs* and *embeddings* are provided, each PDF includes
    the top-3 most similar paragraph pairs with side-by-side comparison.
    Otherwise a simplified report is generated with the available metadata.

    Parameters
    ----------
    flags:
        List of flag dicts returned by :func:`~src.core.similarity.flag_plagiarism`.
        Each dict must contain ``doc_a``, ``doc_b``, ``similarity``, and
        ``threshold_at_time_of_flag``.
    chunked_docs:
        Optional mapping of document name → list of text chunks.
    embeddings:
        Optional mapping of document name → NumPy embedding array.

    Returns
    -------
    bytes
        In-memory ZIP file contents.
    """
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, flag in enumerate(flags):
            doc_a = flag.get("doc_a", f"doc_A_{idx}")
            doc_b = flag.get("doc_b", f"doc_B_{idx}")
            score = float(flag.get("similarity", 0.0))
            threshold = float(flag.get("threshold_at_time_of_flag", 0.5))

            # Attempt to enrich the report with top matching chunk pairs
            top_pairs = []
            if chunked_docs and embeddings and doc_a in chunked_docs and doc_b in chunked_docs:
                try:
                    emb_a = embeddings[doc_a]
                    emb_b = embeddings[doc_b]
                    top_pairs = find_most_similar_chunks(
                        chunked_docs[doc_a],
                        chunked_docs[doc_b],
                        emb_a,
                        emb_b,
                        top_k=3,
                        threshold=threshold,
                    )
                except Exception as exc:
                    logger.debug("Could not compute chunk pairs for %s ↔ %s: %s", doc_a, doc_b, exc)

            try:
                pdf_buffer = generate_plagiarism_report(
                    doc_a=doc_a,
                    doc_b=doc_b,
                    overall_similarity=score,
                    threshold=threshold,
                    top_pairs=top_pairs,
                    report_title=f"Plagiarism Report: {doc_a} vs {doc_b}",
                )
                safe_a = _sanitise_filename(doc_a)
                safe_b = _sanitise_filename(doc_b)
                pdf_filename = f"report_{safe_a}_{safe_b}.pdf"
                zf.writestr(pdf_filename, pdf_buffer.getvalue())
            except Exception as exc:
                logger.error("Failed to generate PDF for %s ↔ %s: %s", doc_a, doc_b, exc)
                # Fallback: include a JSON report if PDF generation fails
                safe_a = _sanitise_filename(doc_a)
                safe_b = _sanitise_filename(doc_b)
                fallback = {
                    "generated_at": datetime.now().isoformat(),
                    "document_a": doc_a,
                    "document_b": doc_b,
                    "similarity_score": score,
                    "threshold": threshold,
                    "note": "PDF generation failed; JSON fallback provided.",
                }
                zf.writestr(
                    f"report_{safe_a}_{safe_b}.json",
                    json.dumps(fallback, indent=2),
                )

    return memory_file.getvalue()
