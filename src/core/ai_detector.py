"""
src/core/ai_detector.py
-----------------------
AI content detection module using transformer models.

Provides probability scoring for AI-generated text using pre-trained
transformer classifiers, with supplemental perplexity analysis for
explicit perplexity ratings that complement the probability scores.

Recent Additions (Issue #1154):
- Added `calculate_text_perplexity` helper function that provides explicit
  perplexity ratings. Lower perplexity scores indicate potential AI-generated
  text, as AI models tend to produce more predictable, lower-perplexity text.
"""

# pylint: disable=streamlit-global-mutation

import logging
import os
import re
import math
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

_DEFAULT_MODEL = "roberta-base-openai-detector"

# Default perplexity value returned when text cannot be analyzed.
# A score of 0.0 indicates the function could not compute a valid perplexity.
_DEFAULT_PERPLEXITY_SCORE = 0.0


def _get_model_name() -> str:
    """Return the configured AI detection model name."""
    return os.getenv("AI_DETECTION_MODEL", _DEFAULT_MODEL)


def _get_model_and_tokenizer():
    """Load and cache the transformer model and tokenizer."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        model_name = _get_model_name()
        logger.info(f"[ai_detector] Loading model: {model_name} …")

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)

            logger.info("[ai_detector] Model loaded successfully.")

        except Exception as err:
            logger.warning(
                "[ai_detector] Warning: Could not load transformer model "
                f"({err}). Using fallback mode."
            )

            _model = "fallback"
            _tokenizer = "fallback"

    return _model, _tokenizer


def calculate_text_perplexity(text: str) -> float:
    """Calculate the perplexity score of a given text string.

    Perplexity is an intrinsic measure of how well a language model predicts
    a sequence of tokens. Lower perplexity indicates the text is more
    predictable, which is characteristic of AI-generated content. Human
    written text tends to have higher perplexity due to more creative and
    less predictable word choices.

    This function uses the same transformer model and tokenizer as the
    probability detection functions. If the model fails to load, or if
    the input text is empty, a default perplexity score of 0.0 is returned.

    Args:
        text: Input text string to evaluate. Must be a non-empty string
              for a meaningful perplexity calculation.

    Returns:
        A numeric perplexity score (float). Lower values indicate higher
        predictability and potential AI-generated text. Returns 0.0 if
        the text is empty, None, or if the model cannot be loaded.

    Examples:
        >>> calculate_text_perplexity("The quick brown fox jumps over the lazy dog.")
        42.5
        >>> calculate_text_perplexity("")
        0.0
        >>> calculate_text_perplexity(None)
        0.0
    """
    # Validate input text: return default score for empty or None input
    if not text or not isinstance(text, str) or not text.strip():
        logger.debug(
            "[ai_detector] calculate_text_perplexity: empty or invalid input text, "
            "returning default perplexity score."
        )
        return float(_DEFAULT_PERPLEXITY_SCORE)

    try:
        model, tokenizer = _get_model_and_tokenizer()

        # If the model failed to load during initialization, we cannot
        # compute perplexity. Return the default score gracefully.
        if model == "fallback" or tokenizer == "fallback":
            logger.warning(
                "[ai_detector] calculate_text_perplexity: model is in fallback mode, "
                "cannot compute perplexity. Returning default score."
            )
            return float(_DEFAULT_PERPLEXITY_SCORE)

        # Tokenize the input text with truncation for long inputs.
        # The model's maximum sequence length is typically 512 tokens.
        max_length = getattr(model.config, "max_position_embeddings", 512)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Move tensors to the appropriate device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, "to"):
            model = model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Compute perplexity using cross-entropy loss over the token sequence.
        # We disable gradient computation for efficiency during inference.
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

            # The loss returned is the average cross-entropy loss per token.
            # Perplexity is the exponential of the cross-entropy loss.
            loss = outputs.loss

            if loss is None:
                logger.warning(
                    "[ai_detector] calculate_text_perplexity: model returned None loss, "
                    "falling back to default score."
                )
                return float(_DEFAULT_PERPLEXITY_SCORE)

            # Compute perplexity: exp(cross_entropy_loss)
            perplexity = math.exp(float(loss))

            # Clamp extremely large perplexity values to prevent overflow
            # and ensure the score remains within a reasonable range.
            # Most human text has perplexity between 20 and 200.
            perplexity = min(perplexity, 10000.0)
            perplexity = max(perplexity, 0.0)

            logger.debug(
                "[ai_detector] calculate_text_perplexity: computed perplexity=%.2f "
                "for text of length %d characters.",
                perplexity,
                len(text),
            )

            return float(perplexity)

    except ValueError as exc:
        # ValueError can occur if the input text is too short to tokenize
        # or contains only special tokens that produce an empty sequence.
        logger.warning(
            "[ai_detector] calculate_text_perplexity: ValueError during computation: %s. "
            "Returning default perplexity score.",
            exc,
        )
        return float(_DEFAULT_PERPLEXITY_SCORE)

    except Exception as err:
        # Catch any unexpected errors (e.g., CUDA OOM, tokenizer errors)
        # and return the default score rather than crashing the pipeline.
        logger.warning(
            "[ai_detector] calculate_text_perplexity: unexpected error: %s. "
            "Returning default perplexity score.",
            err,
        )
        return float(_DEFAULT_PERPLEXITY_SCORE)


def detect_ai_probability_batch(
    texts: List[str],
    batch_size: int = 8,
) -> List[float]:
    """
    Detect AI probability for multiple texts in batches.

    Args:
        texts: List of text strings to analyze.
        batch_size: Number of texts to process in each batch.

    Returns:
        List of probability scores between 0.0 and 1.0,
        corresponding to the input texts.
    """
    if not texts:
        return []

    # Filter out empty strings while tracking their original index mapping
    valid_texts = []
    valid_indices = []

    for index, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(index)

    if not valid_texts:
        return [0.0] * len(texts)

    try:
        model, tokenizer = _get_model_and_tokenizer()

        # Use fallback mode if transformer model failed to initialize
        if model == "fallback":
            return [0.0] * len(texts)
    except Exception:
        return [0.0] * len(texts)

    probabilities = [0.0] * len(texts)

    # Process valid texts in batches
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i : i + batch_size]
        batch_indices = valid_indices[i : i + batch_size]

        try:
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            if torch.cuda.is_available():
                if hasattr(model, "to"):
                    model = model.to("cuda")
                inputs = {key: val.to("cuda") for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            if isinstance(logits, torch.Tensor):
                probs = torch.softmax(logits, dim=-1)

                # Class 1 corresponds to AI/Fake
                if probs.shape[1] > 1:
                    batch_probs = probs[:, 1].tolist()
                else:
                    batch_probs = probs[:, 0].tolist()
            else:
                batch_probs = [0.5] * len(batch_texts)

            for index, probability in zip(batch_indices, batch_probs):
                probabilities[index] = float(probability)

        except Exception as err:
            logger.warning(
                f"[ai_detector] Warning: Failed to process batch "
                f"starting at index {i}: {err}"
            )
            for index in batch_indices:
                probabilities[index] = 0.0

    return probabilities


def detect_ai_probability(text: str) -> float:
    """
    Detect the probability that a given text was AI-generated.

    Args:
        text: Input text string to analyze.

    Returns:
        Probability score between 0.0 (human-written) and 1.0 (AI-generated).
    """
    if not text or not text.strip():
        return 0.0

    results = detect_ai_probability_batch([text])
    return results[0] if results else 0.0


def detect_document_ai_probability(chunks: List[str]) -> Dict[str, Any]:
    """
    Calculate AI-generated text statistics for a single document's chunks.

    Args:
        chunks: List of text chunks belonging to one document.

    Returns:
        Dictionary containing overall probability, maximum probability,
        and individual chunk scores.
    """
    if not chunks:
        return {
            "overall": 0.0,
            "max": 0.0,
            "chunk_scores": [],
        }

    chunk_scores = detect_ai_probability_batch(chunks)

    return {
        "overall": (float(np.mean(chunk_scores)) if chunk_scores else 0.0),
        "max": (float(np.max(chunk_scores)) if chunk_scores else 0.0),
        "chunk_scores": chunk_scores,
    }


def detect_documents_ai_probability(
    chunked_docs: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate AI-generated probabilities across multiple documents.

    Args:
        chunked_docs: Dictionary mapping document names to their text chunks.

    Returns:
        Dictionary containing AI detection results for each document.
    """
    results = {}

    for doc_name, chunks in chunked_docs.items():
        results[doc_name] = detect_document_ai_probability(chunks)

    return results


def _calculate_burstiness(text: str) -> float:
    """Calculate the burstiness score of a text.

    Burstiness measures the variation in sentence lengths. Human text tends
    to have higher burstiness (more varied sentence lengths), while AI text
    tends to be more uniform.

    Returns a float between 0.0 (uniform) and 1.0 (highly varied).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    # Split into sentences using a simple regex
    sentences = re.split(r'[.!?]+', text.strip())
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

    if len(sentence_lengths) < 2:
        return 0.0

    mean_len = np.mean(sentence_lengths)
    if mean_len == 0:
        return 0.0

    # Coefficient of variation = std / mean
    cv = float(np.std(sentence_lengths) / mean_len)
    # Normalize to [0, 1] — typical CV for human text is 0.3-0.8
    return min(cv / 1.0, 1.0)


def _calculate_ngram_repetitiveness(text: str, n: int = 3) -> float:
    """Calculate n-gram repetitiveness score.

    AI text often repeats certain n-gram patterns more than human text.
    Returns a float between 0.0 (no repetition) and 1.0 (highly repetitive).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    words = text.lower().split()
    if len(words) < n:
        return 0.0

    ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0

    unique_ngrams = set(ngrams)
    repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
    return float(repetition_ratio)

def detect_ai_generated_text(text: str) -> Dict[str, Any]:
    """
    Detect the likelihood that a given text was AI-generated using a
    multi-metric classifier (issue #1356).

    Combines three signals:
    1. Transformer-based AI probability (detect_ai_probability)
    2. Perplexity score (calculate_text_perplexity)
    3. Burstiness score (sentence-length variation)
    4. N-gram repetitiveness

    Confidence threshold tiers:
    - 'high': ai_probability >= 0.75
    - 'medium': 0.40 <= ai_probability < 0.75
    - 'low': ai_probability < 0.40

    Args:
        text: Input text string to analyze.

    Returns:
        A dictionary containing:
        - ai_probability (float): Probability score between 0.0 and 1.0.
        - confidence_tier (str): Categorized tier ('high', 'medium', 'low').
        - perplexity_score (float): Estimated perplexity score.
        - burstiness_score (float): Sentence-length variation score (0-1).
        - ngram_repetitiveness (float): N-gram repetition ratio (0-1).
        - classification_tier (str): Same as confidence_tier (for API compat).
    """
    if not text or not text.strip():
        return {
            "ai_probability": 0.0,
            "confidence_tier": "low",
            "classification_tier": "low",
            "perplexity_score": 150.0,
            "burstiness_score": 0.0,
            "ngram_repetitiveness": 0.0,
        }

    ai_probability = detect_ai_probability(text)
    perplexity_score = calculate_text_perplexity(text)
    burstiness_score = _calculate_burstiness(text)
    ngram_repetitiveness = _calculate_ngram_repetitiveness(text)

    if ai_probability >= 0.75:
        confidence_tier = "high"
    elif ai_probability >= 0.40:
        confidence_tier = "medium"
    else:
        confidence_tier = "low"

    return {
        "ai_probability": ai_probability,
        "confidence_tier": confidence_tier,
        "classification_tier": confidence_tier,
        "perplexity_score": perplexity_score,
        "burstiness_score": burstiness_score,
        "ngram_repetitiveness": ngram_repetitiveness,
    }

def categorize_ai_probability(score: float) -> str:
    """
    Map a raw AI probability score to a human-readable confidence category.

    Args:
        score: AI probability score between 0.0 and 1.0.

    Returns:
        "High Probability" for score >= 0.8,
        "Moderate Probability" for 0.5 <= score < 0.8,
        "Low Probability" for score < 0.5.
    """
    if score >= 0.8:
        return "High Probability"
    elif score >= 0.5:
        return "Moderate Probability"
    else:
        return "Low Probability"
