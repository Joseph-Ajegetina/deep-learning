"""
Metrics for speech recognition evaluation.

Implements Word Error Rate (WER) and Character Error Rate (CER)
which are standard metrics for speech recognition systems.
"""

import numpy as np
from typing import List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def levenshtein_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Calculate Levenshtein distance between reference and hypothesis sequences.

    This dynamic programming algorithm computes the minimum number of
    substitutions, deletions, and insertions needed to transform the
    hypothesis into the reference.

    Args:
        ref: Reference sequence (ground truth)
        hyp: Hypothesis sequence (prediction)

    Returns:
        Tuple of (distance, substitutions, deletions, insertions)
    """
    len_ref = len(ref)
    len_hyp = len(hyp)

    # Create DP matrix
    # dp[i][j] represents the edit distance between ref[:i] and hyp[:j]
    dp = np.zeros((len_ref + 1, len_hyp + 1), dtype=np.int32)

    # Track operations
    ops = np.zeros((len_ref + 1, len_hyp + 1, 3), dtype=np.int32)  # [subs, dels, ins]

    # Initialize base cases
    for i in range(1, len_ref + 1):
        dp[i][0] = i
        ops[i][0][1] = i  # All deletions

    for j in range(1, len_hyp + 1):
        dp[0][j] = j
        ops[0][j][2] = j  # All insertions

    # Fill DP matrix
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref[i-1] == hyp[j-1]:
                # Match - no operation needed
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = ops[i-1][j-1].copy()
            else:
                # Three options: substitute, delete, insert
                substitute_cost = dp[i-1][j-1] + 1
                delete_cost = dp[i-1][j] + 1
                insert_cost = dp[i][j-1] + 1

                min_cost = min(substitute_cost, delete_cost, insert_cost)
                dp[i][j] = min_cost

                if min_cost == substitute_cost:
                    ops[i][j] = ops[i-1][j-1].copy()
                    ops[i][j][0] += 1  # Substitution
                elif min_cost == delete_cost:
                    ops[i][j] = ops[i-1][j].copy()
                    ops[i][j][1] += 1  # Deletion
                else:
                    ops[i][j] = ops[i][j-1].copy()
                    ops[i][j][2] += 1  # Insertion

    distance = dp[len_ref][len_hyp]
    substitutions, deletions, insertions = ops[len_ref][len_hyp]

    return int(distance), int(substitutions), int(deletions), int(insertions)


def word_error_rate(reference: Union[str, List[str]], hypothesis: Union[str, List[str]]) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Number of words in reference

    This is the primary metric for speech recognition evaluation.

    Args:
        reference: Ground truth transcription (string or list of words)
        hypothesis: Predicted transcription (string or list of words)

    Returns:
        WER as a percentage (0-100)
    """
    # Convert to word lists if strings
    if isinstance(reference, str):
        ref_words = reference.split()
    else:
        ref_words = reference

    if isinstance(hypothesis, str):
        hyp_words = hypothesis.split()
    else:
        hyp_words = hypothesis

    # Handle empty reference
    if len(ref_words) == 0:
        return 100.0 if len(hyp_words) > 0 else 0.0

    # Calculate edit distance
    distance, subs, dels, ins = levenshtein_distance(ref_words, hyp_words)

    # WER = total errors / reference length
    wer = (distance / len(ref_words)) * 100.0

    return wer


def character_error_rate(reference: Union[str, List[str]], hypothesis: Union[str, List[str]]) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = (Substitutions + Deletions + Insertions) / Number of characters in reference

    Useful for evaluating at character level, especially for languages without
    clear word boundaries or when word-level errors are too coarse.

    Args:
        reference: Ground truth transcription (string or list of characters)
        hypothesis: Predicted transcription (string or list of characters)

    Returns:
        CER as a percentage (0-100)
    """
    # Convert to character lists if strings
    if isinstance(reference, str):
        ref_chars = list(reference)
    else:
        ref_chars = reference

    if isinstance(hypothesis, str):
        hyp_chars = list(hypothesis)
    else:
        hyp_chars = hypothesis

    # Handle empty reference
    if len(ref_chars) == 0:
        return 100.0 if len(hyp_chars) > 0 else 0.0

    # Calculate edit distance
    distance, subs, dels, ins = levenshtein_distance(ref_chars, hyp_chars)

    # CER = total errors / reference length
    cer = (distance / len(ref_chars)) * 100.0

    return cer


def batch_wer(references: List[str], hypotheses: List[str]) -> Tuple[float, List[float]]:
    """
    Calculate WER for a batch of samples.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions

    Returns:
        Tuple of (average WER, list of individual WERs)
    """
    assert len(references) == len(hypotheses), "References and hypotheses must have same length"

    wers = []
    for ref, hyp in zip(references, hypotheses):
        wer = word_error_rate(ref, hyp)
        wers.append(wer)

    avg_wer = np.mean(wers) if wers else 0.0

    return avg_wer, wers


def batch_cer(references: List[str], hypotheses: List[str]) -> Tuple[float, List[float]]:
    """
    Calculate CER for a batch of samples.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions

    Returns:
        Tuple of (average CER, list of individual CERs)
    """
    assert len(references) == len(hypotheses), "References and hypotheses must have same length"

    cers = []
    for ref, hyp in zip(references, hypotheses):
        cer = character_error_rate(ref, hyp)
        cers.append(cer)

    avg_cer = np.mean(cers) if cers else 0.0

    return avg_cer, cers


def detailed_error_analysis(reference: str, hypothesis: str) -> dict:
    """
    Perform detailed error analysis between reference and hypothesis.

    Args:
        reference: Ground truth transcription
        hypothesis: Predicted transcription

    Returns:
        Dictionary containing detailed error statistics
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    distance, subs, dels, ins = levenshtein_distance(ref_words, hyp_words)

    wer = word_error_rate(reference, hypothesis)
    cer = character_error_rate(reference, hypothesis)

    return {
        'word_error_rate': wer,
        'character_error_rate': cer,
        'edit_distance': distance,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'reference_length': len(ref_words),
        'hypothesis_length': len(hyp_words),
        'reference': reference,
        'hypothesis': hypothesis
    }


def accuracy_from_wer(wer: float) -> float:
    """
    Convert WER to accuracy percentage.

    Args:
        wer: Word Error Rate (0-100)

    Returns:
        Accuracy as percentage (0-100)
    """
    return max(0.0, 100.0 - wer)


# For compatibility with jiwer library if needed
try:
    import jiwer
    HAS_JIWER = True
    logger.info("jiwer library available for WER calculation")
except ImportError:
    HAS_JIWER = False
    logger.info("jiwer library not available, using custom implementation")


def calculate_wer_jiwer(references: List[str], hypotheses: List[str]) -> float:
    """
    Calculate WER using jiwer library if available.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions

    Returns:
        WER as percentage
    """
    if not HAS_JIWER:
        # Fall back to custom implementation
        avg_wer, _ = batch_wer(references, hypotheses)
        return avg_wer

    wer = jiwer.wer(references, hypotheses)
    return wer * 100.0  # Convert to percentage
