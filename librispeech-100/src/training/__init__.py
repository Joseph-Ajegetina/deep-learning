"""
Training module for speech recognition models.

Contains:
- SequenceTrainer: Training pipeline with teacher forcing, gradient clipping
- SequenceEvaluator: Comprehensive evaluation with WER/CER and attention visualization
"""

from .trainer import SequenceTrainer
from .evaluator import SequenceEvaluator

__all__ = [
    "SequenceTrainer",
    "SequenceEvaluator",
]
