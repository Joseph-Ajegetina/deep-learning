"""
Utilities module for speech recognition.

Contains:
- Tokenizer: Text tokenization and vocabulary management
- Metrics: WER/CER calculation
- Setup: Environment setup and configuration
"""

from .tokenizer import Tokenizer
from .metrics import (
    word_error_rate,
    character_error_rate,
    batch_wer,
    batch_cer,
    levenshtein_distance,
    detailed_error_analysis
)
from .setup import (
    setup_env,
    set_seed,
    get_device,
    load_config,
    verify_setup
)

__all__ = [
    "Tokenizer",
    "word_error_rate",
    "character_error_rate",
    "batch_wer",
    "batch_cer",
    "levenshtein_distance",
    "detailed_error_analysis",
    "setup_env",
    "set_seed",
    "get_device",
    "load_config",
    "verify_setup",
]
