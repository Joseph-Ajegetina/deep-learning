import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Tokenizer for text sequences in speech recognition.

    Supports character-level, word-level, or subword tokenization.
    Handles special tokens for padding, start-of-sequence, end-of-sequence, and unknown tokens.
    """

    def __init__(
        self,
        tokenization_type: str = "character",  # "character", "word", or "bpe"
        vocab_size: Optional[int] = None,
        lowercase: bool = True
    ):
        """
        Initialize the tokenizer.

        Args:
            tokenization_type: Type of tokenization ("character", "word", "bpe")
            vocab_size: Maximum vocabulary size (for word/bpe tokenization)
            lowercase: Whether to lowercase text
        """
        self.tokenization_type = tokenization_type
        self.vocab_size = vocab_size
        self.lowercase = lowercase

        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"
        self.UNK_TOKEN = "<UNK>"

        self.special_tokens = [
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN
        ]

        # Vocabularies
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}

        # Initialize with special tokens
        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        for idx, token in enumerate(self.special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    @property
    def pad_idx(self) -> int:
        """Get padding token index."""
        return self.token2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        """Get start-of-sequence token index."""
        return self.token2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        """Get end-of-sequence token index."""
        return self.token2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        """Get unknown token index."""
        return self.token2idx[self.UNK_TOKEN]

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of text strings to build vocabulary from
        """
        logger.info(f"Building {self.tokenization_type} vocabulary from {len(texts)} texts...")

        # Process texts
        all_tokens = []
        for text in texts:
            if self.lowercase:
                text = text.lower()

            if self.tokenization_type == "character":
                tokens = list(text)
            elif self.tokenization_type == "word":
                tokens = text.split()
            else:
                raise ValueError(f"Unsupported tokenization type: {self.tokenization_type}")

            all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Get most common tokens (respecting vocab_size if specified)
        if self.vocab_size:
            # Reserve space for special tokens
            max_vocab = self.vocab_size - len(self.special_tokens)
            most_common = token_counts.most_common(max_vocab)
        else:
            most_common = token_counts.most_common()

        # Build vocabulary (special tokens already added)
        current_idx = len(self.special_tokens)
        for token, count in most_common:
            if token not in self.token2idx:
                self.token2idx[token] = current_idx
                self.idx2token[current_idx] = token
                current_idx += 1

        logger.info(f"Vocabulary built with {len(self.token2idx)} tokens")
        logger.info(f"Special tokens: {self.special_tokens}")
        logger.info(f"Sample tokens: {list(self.token2idx.keys())[len(self.special_tokens):len(self.special_tokens)+20]}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token indices.

        Args:
            text: Text string to encode
            add_special_tokens: Whether to add SOS and EOS tokens

        Returns:
            List of token indices
        """
        if self.lowercase:
            text = text.lower()

        # Tokenize
        if self.tokenization_type == "character":
            tokens = list(text)
        elif self.tokenization_type == "word":
            tokens = text.split()
        else:
            raise ValueError(f"Unsupported tokenization type: {self.tokenization_type}")

        # Convert to indices
        indices = [self.token2idx.get(token, self.unk_idx) for token in tokens]

        # Add special tokens if requested
        if add_special_tokens:
            indices = [self.sos_idx] + indices + [self.eos_idx]

        return indices

    def decode(self, indices: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token indices back into text.

        Args:
            indices: Token indices (single sequence or batch)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string or list of strings
        """
        # Handle batch decoding
        if indices and isinstance(indices[0], list):
            return [self.decode(seq, skip_special_tokens) for seq in indices]

        # Single sequence decoding
        tokens = []
        for idx in indices:
            if isinstance(idx, int):
                token = self.idx2token.get(idx, self.UNK_TOKEN)
            else:
                token = self.idx2token.get(idx.item(), self.UNK_TOKEN)

            # Skip special tokens if requested
            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        # Join based on tokenization type
        if self.tokenization_type == "character":
            return "".join(tokens)
        elif self.tokenization_type == "word":
            return " ".join(tokens)
        else:
            return " ".join(tokens)

    def save(self, path: Union[str, Path]):
        """
        Save tokenizer to file.

        Args:
            path: Path to save tokenizer
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer_data = {
            'tokenization_type': self.tokenization_type,
            'vocab_size': self.vocab_size,
            'lowercase': self.lowercase,
            'token2idx': self.token2idx,
            'idx2token': {int(k): v for k, v in self.idx2token.items()},
            'special_tokens': self.special_tokens
        }

        # Save as JSON for readability
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Tokenizer':
        """
        Load tokenizer from file.

        Args:
            path: Path to load tokenizer from

        Returns:
            Loaded tokenizer instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        tokenizer = cls(
            tokenization_type=tokenizer_data['tokenization_type'],
            vocab_size=tokenizer_data.get('vocab_size'),
            lowercase=tokenizer_data.get('lowercase', True)
        )

        tokenizer.token2idx = tokenizer_data['token2idx']
        tokenizer.idx2token = {int(k): v for k, v in tokenizer_data['idx2token'].items()}

        logger.info(f"Tokenizer loaded from {path}")
        logger.info(f"Vocabulary size: {len(tokenizer.token2idx)}")

        return tokenizer

    def get_vocab_info(self) -> Dict[str, any]:
        """
        Get vocabulary information.

        Returns:
            Dictionary containing vocabulary metadata
        """
        return {
            'tokenization_type': self.tokenization_type,
            'vocab_size': len(self.token2idx),
            'lowercase': self.lowercase,
            'pad_idx': self.pad_idx,
            'sos_idx': self.sos_idx,
            'eos_idx': self.eos_idx,
            'unk_idx': self.unk_idx,
            'special_tokens': self.special_tokens
        }

    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.token2idx)
