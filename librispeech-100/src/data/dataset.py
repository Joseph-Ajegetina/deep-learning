"""
LibriSpeech dataset module with PyTorch DataLoader support.

Handles:
- Loading LibriSpeech from torchaudio
- Audio preprocessing and feature extraction
- Text tokenization
- Batching with padding for variable-length sequences
- Data augmentation during training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import numpy as np

from .preprocessing import AudioPreprocessor, AudioAugmentor
from .download import setup_librispeech_dataset
from ..utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for LibriSpeech.

    Handles audio loading, preprocessing, and text tokenization.
    """

    def __init__(
        self,
        librispeech_dataset,
        audio_preprocessor: AudioPreprocessor,
        tokenizer: Tokenizer,
        audio_augmentor: Optional[AudioAugmentor] = None,
        max_audio_duration: float = 15.0,
        min_audio_duration: float = 1.0,
        apply_augmentation: bool = False
    ):
        """
        Initialize dataset.

        Args:
            librispeech_dataset: Torchaudio LibriSpeech dataset object
            audio_preprocessor: Audio preprocessing pipeline
            tokenizer: Text tokenizer
            audio_augmentor: Optional audio augmentation
            max_audio_duration: Maximum audio duration in seconds
            min_audio_duration: Minimum audio duration in seconds
            apply_augmentation: Whether to apply augmentation
        """
        self.librispeech_dataset = librispeech_dataset
        self.audio_preprocessor = audio_preprocessor
        self.tokenizer = tokenizer
        self.audio_augmentor = audio_augmentor
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        self.apply_augmentation = apply_augmentation

        # Filter dataset by duration
        self.valid_indices = self._filter_by_duration()

        logger.info(f"LibriSpeech dataset initialized with {len(self.valid_indices)} valid samples "
                   f"(out of {len(self.librispeech_dataset)} total)")

    def _filter_by_duration(self) -> List[int]:
        """
        Filter samples by audio duration.

        Returns:
            List of valid indices
        """
        valid_indices = []

        logger.info(f"Filtering samples by duration ({self.min_audio_duration}s - {self.max_audio_duration}s)...")

        for idx in range(len(self.librispeech_dataset)):
            try:
                waveform, sample_rate, _, _, _, _ = self.librispeech_dataset[idx]
                duration = waveform.size(1) / sample_rate

                if self.min_audio_duration <= duration <= self.max_audio_duration:
                    valid_indices.append(idx)

            except Exception as e:
                logger.warning(f"Skipping sample {idx}: {e}")
                continue

        logger.info(f"Found {len(valid_indices)} samples within duration limits")

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - audio_features: (time_steps, num_features)
            - audio_length: scalar
            - text_tokens: (seq_len,)
            - text_length: scalar
            - transcription: original text string
        """
        # Get actual index from valid indices
        actual_idx = self.valid_indices[idx]

        # LibriSpeech returns: (waveform, sample_rate, transcription, speaker_id, chapter_id, utterance_id)
        waveform, sample_rate, transcription, speaker_id, chapter_id, utterance_id = \
            self.librispeech_dataset[actual_idx]

        # Ensure waveform is (1, time) or (time,)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)  # Remove channel dimension if stereo
        elif waveform.dim() == 1:
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        # Apply augmentation if enabled
        if self.apply_augmentation and self.audio_augmentor is not None:
            waveform = self.audio_augmentor(waveform, sample_rate)

        # Extract features (MFCC)
        audio_features = self.audio_preprocessor(waveform, sample_rate)

        # audio_features shape: (num_features, time_steps)
        # Transpose to (time_steps, num_features) for RNN
        audio_features = audio_features.transpose(0, 1)

        audio_length = audio_features.size(0)

        # Tokenize transcription
        text_tokens = self.tokenizer.encode(transcription)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        text_length = text_tokens.size(0)

        return {
            'audio_features': audio_features,
            'audio_length': torch.tensor(audio_length, dtype=torch.long),
            'text_tokens': text_tokens,
            'text_length': torch.tensor(text_length, dtype=torch.long),
            'transcription': transcription,
            'speaker_id': speaker_id,
            'chapter_id': chapter_id,
            'utterance_id': utterance_id
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary with padded sequences
    """
    # Sort batch by audio length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['audio_length'], reverse=True)

    # Extract fields
    audio_features = [item['audio_features'] for item in batch]
    audio_lengths = torch.stack([item['audio_length'] for item in batch])

    text_tokens = [item['text_tokens'] for item in batch]
    text_lengths = torch.stack([item['text_length'] for item in batch])

    transcriptions = [item['transcription'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    chapter_ids = [item['chapter_id'] for item in batch]
    utterance_ids = [item['utterance_id'] for item in batch]

    # Pad sequences
    # audio_features: List of (time_steps, num_features) -> (batch, max_time, num_features)
    audio_features_padded = pad_sequence(audio_features, batch_first=True, padding_value=0.0)

    # text_tokens: List of (seq_len,) -> (batch, max_seq_len)
    text_tokens_padded = pad_sequence(text_tokens, batch_first=True, padding_value=0)  # PAD token = 0

    return {
        'audio_features': audio_features_padded,
        'audio_lengths': audio_lengths,
        'text_tokens': text_tokens_padded,
        'text_lengths': text_lengths,
        'transcriptions': transcriptions,
        'speaker_ids': speaker_ids,
        'chapter_ids': chapter_ids,
        'utterance_ids': utterance_ids
    }


class LibriSpeechDataModule:
    """
    Data module for LibriSpeech that handles:
    - Dataset downloading and setup
    - Tokenizer creation
    - Data loader creation
    - Train/val/test splits
    """

    def __init__(self, config: Dict):
        """
        Initialize LibriSpeech data module.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tokenizer = None
        self.audio_preprocessor = None
        self.audio_augmentor = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.splits = None

    def setup(self):
        """
        Setup datasets, tokenizer, and preprocessors.
        """
        logger.info("Setting up LibriSpeech data module...")

        # Download/load datasets
        self.splits = setup_librispeech_dataset(self.config)

        # Build tokenizer from training data
        logger.info("Building character-level tokenizer...")
        self.tokenizer = self._build_tokenizer()

        # Create audio preprocessor
        logger.info("Creating audio preprocessor...")
        self.audio_preprocessor = AudioPreprocessor(
            feature_type=self.config['audio']['feature_type'],
            sample_rate=self.config['dataset']['sample_rate'],
            n_mfcc=self.config['audio']['n_mfcc'],
            n_fft=self.config['audio']['n_fft'],
            hop_length=self.config['audio']['hop_length'],
            win_length=self.config['audio']['win_length'],
            n_mels=self.config['audio']['n_mels'],
            normalize=self.config['audio']['normalize']
        )

        # Create audio augmentor
        if self.config['augmentation']['train']['spec_augment']:
            logger.info("Creating audio augmentor...")
            self.audio_augmentor = AudioAugmentor(
                freq_mask_param=self.config['augmentation']['train']['freq_mask_param'],
                time_mask_param=self.config['augmentation']['train']['time_mask_param'],
                sample_rate=self.config['dataset']['sample_rate']
            )

        # Create datasets
        logger.info("Creating PyTorch datasets...")

        self.train_dataset = LibriSpeechDataset(
            librispeech_dataset=self.splits['train'],
            audio_preprocessor=self.audio_preprocessor,
            tokenizer=self.tokenizer,
            audio_augmentor=self.audio_augmentor,
            max_audio_duration=self.config['dataset']['max_audio_duration'],
            min_audio_duration=self.config['dataset']['min_audio_duration'],
            apply_augmentation=True
        )

        self.val_dataset = LibriSpeechDataset(
            librispeech_dataset=self.splits['dev'],
            audio_preprocessor=self.audio_preprocessor,
            tokenizer=self.tokenizer,
            max_audio_duration=self.config['dataset']['max_audio_duration'],
            min_audio_duration=self.config['dataset']['min_audio_duration'],
            apply_augmentation=False
        )

        self.test_dataset = LibriSpeechDataset(
            librispeech_dataset=self.splits['test'],
            audio_preprocessor=self.audio_preprocessor,
            tokenizer=self.tokenizer,
            max_audio_duration=self.config['dataset']['max_audio_duration'],
            min_audio_duration=self.config['dataset']['min_audio_duration'],
            apply_augmentation=False
        )

        logger.info("LibriSpeech data module setup complete!")

    def _build_tokenizer(self) -> Tokenizer:
        """Build character-level tokenizer from training data."""
        # Collect all transcriptions
        transcriptions = []

        logger.info("Collecting transcriptions for tokenizer...")

        # Sample from training data (use all or subset for speed)
        max_samples = min(len(self.splits['train']), 10000)

        for idx in range(max_samples):
            _, _, transcription, _, _, _ = self.splits['train'][idx]
            transcriptions.append(transcription)

        # Create tokenizer
        tokenizer = Tokenizer(
            tokenization_type=self.config['tokenizer']['type'],
            lowercase=self.config['tokenizer']['lowercase']
        )

        # Build vocabulary
        tokenizer.build_vocab(transcriptions)

        logger.info(f"Tokenizer created with vocabulary size: {len(tokenizer)}")

        # Save tokenizer
        save_path = Path(self.config['tokenizer']['save_path'])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))

        return tokenizer

    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """
        Create data loaders for train/val/test.

        Returns:
            Dictionary of data loaders
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data_loader']['batch_size'],
            shuffle=self.config['data_loader']['shuffle_train'],
            num_workers=self.config['dataset']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=self.config['data_loader']['pin_memory'],
            drop_last=self.config['data_loader']['drop_last']
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data_loader']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=self.config['data_loader']['pin_memory'],
            drop_last=False
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['data_loader']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=self.config['data_loader']['pin_memory'],
            drop_last=False
        )

        return {
            'train': train_loader,
            'validation': val_loader,
            'test': test_loader
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'dataset_name': self.config['dataset']['name'],
            'train_samples': len(self.train_dataset),
            'validation_samples': len(self.val_dataset),
            'test_samples': len(self.test_dataset),
            'vocab_size': len(self.tokenizer),
            'feature_dim': self.config['audio']['n_mfcc'],
            'feature_type': self.config['audio']['feature_type'],
            'sample_rate': self.config['dataset']['sample_rate'],
            'tokenizer_type': self.config['tokenizer']['type']
        }
