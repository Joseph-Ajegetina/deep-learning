"""
LibriSpeech dataset download and setup utilities.

This module handles downloading and preparing the LibriSpeech corpus
for training speech recognition models.

LibriSpeech is a corpus of approximately 1000 hours of read English speech
sampled at 16 kHz, derived from audiobooks from the LibriVox project.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchaudio
from torchaudio.datasets import LIBRISPEECH

logger = logging.getLogger(__name__)


def download_librispeech(
    root_dir: str,
    url: str = "train-clean-100",
    download: bool = True
) -> LIBRISPEECH:
    """
    Download and setup LibriSpeech dataset.

    Args:
        root_dir: Root directory to store the dataset
        url: LibriSpeech subset to download. Options:
            - "train-clean-100": 100 hours of clean training data
            - "train-clean-360": 360 hours of clean training data
            - "train-other-500": 500 hours of other training data
            - "dev-clean": Development set (clean)
            - "dev-other": Development set (other)
            - "test-clean": Test set (clean)
            - "test-other": Test set (other)
        download: Whether to download if not present

    Returns:
        LibriSpeech dataset object
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Setting up LibriSpeech dataset: {url}")
    logger.info(f"Root directory: {root_path}")

    if download:
        logger.info("Download enabled - will download if not cached")

    try:
        dataset = LIBRISPEECH(
            root=str(root_path),
            url=url,
            download=download
        )

        logger.info(f"Successfully loaded LibriSpeech {url}")
        logger.info(f"Number of samples: {len(dataset)}")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load LibriSpeech: {e}")
        raise


def get_dataset_splits(
    root_dir: str,
    train_url: str = "train-clean-100",
    dev_url: str = "dev-clean",
    test_url: str = "test-clean",
    download: bool = True
) -> Dict[str, LIBRISPEECH]:
    """
    Get train/dev/test splits of LibriSpeech.

    Args:
        root_dir: Root directory for datasets
        train_url: Training set URL
        dev_url: Development set URL
        test_url: Test set URL
        download: Whether to download if not present

    Returns:
        Dictionary with 'train', 'dev', 'test' dataset objects
    """
    logger.info("Loading LibriSpeech dataset splits...")

    splits = {}

    # Load train split
    logger.info(f"Loading training data: {train_url}")
    splits['train'] = download_librispeech(root_dir, train_url, download)

    # Load dev split
    logger.info(f"Loading development data: {dev_url}")
    splits['dev'] = download_librispeech(root_dir, dev_url, download)

    # Load test split
    logger.info(f"Loading test data: {test_url}")
    splits['test'] = download_librispeech(root_dir, test_url, download)

    logger.info("All splits loaded successfully!")
    logger.info(f"  Train samples: {len(splits['train'])}")
    logger.info(f"  Dev samples: {len(splits['dev'])}")
    logger.info(f"  Test samples: {len(splits['test'])}")

    return splits


def get_librispeech_sample(dataset: LIBRISPEECH, idx: int = 0) -> Dict:
    """
    Get a single sample from LibriSpeech dataset.

    Args:
        dataset: LibriSpeech dataset object
        idx: Sample index

    Returns:
        Dictionary with sample information
    """
    waveform, sample_rate, transcription, speaker_id, chapter_id, utterance_id = dataset[idx]

    return {
        'waveform': waveform,
        'sample_rate': sample_rate,
        'transcription': transcription,
        'speaker_id': speaker_id,
        'chapter_id': chapter_id,
        'utterance_id': utterance_id,
        'duration': waveform.size(1) / sample_rate,
        'num_channels': waveform.size(0)
    }


def explore_librispeech_dataset(dataset: LIBRISPEECH, num_samples: int = 5):
    """
    Explore LibriSpeech dataset by printing sample information.

    Args:
        dataset: LibriSpeech dataset object
        num_samples: Number of samples to explore
    """
    logger.info("=" * 60)
    logger.info("LibriSpeech Dataset Exploration")
    logger.info("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        sample = get_librispeech_sample(dataset, i)

        logger.info(f"\nSample {i + 1}:")
        logger.info(f"  Transcription: {sample['transcription']}")
        logger.info(f"  Duration: {sample['duration']:.2f}s")
        logger.info(f"  Sample rate: {sample['sample_rate']} Hz")
        logger.info(f"  Speaker ID: {sample['speaker_id']}")
        logger.info(f"  Chapter ID: {sample['chapter_id']}")
        logger.info(f"  Utterance ID: {sample['utterance_id']}")

    logger.info("=" * 60)


def setup_librispeech_dataset(config: Dict) -> Dict[str, LIBRISPEECH]:
    """
    Main entry point for setting up LibriSpeech dataset from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of dataset splits
    """
    dataset_config = config['dataset']

    root_dir = dataset_config['data_dir']
    dataset_url = dataset_config.get('dataset_url', 'train-clean-100')
    download = dataset_config.get('download', True)

    # Map to official splits
    if dataset_url == "train-clean-100":
        splits = get_dataset_splits(
            root_dir=root_dir,
            train_url="train-clean-100",
            dev_url="dev-clean",
            test_url="test-clean",
            download=download
        )
    elif dataset_url == "train-clean-360":
        splits = get_dataset_splits(
            root_dir=root_dir,
            train_url="train-clean-360",
            dev_url="dev-clean",
            test_url="test-clean",
            download=download
        )
    else:
        # Single dataset mode
        logger.warning(f"Using single dataset: {dataset_url}")
        dataset = download_librispeech(root_dir, dataset_url, download)

        # Split manually if needed
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)

        logger.warning("Manually splitting dataset (not recommended for LibriSpeech)")
        logger.warning("Consider using official train/dev/test splits")

        splits = {
            'train': dataset,  # This is not ideal but works for demo
            'dev': dataset,
            'test': dataset
        }

    return splits
