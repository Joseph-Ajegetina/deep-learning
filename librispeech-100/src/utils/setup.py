"""
Environment setup utility for AfriSpeech speech recognition project.

Handles:
- Device detection (GPU/CPU)
- Random seed setting
- Configuration loading
- Directory creation
- Dataset verification
"""

import torch
import numpy as np
import random
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cudnn deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"🎲 Random seed set to: {seed}")


def get_device(config: Optional[Dict[str, Any]] = None) -> torch.device:
    """
    Get computing device (GPU or CPU).

    Args:
        config: Optional configuration dict with 'device' key

    Returns:
        torch.device
    """
    if config and 'device' in config:
        device_name = config['device']
        if device_name == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(device_name)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("📱 Using CPU (GPU not available)")

    return device


def load_config(config_path: str = "./configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"📋 Configuration loaded from: {config_path}")
    return config


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories for the project.

    Args:
        config: Configuration dictionary
    """
    directories = [
        config['dataset'].get('data_dir', './data/afrispeech'),
        config['dataset'].get('cache_dir', './data/cache'),
        config['logging'].get('log_dir', './logs'),
        config['logging'].get('tensorboard_dir', './runs'),
        config['logging'].get('plot_dir', './plots'),
        './checkpoints',
        './exported_models',
        './evaluation_results',
        './notebooks'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("📁 Created project directories")


def verify_dataset(data_dir: str) -> bool:
    """
    Verify that dataset exists and is properly organized.

    Args:
        data_dir: Path to dataset directory

    Returns:
        True if dataset is valid
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"⚠️ Dataset directory not found: {data_dir}")
        logger.info("   Dataset will be downloaded automatically on first use")
        return False

    logger.info(f"✅ Dataset directory exists: {data_dir}")
    return True


def setup_env(config_path: str = "./configs/config.yaml") -> Dict[str, Any]:
    """
    Complete environment setup for speech recognition project.

    This is the main setup function that should be called at the start
    of training scripts or notebooks.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing setup information
    """
    print("🚀 Setting up AfriSpeech speech recognition environment...")

    # Setup logging
    setup_logging(level="INFO")

    # Load configuration
    config = load_config(config_path)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Get device
    device = get_device(config)

    # Create directories
    create_directories(config)

    # Verify dataset (don't fail if not present, will download later)
    data_dir = config['dataset'].get('data_dir', './data/afrispeech')
    dataset_exists = verify_dataset(data_dir)

    setup_info = {
        'config': config,
        'device': device,
        'seed': seed,
        'cuda_available': torch.cuda.is_available(),
        'dataset_exists': dataset_exists,
        'data_dir': data_dir
    }

    print("\n✅ Environment setup complete!")
    print(f"📱 Device: {device}")
    print(f"🎲 Seed: {seed}")
    print(f"🔥 CUDA: {torch.cuda.is_available()}")
    print(f"📊 Dataset: {'Found' if dataset_exists else 'Will be downloaded'}")
    print("")

    return setup_info


def verify_setup() -> bool:
    """
    Verify that the environment is properly setup.

    Returns:
        True if setup is valid
    """
    try:
        # Check PyTorch
        import torch
        assert torch.__version__ >= "1.11.0", "PyTorch version >= 1.11.0 required"

        # Check torchaudio
        import torchaudio
        assert torchaudio.__version__ >= "0.11.0", "torchaudio version >= 0.11.0 required"

        # Check HuggingFace datasets
        import datasets

        # Check other key dependencies
        import librosa
        import yaml

        logger.info("✅ Environment verification passed")
        return True

    except ImportError as e:
        logger.error(f"❌ Environment verification failed: {e}")
        logger.error("   Please install required dependencies: pip install -r requirements.txt")
        return False
    except AssertionError as e:
        logger.error(f"❌ Version check failed: {e}")
        return False


def print_model_info(model: torch.nn.Module):
    """
    Print model information for debugging.

    Args:
        model: PyTorch model
    """
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)
        print(f"Type: {info.get('type', 'Unknown')}")

        if hasattr(model, 'count_parameters'):
            params = model.count_parameters()
            print(f"Parameters: {params:,}")

        if hasattr(model, 'get_parameter_size_mb'):
            size = model.get_parameter_size_mb()
            print(f"Size: {size:.2f} MB")

        print("=" * 60 + "\n")
    else:
        print(f"\nModel: {model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,}\n")


def get_data_location() -> str:
    """
    Get the location of the dataset.

    Returns:
        Path to dataset directory
    """
    try:
        config = load_config()
        data_dir = config['dataset'].get('data_dir', './data/afrispeech')
        return data_dir
    except:
        return './data/afrispeech'
