# Quick Start Guide - LibriSpeech ASR

Get started with the LibriSpeech ASR implementation in 5 minutes!

## Prerequisites

- Python 3.9 or later
- 10 GB free disk space (for dataset)
- CUDA-capable GPU (optional but recommended)
- Internet connection (for first-time download)

## Installation (5 minutes)

### Option 1: Automatic Setup (Recommended)

```bash
cd librispeech-100/
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/librispeech checkpoints evaluation_results logs plots
```

## Quick Test (2 minutes)

Verify everything works:

```python
python -c "
import torch
import torchaudio
from src.models import ClassicalRNN, RNNWithAttention

print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Run the Notebook (Recommended)

```bash
cd notebooks/
jupyter notebook librispeech_asr_walkthrough.ipynb
```

Then execute cells in order to:
1. Download LibriSpeech (~30 min first time, cached after)
2. Train vanilla RNN
3. Train RNN with attention
4. Compare results
5. Visualize attention weights

## Quick Training Script

For a minimal working example:

```python
import yaml
import torch
from src.data.dataset import LibriSpeechDataModule
from src.models import RNNWithAttention
from src.training import SequenceTrainer

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup data (downloads LibriSpeech if needed)
print("Setting up data...")
data_module = LibriSpeechDataModule(config)
data_module.setup()
data_loaders = data_module.get_data_loaders()

print(f"Train samples: {len(data_module.train_dataset)}")
print(f"Vocab size: {len(data_module.tokenizer)}")

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RNNWithAttention(
    vocab_size=len(data_module.tokenizer),
    encoder_input_size=config['audio']['n_mfcc'],
    **config['models']['attention_rnn'],
    pad_idx=data_module.tokenizer.pad_idx
).to(device)

print(f"Model parameters: {model.count_parameters():,}")

# Train
trainer = SequenceTrainer(
    model=model,
    tokenizer=data_module.tokenizer,
    device=device,
    config=config,
    experiment_name="QuickStart_Attention"
)

print("Starting training...")
history = trainer.train(
    train_loader=data_loaders['train'],
    val_loader=data_loaders['validation']
)

print(f"Best WER: {trainer.best_val_wer:.2f}%")
```

## Configuration Tips

Edit `configs/config.yaml` to customize:

### For Quick Testing (Faster)
```yaml
data_loader:
  batch_size: 8  # Smaller batch

training:
  epochs: 5  # Fewer epochs

dataset:
  max_audio_duration: 10  # Shorter utterances
```

### For Best Results (Slower)
```yaml
data_loader:
  batch_size: 32  # Larger batch

training:
  epochs: 30
  learning_rate: 0.0003

models:
  attention_rnn:
    encoder_hidden_size: 512  # Larger model
    decoder_hidden_size: 1024
```

## Common Commands

### Training
```bash
# Train vanilla RNN
python train.py --model classical_rnn --config configs/config.yaml

# Train RNN with attention
python train.py --model attention_rnn --config configs/config.yaml
```

### Evaluation
```bash
# Evaluate model
python evaluate.py --checkpoint checkpoints/best_model.pth --data test

# Visualize attention
python visualize_attention.py --checkpoint checkpoints/best_model.pth
```

### Testing Dataset Download
```python
from torchaudio.datasets import LIBRISPEECH

# Download small test set first (346 MB)
dataset = LIBRISPEECH(root='./data/librispeech', url='test-clean', download=True)
print(f"Downloaded {len(dataset)} samples")
```

## Expected Timeline

| Task | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Setup & install | 5 min | 5 min |
| Download LibriSpeech-100 | 30 min | 30 min |
| Train vanilla RNN (1 epoch) | 15 min | 2 hours |
| Train vanilla RNN (30 epochs) | 6 hours | 2-3 days |
| Train attention RNN (30 epochs) | 8 hours | 3-4 days |
| Evaluation | 10 min | 30 min |

## Troubleshooting

### "CUDA out of memory"
```yaml
# In config.yaml
data_loader:
  batch_size: 4  # Reduce from 16
```

### "Download too slow"
```python
# Use smaller subset first
url='dev-clean'  # Only 337 MB instead of 6.3 GB
```

### "Import errors"
```bash
# Verify you're in the right directory
cd librispeech-100/
python -c "import sys; print(sys.path)"

# Check __init__.py files exist
find src -name "__init__.py"
```

### "Poor WER"
- Train longer (30+ epochs)
- Increase model size
- Check learning rate
- Verify data preprocessing

## Directory Structure After Setup

```
librispeech-100/
├── data/
│   ├── librispeech/          # Downloaded LibriSpeech
│   │   └── LibriSpeech/
│   │       ├── train-clean-100/
│   │       ├── dev-clean/
│   │       └── test-clean/
│   ├── cache/                # Cached preprocessed features
│   └── tokenizer.json        # Saved tokenizer
├── checkpoints/
│   └── RNN_Attention_LibriSpeech/
│       ├── best_model.pth
│       └── training_history.yaml
├── evaluation_results/
│   ├── attention_heatmap.png
│   └── model_comparison.png
└── logs/
    └── training.log
```

## Key Files to Modify

1. **configs/config.yaml** - All hyperparameters
2. **src/models/rnn_attention.py** - Model architecture
3. **notebooks/librispeech_asr_walkthrough.ipynb** - Experiments

## Next Steps

After completing this quick start:

1. ✅ Complete the full notebook walkthrough
2. ✅ Experiment with different hyperparameters
3. ✅ Try Luong attention instead of Bahdanau
4. ✅ Implement beam search decoding
5. ✅ Compare on Afrinic dataset

## Help & Support

- **Documentation**: See README.md
- **Implementation details**: See IMPLEMENTATION_SUMMARY.md
- **Issues**: Check troubleshooting section
- **Examples**: See notebook for complete walkthrough

## Performance Expectations

After training for 30 epochs on LibriSpeech-100:

- **Vanilla RNN WER**: 35-45%
- **Attention RNN WER**: 25-35%
- **Improvement**: 15-30% relative

These are baseline results. State-of-the-art models (Conformer, Whisper) achieve <5% WER on LibriSpeech.

---

**Ready to start?** Open the notebook and begin learning! 🎓

```bash
jupyter notebook notebooks/librispeech_asr_walkthrough.ipynb
```
