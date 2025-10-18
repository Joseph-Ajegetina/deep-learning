# LibriSpeech ASR: Vanilla RNN vs RNN with Attention

A comprehensive educational implementation of Automatic Speech Recognition (ASR) using vanilla RNN and RNN with attention mechanisms on the LibriSpeech corpus.

## Overview

This project implements two ASR architectures from scratch:

1. **Vanilla RNN (Classical Encoder-Decoder)**
   - Bidirectional LSTM encoder
   - Unidirectional LSTM decoder
   - Fixed context vector (bottleneck)

2. **RNN with Attention (Bahdanau)**
   - Same encoder as vanilla RNN
   - Attention mechanism for dynamic context
   - Eliminates the bottleneck problem

## Dataset

**LibriSpeech train-clean-100**
- 100 hours of clean English speech
- 16 kHz sample rate
- Audiobook recordings from LibriVox
- ~28,000 training utterances

## Project Structure

```
librispeech-100/
├── configs/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data/
│   │   ├── download.py          # LibriSpeech download utilities
│   │   ├── dataset.py           # PyTorch dataset & data module
│   │   └── preprocessing.py     # Audio preprocessing (MFCC)
│   ├── models/
│   │   ├── base.py              # Base model class
│   │   ├── rnn_encoder_decoder.py  # Vanilla RNN
│   │   └── rnn_attention.py     # RNN with attention
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   └── evaluator.py         # Evaluation & metrics
│   └── utils/
│       ├── tokenizer.py         # Character-level tokenizer
│       ├── metrics.py           # WER/CER calculation
│       └── setup.py             # Environment setup
├── notebooks/
│   └── librispeech_asr_walkthrough.ipynb  # Complete walkthrough
├── checkpoints/                 # Saved models
├── evaluation_results/          # Evaluation outputs
└── requirements.txt

```

## Installation

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'Torchaudio: {torchaudio.__version__}')"
```

## Quick Start

### Option 1: Use the Jupyter Notebook (Recommended)

The notebook provides a complete, step-by-step walkthrough with explanations:

```bash
cd notebooks/
jupyter notebook librispeech_asr_walkthrough.ipynb
```

Execute cells in order to:
1. Download and explore LibriSpeech
2. Understand the mathematical foundations
3. Train vanilla RNN
4. Train RNN with attention
5. Compare performance
6. Visualize attention weights

### Option 2: Use Python Scripts

```python
import yaml
from src.data.dataset import LibriSpeechDataModule
from src.models import ClassicalRNN, RNNWithAttention
from src.training import SequenceTrainer

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup data
data_module = LibriSpeechDataModule(config)
data_module.setup()
data_loaders = data_module.get_data_loaders()

# Create model
model = RNNWithAttention(
    vocab_size=len(data_module.tokenizer),
    encoder_input_size=config['audio']['n_mfcc'],
    **config['models']['attention_rnn'],
    pad_idx=data_module.tokenizer.pad_idx
)

# Train
trainer = SequenceTrainer(model, data_module.tokenizer, device='cuda', config=config)
history = trainer.train(data_loaders['train'], data_loaders['validation'])
```

## Architecture Details

### Vanilla RNN

```
Audio (MFCC) → BiLSTM Encoder → Fixed Context (c) → LSTM Decoder → Text
```

**Mathematical formulation:**
- Encoder: `h₁, ..., hₜ = BiLSTM(x₁, ..., xₜ)`
- Context: `c = hₜ` (fixed!)
- Decoder: `sₜ = LSTM([emb(y_{t-1}); c], s_{t-1})`
- Output: `p(yₜ) = softmax(W·sₜ + b)`

**Limitation:** Fixed context `c` is a bottleneck for long sequences.

### RNN with Attention

```
Audio (MFCC) → BiLSTM Encoder → Attention → Dynamic Context (cₜ) → LSTM Decoder → Text
```

**Mathematical formulation:**
- Encoder: `h₁, ..., hₜ = BiLSTM(x₁, ..., xₜ)`
- Attention scores: `eₜ,ᵢ = v^T·tanh(W₁·hᵢ + W₂·s_{t-1})`
- Attention weights: `αₜ,ᵢ = exp(eₜ,ᵢ) / Σⱼ exp(eₜ,ⱼ)`
- Dynamic context: `cₜ = Σᵢ αₜ,ᵢ·hᵢ` (different for each step!)
- Decoder: `sₜ = LSTM([emb(y_{t-1}); cₜ], s_{t-1})`
- Output: `p(yₜ) = softmax(W·[sₜ; cₜ] + b)`

**Advantage:** No bottleneck - can access all encoder states dynamically.

## Expected Results

Based on typical LibriSpeech-100 training:

| Model | WER | CER | Parameters | Inference Speed |
|-------|-----|-----|------------|-----------------|
| Vanilla RNN | ~35-45% | ~15-20% | ~15M | Fast |
| RNN + Attention | ~25-35% | ~10-15% | ~18M | Moderate |

**Improvement:** 15-30% WER reduction with attention!

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
dataset:
  dataset_url: "train-clean-100"  # or "train-clean-360"
  batch_size: 16

training:
  epochs: 30
  learning_rate: 0.0003
  teacher_forcing_ratio: 0.5

models:
  attention_rnn:
    attention_type: "bahdanau"  # or "luong"
    encoder_hidden_size: 256
    decoder_hidden_size: 512
```

## Training Tips

1. **Start with a subset:** Use only 1000 samples for quick testing
2. **Monitor WER:** Should decrease steadily during training
3. **Use GPU:** Training on CPU takes much longer
4. **Adjust batch size:** Reduce if you run out of memory
5. **Teacher forcing:** Helps training but decay it over time

## Evaluation Metrics

- **WER (Word Error Rate):** Primary metric for ASR
  - Formula: `(S + D + I) / N`
  - S = substitutions, D = deletions, I = insertions, N = total words
  - Lower is better

- **CER (Character Error Rate):** More fine-grained than WER
  - Similar to WER but at character level
  - More suitable for character-based models

## Attention Visualization

The notebook includes attention heatmap visualization:
- Shows which audio frames the model attends to for each character
- Diagonal patterns indicate proper alignment
- Helps interpret model behavior

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `encoder_hidden_size` or `decoder_hidden_size`
- Use gradient accumulation

### Slow Download
- LibriSpeech train-clean-100 is 6.3 GB
- Download once, then cached for future runs
- Check internet connection

### Poor Performance
- Train for more epochs
- Adjust learning rate
- Increase model size
- Check data preprocessing

## Citation

If you use this code, please cite:

```bibtex
@misc{librispeech_asr,
  title={LibriSpeech ASR: Vanilla RNN vs RNN with Attention},
  author={Ashesi Deep Learning Course},
  year={2025},
  howpublished={\\url{https://github.com/...}}
}
```

### LibriSpeech Dataset

```bibtex
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={ICASSP},
  year={2015}
}
```

### Attention Mechanism

```bibtex
@article{bahdanau2014neural,
  title={Neural machine translation by jointly learning to align and translate},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1409.0473},
  year={2014}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- LibriSpeech corpus creators
- PyTorch and torchaudio teams
- Bahdanau et al. for the attention mechanism
- Ashesi University Deep Learning Course

## Contact

For questions or issues, please open an issue on GitHub or contact the course instructors.

---

**Happy Learning!** 🎓🎤🔊
