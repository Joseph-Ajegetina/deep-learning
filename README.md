# Deep Learning - Prosits

This repository contains coursework for the Deep Learning course at Ashesi University, focusing on Recurrent Neural Networks (RNNs) and sequence-to-sequence models.

## Repository Structure

```
deep-learning/
├── afrinic-20/
│   └── afrinic-20-asr.ipynb          # African speech recognition using RNNs
├── librispeech-100/
│   └── librispeech.ipynb             # English ASR experiments on LibriSpeech dataset
├── RNN-machine-translation/
│   └── machine-translation.ipynb     # French→English neural machine translation
└── README.md                          # This file
```

## Projects

### 1. AfriSpeech-200 ASR (`afrinic-20/`)
**Automatic Speech Recognition for African Languages**

- Dataset: AfriSpeech-200 (Twi language)
- Models: Comparison of RNN architectures (RNN, LSTM, GRU)
- Techniques: Classical encoder-decoder vs. attention-based seq2seq

**Notebook:** [`afrinic-20/afrinic-20-asr.ipynb`](afrinic-20/afrinic-20-asr.ipynb)

### 2. LibriSpeech ASR (`librispeech-100/`)
**English Speech Recognition Experiments**

- Dataset: LibriSpeech (English)
- Focus: ASR pipeline validation and preprocessing
- Feature extraction: MFCC

**Notebook:** [`librispeech-100/librispeech.ipynb`](librispeech-100/librispeech.ipynb)

### 3. Neural Machine Translation (`RNN-machine-translation/`)
**French to English Translation**

- Task: Sequence-to-sequence learning
- Architectures: Vanilla RNN encoder-decoder, attention-based seq2seq
- Demonstrates the impact of attention mechanisms

**Notebook:** [`RNN-machine-translation/machine-translation.ipynb`](RNN-machine-translation/machine-translation.ipynb)

## Quick Start

### Prerequisites
```bash
pip install torch torchaudio datasets librosa soundfile jiwer matplotlib seaborn numpy tqdm
```

### Running Notebooks
```bash
jupyter notebook
```

Navigate to the respective project folder and open the notebook.

## Key Concepts Covered

- Recurrent Neural Networks (RNN, LSTM, GRU)
- Sequence-to-sequence models
- Attention mechanisms (Bahdanau attention)
- Automatic Speech Recognition (ASR)
- Neural Machine Translation (NMT)
- Audio preprocessing (MFCC features)
- Evaluation metrics (WER, CER, BLEU)

## Course Information

- **Institution:** Ashesi University
- **Course:** Deep Learning
- **Semester:** 1

## License

This project is for educational purposes.
