#!/bin/bash

# LibriSpeech ASR Setup Script
# This script sets up the environment for training ASR models

echo "========================================="
echo "LibriSpeech ASR Setup"
echo "========================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment already exists"
else
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/librispeech
mkdir -p data/cache
mkdir -p checkpoints
mkdir -p evaluation_results
mkdir -p logs
mkdir -p plots
echo "✅ Directories created"
echo ""

# Verify installations
echo "🔍 Verifying installations..."
python -c "import torch; print(f'   PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'   Torchaudio: {torchaudio.__version__}')"
python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Open the Jupyter notebook:"
echo "   cd notebooks/"
echo "   jupyter notebook librispeech_asr_walkthrough.ipynb"
echo ""
echo "3. Or run Python scripts directly:"
echo "   python -c 'from src.data.dataset import LibriSpeechDataModule; print(\"Ready!\")'"
echo ""
echo "Happy learning! 🎓🎤"
