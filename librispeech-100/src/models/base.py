import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseSequenceModel(nn.Module, ABC):
    """
    Abstract base class for all sequence-to-sequence models in speech recognition.

    This class provides a common interface for Classical RNN and RNN with Attention models,
    ensuring consistent implementation and evaluation across different architectures.

    Designed for speech-to-text transcription tasks where:
    - Input: Audio features (MFCC, spectrograms) of variable length
    - Output: Text transcription as sequence of tokens
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.model_type = None

    @abstractmethod
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            audio_features: Input audio features (batch_size, max_time, feature_dim)
            audio_lengths: Actual lengths of each audio sequence (batch_size,)
            targets: Target token sequences (batch_size, max_target_len) for training
            teacher_forcing_ratio: Probability of using teacher forcing during training

        Returns:
            logits: Output logits (batch_size, max_target_len, vocab_size)
            attention_weights: Attention weights if applicable (batch_size, max_target_len, max_time)
        """
        pass

    @abstractmethod
    def decode(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        max_length: int = 200,
        beam_size: int = 1,
        sos_idx: int = 1,
        eos_idx: int = 2
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode audio features into text sequence (inference).

        Args:
            audio_features: Input audio features (batch_size, max_time, feature_dim)
            audio_lengths: Actual lengths of each audio sequence (batch_size,)
            max_length: Maximum length of decoded sequence
            beam_size: Beam size for beam search (1 = greedy decoding)
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index

        Returns:
            predictions: Predicted token sequences (batch_size, max_length)
            attention_weights: Attention weights if applicable
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns model information including architecture details.

        Returns:
            Dictionary containing model metadata and architecture description
        """
        pass

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_size_mb(self) -> float:
        """
        Calculate model size in megabytes.

        Returns:
            Model size in MB
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def summary(self) -> str:
        """
        Generate a summary of the model.

        Returns:
            String containing model summary
        """
        info = self.get_model_info()
        summary_str = f"""
Model Summary:
=============
Type: {info.get('type', 'Unknown')}
Architecture: {info.get('architecture', 'Not specified')}
Total Parameters: {self.count_parameters():,}
Model Size: {self.get_parameter_size_mb():.2f} MB
Vocabulary Size: {self.vocab_size}
        """
        return summary_str.strip()

    def save_model(self, path: str):
        """
        Save model state dict.

        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'num_parameters': self.count_parameters(),
            'model_size_mb': self.get_parameter_size_mb(),
            'vocab_size': self.vocab_size,
            'pad_idx': self.pad_idx
        }, path)

    def load_model(self, path: str):
        """
        Load model state dict.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
