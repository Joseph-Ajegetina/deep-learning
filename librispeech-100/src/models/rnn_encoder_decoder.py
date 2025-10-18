"""
Classical RNN Encoder-Decoder for Speech Recognition.

Architecture:
- Encoder: Bidirectional LSTM/GRU to process audio features
- Decoder: Unidirectional LSTM/GRU to generate text sequence
- No attention mechanism (uses fixed context vector)

Mathematical Foundation:
========================================

LSTM Cell Operations:
--------------------
For each time step t:

1. Forget Gate:  f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
   Controls what to forget from cell state

2. Input Gate:   i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
   Controls what new information to add

3. Cell Update:  C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)
   New candidate values

4. Cell State:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
   Update cell state

5. Output Gate:  o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
   Controls what to output

6. Hidden State: h_t = o_t ⊙ tanh(C_t)
   Final hidden state

Encoder-Decoder Framework:
--------------------------
1. Encoder: h_1,...,h_T = BiLSTM(x_1,...,x_T)
2. Context: c = h_T (or mean pooling)
3. Decoder: s_t = LSTM([emb(y_{t-1}); c], s_{t-1})
4. Output:  p(y_t) = softmax(W·s_t + b)

Limitation: Fixed context vector is a bottleneck for long sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import random

from .base import BaseSequenceModel


class Encoder(nn.Module):
    """
    RNN Encoder for audio features.

    Processes variable-length audio sequences and produces
    a fixed-size context vector (and optionally all hidden states).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout = dropout

        # RNN layer
        rnn_class = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode audio features.

        Args:
            audio_features: (batch_size, max_time, input_size)
            audio_lengths: (batch_size,)

        Returns:
            outputs: (batch_size, max_time, hidden_size * num_directions)
            hidden: Tuple of (h_n, c_n) for LSTM or just h_n for GRU
        """
        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            audio_features,
            audio_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Forward through RNN
        packed_outputs, hidden = self.rnn(packed)

        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        return outputs, hidden


class Decoder(nn.Module):
    """
    RNN Decoder for text generation.

    Generates text sequence one token at a time, conditioned on
    a fixed context vector from the encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.dropout = dropout
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )

        # RNN layer (input is embedding + context)
        rnn_class = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_class(
            input_size=embedding_dim + hidden_size,  # Concatenate embedding and context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        context: torch.Tensor,
        hidden: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Decoder forward pass for one time step.

        Args:
            input_token: (batch_size,) or (batch_size, 1)
            context: (batch_size, hidden_size) - fixed context from encoder
            hidden: Previous hidden state

        Returns:
            output: (batch_size, vocab_size)
            hidden: Updated hidden state
        """
        # Ensure input_token is (batch_size, 1)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        # Embed input token: (batch_size, 1, embedding_dim)
        embedded = self.embedding(input_token)
        embedded = self.dropout_layer(embedded)

        # Expand context to match sequence length
        # context: (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        context_expanded = context.unsqueeze(1)

        # Concatenate embedding and context
        rnn_input = torch.cat([embedded, context_expanded], dim=2)

        # Forward through RNN
        output, hidden = self.rnn(rnn_input, hidden)

        # Project to vocabulary
        # output: (batch_size, 1, hidden_size) -> (batch_size, 1, vocab_size)
        logits = self.fc_out(output)

        # Squeeze time dimension: (batch_size, vocab_size)
        logits = logits.squeeze(1)

        return logits, hidden


class ClassicalRNN(BaseSequenceModel):
    """
    Classical RNN Encoder-Decoder for Speech Recognition (PROSIT 2a).

    This model uses a fixed-size context vector to represent the entire
    input audio sequence, which is then used by the decoder to generate
    the output text sequence.

    Key Limitation:
    - The fixed context vector is a bottleneck for long sequences
    - All information must be compressed into a single vector
    - Early parts of the sequence may be "forgotten"

    This limitation motivates the use of attention mechanisms (PROSIT 2b).
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_input_size: int,
        encoder_hidden_size: int = 256,
        encoder_num_layers: int = 3,
        encoder_bidirectional: bool = True,
        encoder_dropout: float = 0.3,
        decoder_hidden_size: int = 512,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.3,
        embedding_dim: int = 256,
        encoder_type: str = "lstm",
        pad_idx: int = 0
    ):
        super().__init__(vocab_size=vocab_size, pad_idx=pad_idx)

        self.model_type = "Classical RNN (Encoder-Decoder)"
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bidirectional = encoder_bidirectional
        self.decoder_hidden_size = decoder_hidden_size

        # Encoder
        self.encoder = Encoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            rnn_type=encoder_type,
            bidirectional=encoder_bidirectional,
            dropout=encoder_dropout
        )

        # Context size (bidirectional doubles the hidden size)
        context_size = encoder_hidden_size * (2 if encoder_bidirectional else 1)

        # Bridge layer to transform encoder hidden to decoder hidden
        # (needed if encoder and decoder have different hidden sizes)
        self.bridge = nn.Linear(context_size, decoder_hidden_size)

        # Decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            rnn_type=encoder_type,
            dropout=decoder_dropout,
            pad_idx=pad_idx
        )

    def _get_context(
        self,
        encoder_outputs: torch.Tensor,
        encoder_hidden: Tuple[torch.Tensor, ...],
        audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract fixed context vector from encoder outputs.

        Two strategies:
        1. Use final hidden state
        2. Mean pooling over all outputs

        Args:
            encoder_outputs: (batch_size, max_time, hidden_size * 2)
            encoder_hidden: Final hidden state from encoder
            audio_lengths: Actual sequence lengths

        Returns:
            context: (batch_size, context_size)
        """
        # Strategy 1: Use final hidden state
        # For bidirectional LSTM, concatenate forward and backward final states
        if isinstance(encoder_hidden, tuple):  # LSTM returns (h, c)
            h_n = encoder_hidden[0]  # (num_layers * num_directions, batch, hidden_size)
        else:  # GRU returns just h
            h_n = encoder_hidden

        # Get last layer's hidden states
        # For bidirectional: concat forward and backward
        if self.encoder_bidirectional:
            # Forward: h_n[-2, :, :]
            # Backward: h_n[-1, :, :]
            context = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        else:
            context = h_n[-1, :, :]

        return context

    def _init_decoder_hidden(
        self,
        context: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Initialize decoder hidden state from context.

        Args:
            context: (batch_size, context_size)
            batch_size: Batch size

        Returns:
            Initial hidden state for decoder
        """
        # Transform context to decoder hidden size
        h_0 = torch.tanh(self.bridge(context))

        # Repeat for all decoder layers
        h_0 = h_0.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        if self.decoder.rnn_type == "lstm":
            # Initialize cell state to zeros
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0)
        else:
            return h_0

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Classical RNN.

        Args:
            audio_features: (batch_size, max_time, feature_dim)
            audio_lengths: (batch_size,)
            targets: (batch_size, max_target_len) - for training
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            logits: (batch_size, max_target_len, vocab_size)
            attention_weights: None (no attention in classical RNN)
        """
        batch_size = audio_features.size(0)

        # Encode audio
        encoder_outputs, encoder_hidden = self.encoder(audio_features, audio_lengths)

        # Get fixed context vector
        context = self._get_context(encoder_outputs, encoder_hidden, audio_lengths)

        # Initialize decoder hidden state
        decoder_hidden = self._init_decoder_hidden(context, batch_size)

        # Decoding
        if targets is not None:
            # Training mode with teacher forcing
            max_target_len = targets.size(1)
            all_logits = []

            # Start with SOS token (index 1)
            decoder_input = torch.ones(batch_size, dtype=torch.long, device=audio_features.device)

            for t in range(max_target_len - 1):  # -1 because we don't predict after EOS
                # Decoder step
                logits, decoder_hidden = self.decoder(decoder_input, context, decoder_hidden)
                all_logits.append(logits.unsqueeze(1))

                # Teacher forcing: use ground truth as next input
                use_teacher_forcing = random.random() < teacher_forcing_ratio

                if use_teacher_forcing:
                    decoder_input = targets[:, t]
                else:
                    # Use model's own prediction
                    decoder_input = logits.argmax(dim=1)

            # Concatenate all logits
            logits = torch.cat(all_logits, dim=1)  # (batch_size, max_target_len-1, vocab_size)

            # Pad to match target length
            pad = torch.zeros(batch_size, 1, self.vocab_size, device=logits.device)
            logits = torch.cat([logits, pad], dim=1)

        else:
            # Inference mode (handled by decode method)
            raise NotImplementedError("Use decode() method for inference")

        return logits, None  # No attention weights for classical RNN

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
        Greedy or beam search decoding for inference.

        Args:
            audio_features: (batch_size, max_time, feature_dim)
            audio_lengths: (batch_size,)
            max_length: Maximum output length
            beam_size: Beam size (1 = greedy)
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index

        Returns:
            predictions: (batch_size, max_length)
            attention_weights: None
        """
        batch_size = audio_features.size(0)
        device = audio_features.device

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(audio_features, audio_lengths)
        context = self._get_context(encoder_outputs, encoder_hidden, audio_lengths)
        decoder_hidden = self._init_decoder_hidden(context, batch_size)

        # Greedy decoding (beam search not implemented for simplicity)
        predictions = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)

        for t in range(max_length):
            logits, decoder_hidden = self.decoder(decoder_input, context, decoder_hidden)
            predicted = logits.argmax(dim=1)
            predictions[:, t] = predicted

            # Use prediction as next input
            decoder_input = predicted

            # Stop if all sequences have generated EOS
            if (predicted == eos_idx).all():
                break

        return predictions, None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            "type": self.model_type,
            "architecture": {
                "encoder": {
                    "type": self.encoder.rnn_type.upper(),
                    "hidden_size": self.encoder_hidden_size,
                    "num_layers": self.encoder.num_layers,
                    "bidirectional": self.encoder_bidirectional,
                    "dropout": self.encoder.dropout
                },
                "decoder": {
                    "type": self.decoder.rnn_type.upper(),
                    "hidden_size": self.decoder_hidden_size,
                    "num_layers": self.decoder.num_layers,
                    "embedding_dim": self.decoder.embedding_dim,
                    "dropout": self.decoder.dropout
                },
                "context": "Fixed-size vector (encoder final state)"
            },
            "vocab_size": self.vocab_size,
            "limitation": "Fixed context bottleneck - all audio info compressed to single vector",
            "mathematical_foundation": {
                "encoder": "h_1,...,h_T = BiLSTM(x_1,...,x_T)",
                "context": "c = h_T (final hidden state)",
                "decoder": "s_t = LSTM([emb(y_{t-1}); c], s_{t-1})",
                "output": "p(y_t) = softmax(W·s_t + b)"
            }
        }
