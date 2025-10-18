"""
RNN with Attention Mechanism for Speech Recognition.

Architecture:
- Encoder: Bidirectional LSTM/GRU (same as classical RNN)
- Attention: Bahdanau or Luong attention mechanism
- Decoder: Unidirectional LSTM/GRU with dynamic context

Mathematical Foundation:
========================================

Attention Mechanism (Bahdanau Style):
------------------------------------
The key innovation: Instead of using a fixed context vector,
we compute a different context vector for each decoder step
by attending to different parts of the encoder outputs.

For decoder time step t:

1. Energy (Alignment Score):
   e_{t,i} = v^T · tanh(W_1·h_i + W_2·s_{t-1} + b)

   where:
   - h_i is encoder hidden state at time i
   - s_{t-1} is previous decoder hidden state
   - W_1, W_2, v, b are learnable parameters

2. Attention Weights:
   α_{t,i} = exp(e_{t,i}) / Σ_j exp(e_{t,j})

   Softmax ensures weights sum to 1.0
   High weights = model "attends" to these positions

3. Context Vector (Dynamic!):
   c_t = Σ_i α_{t,i} · h_i

   Weighted sum of all encoder hidden states
   Different for each decoder step!

4. Decoder Step:
   s_t = LSTM([emb(y_{t-1}); c_t], s_{t-1})

5. Output:
   p(y_t) = softmax(W · [s_t; c_t] + b)

Advantages over Classical RNN:
------------------------------
1. No information bottleneck - can access all encoder states
2. Better for long sequences - no need to compress everything
3. Interpretable - attention weights show what model focuses on
4. Empirically better performance (5-20% WER improvement)

Luong Attention (Alternative):
-----------------------------
Simpler variant, computes attention after decoder step:

1. Score: e_{t,i} = s_t^T · h_i  (dot product)
   or: e_{t,i} = s_t^T · W · h_i  (general)

2. Attention: α_t = softmax(e_t)

3. Context: c_t = Σ_i α_{t,i} · h_i

4. Output: h̃_t = tanh(W · [s_t; c_t])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import random

from .base import BaseSequenceModel
from .rnn_encoder_decoder import Encoder


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.

    Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
    Bahdanau et al., 2015

    This is the original attention mechanism for sequence-to-sequence models.
    """

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, attention_hidden_size: int):
        super().__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_hidden_size = attention_hidden_size

        # Linear layers for computing alignment scores
        # W_1 for encoder hidden states
        self.W_encoder = nn.Linear(encoder_hidden_size, attention_hidden_size, bias=False)

        # W_2 for decoder hidden state
        self.W_decoder = nn.Linear(decoder_hidden_size, attention_hidden_size, bias=False)

        # v for final scoring
        self.v = nn.Linear(attention_hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: (batch_size, decoder_hidden_size)
            encoder_outputs: (batch_size, max_time, encoder_hidden_size)
            encoder_mask: (batch_size, max_time) - 1 for valid positions, 0 for padding

        Returns:
            context: (batch_size, encoder_hidden_size)
            attention_weights: (batch_size, max_time)
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)

        # Project encoder outputs: (batch, max_time, attention_hidden)
        encoder_features = self.W_encoder(encoder_outputs)

        # Project decoder hidden: (batch, decoder_hidden) -> (batch, attention_hidden)
        # Then expand to (batch, 1, attention_hidden) for broadcasting
        decoder_features = self.W_decoder(decoder_hidden).unsqueeze(1)

        # Compute energy (alignment scores)
        # (batch, max_time, attention_hidden)
        combined = torch.tanh(encoder_features + decoder_features)

        # (batch, max_time, 1) -> (batch, max_time)
        energy = self.v(combined).squeeze(2)

        # Apply mask if provided (set padding positions to very negative value)
        if encoder_mask is not None:
            energy = energy.masked_fill(encoder_mask == 0, -1e9)

        # Compute attention weights (batch, max_time)
        attention_weights = F.softmax(energy, dim=1)

        # Compute context vector: weighted sum of encoder outputs
        # attention_weights: (batch, max_time) -> (batch, max_time, 1)
        # encoder_outputs: (batch, max_time, encoder_hidden)
        # context: (batch, encoder_hidden)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention Mechanism.

    Reference: "Effective Approaches to Attention-based Neural Machine Translation"
    Luong et al., 2015

    Simpler and more commonly used variant of attention.
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        score_type: str = "general"
    ):
        super().__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.score_type = score_type

        if score_type == "general":
            # General: score(h_i, s_t) = s_t^T W h_i
            self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        elif score_type == "concat":
            # Concat (similar to Bahdanau but simpler)
            self.W = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
            self.v = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: (batch_size, decoder_hidden_size)
            encoder_outputs: (batch_size, max_time, encoder_hidden_size)
            encoder_mask: (batch_size, max_time)

        Returns:
            context: (batch_size, encoder_hidden_size)
            attention_weights: (batch_size, max_time)
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)

        if self.score_type == "dot":
            # Dot: score(h_i, s_t) = s_t^T h_i
            # Requires encoder_hidden_size == decoder_hidden_size
            energy = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch, 1, hidden)
                encoder_outputs.transpose(1, 2)  # (batch, hidden, max_time)
            ).squeeze(1)  # (batch, max_time)

        elif self.score_type == "general":
            # General: score(h_i, s_t) = s_t^T W h_i
            # Project encoder outputs
            encoder_projected = self.W(encoder_outputs)  # (batch, max_time, decoder_hidden)

            energy = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch, 1, decoder_hidden)
                encoder_projected.transpose(1, 2)  # (batch, decoder_hidden, max_time)
            ).squeeze(1)  # (batch, max_time)

        elif self.score_type == "concat":
            # Concat: score(h_i, s_t) = v^T tanh(W [h_i; s_t])
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, max_time, -1)
            combined = torch.cat([encoder_outputs, decoder_expanded], dim=2)
            energy = self.v(torch.tanh(self.W(combined))).squeeze(2)

        # Apply mask
        if encoder_mask is not None:
            energy = energy.masked_fill(encoder_mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(energy, dim=1)

        # Compute context
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights


class AttentionDecoder(nn.Module):
    """
    RNN Decoder with Attention Mechanism.

    Key difference from classical decoder:
    - Uses dynamic context vector (different for each step)
    - Context computed via attention over all encoder outputs
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        attention_type: str = "bahdanau",
        attention_hidden_size: int = 256,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
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

        # Attention mechanism
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                attention_hidden_size=attention_hidden_size
            )
        elif attention_type == "luong":
            self.attention = LuongAttention(
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                score_type="general"
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # RNN layer (input is embedding + previous context)
        rnn_class = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_class(
            input_size=embedding_dim + encoder_hidden_size,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection (combines decoder state and context)
        self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size, vocab_size)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        decoder_hidden: Tuple[torch.Tensor, ...],
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Decoder forward pass with attention.

        Args:
            input_token: (batch_size,) or (batch_size, 1)
            decoder_hidden: Previous hidden state
            encoder_outputs: (batch_size, max_time, encoder_hidden_size)
            encoder_mask: (batch_size, max_time)

        Returns:
            output: (batch_size, vocab_size)
            decoder_hidden: Updated hidden state
            attention_weights: (batch_size, max_time)
        """
        # Ensure input_token is (batch_size, 1)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        # Embed input token
        embedded = self.embedding(input_token)
        embedded = self.dropout_layer(embedded)

        # Get current decoder hidden state (for attention)
        if isinstance(decoder_hidden, tuple):  # LSTM
            current_hidden = decoder_hidden[0][-1]  # Last layer
        else:  # GRU
            current_hidden = decoder_hidden[-1]

        # Compute attention
        context, attention_weights = self.attention(
            current_hidden,
            encoder_outputs,
            encoder_mask
        )

        # Concatenate embedding and context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        # Forward through RNN
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)

        # Combine decoder output and context for final prediction
        combined = torch.cat([output.squeeze(1), context], dim=1)
        logits = self.fc_out(combined)

        return logits, decoder_hidden, attention_weights


class RNNWithAttention(BaseSequenceModel):
    """
    RNN Encoder-Decoder with Attention Mechanism (PROSIT 2b).

    This model addresses the bottleneck limitation of classical RNNs
    by using attention to dynamically focus on different parts of the
    input sequence when generating each output token.

    Key Advantages:
    1. No fixed bottleneck - can access all encoder hidden states
    2. Better performance on long sequences
    3. Interpretable attention weights
    4. Empirically 5-20% better WER than classical RNN

    Mathematical Innovation:
    - Context vector is now dynamic: c_t (different for each step)
    - Attention weights α_{t,i} show which input positions are important
    - Model learns to "align" input and output sequences
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
        attention_type: str = "bahdanau",
        attention_hidden_size: int = 256,
        pad_idx: int = 0
    ):
        super().__init__(vocab_size=vocab_size, pad_idx=pad_idx)

        self.model_type = "RNN with Attention"
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bidirectional = encoder_bidirectional
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_type = attention_type

        # Encoder (same as classical RNN)
        self.encoder = Encoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            rnn_type=encoder_type,
            bidirectional=encoder_bidirectional,
            dropout=encoder_dropout
        )

        # Context size
        context_size = encoder_hidden_size * (2 if encoder_bidirectional else 1)

        # Bridge layer
        self.bridge = nn.Linear(context_size, decoder_hidden_size)

        # Decoder with Attention
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_hidden_size=context_size,
            decoder_hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            rnn_type=encoder_type,
            attention_type=attention_type,
            attention_hidden_size=attention_hidden_size,
            dropout=decoder_dropout,
            pad_idx=pad_idx
        )

    def _init_decoder_hidden(
        self,
        encoder_hidden: Tuple[torch.Tensor, ...],
        batch_size: int
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize decoder hidden state from encoder final state."""
        if isinstance(encoder_hidden, tuple):  # LSTM
            h_n = encoder_hidden[0]
        else:  # GRU
            h_n = encoder_hidden

        # Get final hidden states
        if self.encoder_bidirectional:
            context = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        else:
            context = h_n[-1, :, :]

        # Transform to decoder hidden size
        h_0 = torch.tanh(self.bridge(context))
        h_0 = h_0.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        if self.decoder.rnn_type == "lstm":
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0)
        else:
            return h_0

    def _create_encoder_mask(
        self,
        encoder_outputs: torch.Tensor,
        audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Create mask for padded positions in encoder outputs.

        Args:
            encoder_outputs: (batch_size, max_time, hidden_size)
            audio_lengths: (batch_size,)

        Returns:
            mask: (batch_size, max_time) - 1 for valid, 0 for padding
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)

        # Create range tensor
        range_tensor = torch.arange(max_time, device=encoder_outputs.device)
        range_tensor = range_tensor.unsqueeze(0).expand(batch_size, -1)

        # Create mask
        mask = range_tensor < audio_lengths.unsqueeze(1)

        return mask

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention.

        Args:
            audio_features: (batch_size, max_time, feature_dim)
            audio_lengths: (batch_size,)
            targets: (batch_size, max_target_len)
            teacher_forcing_ratio: Probability of teacher forcing

        Returns:
            logits: (batch_size, max_target_len, vocab_size)
            attention_weights: (batch_size, max_target_len, max_time)
        """
        batch_size = audio_features.size(0)

        # Encode audio
        encoder_outputs, encoder_hidden = self.encoder(audio_features, audio_lengths)

        # Create mask for padded positions
        encoder_mask = self._create_encoder_mask(encoder_outputs, audio_lengths)

        # Initialize decoder hidden state
        decoder_hidden = self._init_decoder_hidden(encoder_hidden, batch_size)

        # Decoding
        if targets is not None:
            # Training mode with teacher forcing
            max_target_len = targets.size(1)
            all_logits = []
            all_attention_weights = []

            # Start with SOS token
            decoder_input = torch.ones(batch_size, dtype=torch.long, device=audio_features.device)

            for t in range(max_target_len - 1):
                # Decoder step with attention
                logits, decoder_hidden, attention_weights = self.decoder(
                    decoder_input,
                    decoder_hidden,
                    encoder_outputs,
                    encoder_mask
                )

                all_logits.append(logits.unsqueeze(1))
                all_attention_weights.append(attention_weights.unsqueeze(1))

                # Teacher forcing
                use_teacher_forcing = random.random() < teacher_forcing_ratio

                if use_teacher_forcing:
                    decoder_input = targets[:, t]
                else:
                    decoder_input = logits.argmax(dim=1)

            # Concatenate all outputs
            logits = torch.cat(all_logits, dim=1)
            attention_weights = torch.cat(all_attention_weights, dim=1)

            # Pad to match target length
            pad = torch.zeros(batch_size, 1, self.vocab_size, device=logits.device)
            logits = torch.cat([logits, pad], dim=1)

            pad_attn = torch.zeros(batch_size, 1, encoder_outputs.size(1), device=logits.device)
            attention_weights = torch.cat([attention_weights, pad_attn], dim=1)

        else:
            # Inference mode (use decode method)
            raise NotImplementedError("Use decode() method for inference")

        return logits, attention_weights

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
        Greedy decoding for inference with attention.

        Args:
            audio_features: (batch_size, max_time, feature_dim)
            audio_lengths: (batch_size,)
            max_length: Maximum output length
            beam_size: Beam size (not implemented, always greedy)
            sos_idx: Start-of-sequence token
            eos_idx: End-of-sequence token

        Returns:
            predictions: (batch_size, max_length)
            attention_weights: (batch_size, max_length, max_time)
        """
        batch_size = audio_features.size(0)
        device = audio_features.device

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(audio_features, audio_lengths)
        encoder_mask = self._create_encoder_mask(encoder_outputs, audio_lengths)
        decoder_hidden = self._init_decoder_hidden(encoder_hidden, batch_size)

        # Greedy decoding
        predictions = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        all_attention_weights = torch.zeros(
            batch_size, max_length, encoder_outputs.size(1), device=device
        )

        decoder_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=device)

        for t in range(max_length):
            logits, decoder_hidden, attention_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                encoder_mask
            )

            predicted = logits.argmax(dim=1)
            predictions[:, t] = predicted
            all_attention_weights[:, t, :] = attention_weights

            # Use prediction as next input
            decoder_input = predicted

            # Stop if all sequences have EOS
            if (predicted == eos_idx).all():
                break

        return predictions, all_attention_weights

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
                "attention": {
                    "type": self.attention_type,
                    "mechanism": "Dynamic context vector computed at each decoder step",
                    "interpretable": "Attention weights show input-output alignment"
                },
                "decoder": {
                    "type": self.decoder.rnn_type.upper(),
                    "hidden_size": self.decoder_hidden_size,
                    "num_layers": self.decoder.num_layers,
                    "embedding_dim": self.decoder.embedding_dim,
                    "dropout": self.decoder.dropout
                }
            },
            "vocab_size": self.vocab_size,
            "advantage": "No bottleneck - attention accesses all encoder states",
            "mathematical_foundation": {
                "encoder": "h_1,...,h_T = BiLSTM(x_1,...,x_T)",
                "attention_energy": "e_{t,i} = v^T·tanh(W_1·h_i + W_2·s_{t-1})",
                "attention_weights": "α_{t,i} = exp(e_{t,i}) / Σ_j exp(e_{t,j})",
                "dynamic_context": "c_t = Σ_i α_{t,i}·h_i (different for each t!)",
                "decoder": "s_t = LSTM([emb(y_{t-1}); c_t], s_{t-1})",
                "output": "p(y_t) = softmax(W·[s_t; c_t] + b)"
            }
        }
