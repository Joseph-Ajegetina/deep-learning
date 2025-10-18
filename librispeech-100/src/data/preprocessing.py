"""
Audio preprocessing for speech recognition.

Handles:
- Loading audio files
- Resampling to target sample rate
- Feature extraction (MFCC, Mel-spectrograms)
- Normalization
- Audio augmentation
"""

import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocessor for converting audio files into features for speech recognition.

    Mathematical Foundation:
    - MFCC (Mel-Frequency Cepstral Coefficients):
      1. Pre-emphasis: y[n] = x[n] - α*x[n-1]
      2. Framing: Split signal into short frames (20-40ms)
      3. Windowing: Apply Hamming window to reduce spectral leakage
      4. FFT: Convert to frequency domain
      5. Mel filterbank: Apply triangular filters on mel scale
      6. Log: Take logarithm of energies
      7. DCT: Discrete Cosine Transform to decorrelate coefficients

    - Mel Scale: mel(f) = 2595 * log10(1 + f/700)
      Approximates human auditory system's frequency perception
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        feature_type: str = "mfcc",
        n_mfcc: int = 40,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        normalize: bool = True,
        normalize_type: str = "per_feature"
    ):
        """
        Initialize audio preprocessor.

        Args:
            sample_rate: Target sampling rate (Hz)
            feature_type: Type of features to extract ("mfcc" or "mel_spectrogram")
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window size for FFT
            normalize: Whether to normalize features
            normalize_type: Type of normalization ("per_feature" or "global")
        """
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalize = normalize
        self.normalize_type = normalize_type

        # Initialize transforms
        if feature_type == "mfcc":
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": n_mels,
                    "center": False
                }
            )
        elif feature_type == "mel_spectrogram":
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                center=False
            )
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        # For global normalization (computed on training set)
        self.global_mean = None
        self.global_std = None

    def load_audio(
        self,
        audio_path: Union[str, Path],
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file
            offset: Start time in seconds
            duration: Duration to load in seconds (None = load all)

        Returns:
            Tuple of (waveform, sample_rate)
            waveform shape: (num_channels, num_samples)
        """
        try:
            # Load audio using torchaudio
            waveform, sr = torchaudio.load(
                audio_path,
                frame_offset=int(offset * self.sample_rate) if offset > 0 else 0,
                num_frames=int(duration * self.sample_rate) if duration else -1
            )

            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            return waveform, self.sample_rate

        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features (MFCC or Mel-spectrogram) from waveform.

        Args:
            waveform: Audio waveform (1, num_samples)

        Returns:
            Features tensor (num_features, time_steps)
        """
        if self.feature_type == "mfcc":
            # Extract MFCCs
            # Output shape: (1, n_mfcc, time_steps)
            features = self.mfcc_transform(waveform)
            features = features.squeeze(0)  # (n_mfcc, time_steps)

        elif self.feature_type == "mel_spectrogram":
            # Extract Mel-spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = mel_spec.squeeze(0)  # (n_mels, time_steps)

            # Convert to log scale (log mel-spectrogram)
            features = torch.log(mel_spec + 1e-9)

        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        return features

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features using per-feature or global statistics.

        Mathematical operation:
        z = (x - μ) / σ
        where μ is mean, σ is standard deviation

        Args:
            features: Feature tensor (num_features, time_steps)

        Returns:
            Normalized features (num_features, time_steps)
        """
        if not self.normalize:
            return features

        if self.normalize_type == "per_feature":
            # Normalize each feature (MFCC coefficient) independently
            # Compute statistics along time axis
            mean = features.mean(dim=1, keepdim=True)
            std = features.std(dim=1, keepdim=True) + 1e-9
            normalized = (features - mean) / std

        elif self.normalize_type == "global":
            # Use global statistics (computed on training set)
            if self.global_mean is None or self.global_std is None:
                logger.warning("Global statistics not set, using per-feature normalization")
                mean = features.mean(dim=1, keepdim=True)
                std = features.std(dim=1, keepdim=True) + 1e-9
            else:
                mean = self.global_mean.unsqueeze(1)
                std = self.global_std.unsqueeze(1)

            normalized = (features - mean) / std

        else:
            raise ValueError(f"Unsupported normalize_type: {self.normalize_type}")

        return normalized

    def compute_global_stats(self, features_list: list):
        """
        Compute global mean and std from a list of feature tensors.

        Args:
            features_list: List of feature tensors (num_features, time_steps)
        """
        # Concatenate all features
        all_features = torch.cat([f for f in features_list], dim=1)

        # Compute global statistics
        self.global_mean = all_features.mean(dim=1)
        self.global_std = all_features.std(dim=1) + 1e-9

        logger.info(f"Global statistics computed from {len(features_list)} samples")
        logger.info(f"Mean shape: {self.global_mean.shape}, Std shape: {self.global_std.shape}")

    def process_audio(
        self,
        audio_path: Union[str, Path],
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> torch.Tensor:
        """
        Complete preprocessing pipeline: load → extract → normalize.

        Args:
            audio_path: Path to audio file
            offset: Start time in seconds
            duration: Duration in seconds

        Returns:
            Preprocessed features (num_features, time_steps)
        """
        # Load audio
        waveform, sr = self.load_audio(audio_path, offset, duration)

        # Extract features
        features = self.extract_features(waveform)

        # Normalize
        features = self.normalize_features(features)

        return features

    def process_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process a waveform that's already loaded.

        Args:
            waveform: Audio waveform (1, num_samples)

        Returns:
            Preprocessed features (num_features, time_steps)
        """
        features = self.extract_features(waveform)
        features = self.normalize_features(features)
        return features

    def get_feature_dim(self) -> int:
        """
        Get the feature dimension.

        Returns:
            Feature dimension (n_mfcc or n_mels)
        """
        if self.feature_type == "mfcc":
            return self.n_mfcc
        elif self.feature_type == "mel_spectrogram":
            return self.n_mels
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")


class AudioAugmentor:
    """
    Audio augmentation for training robustness.

    Techniques:
    - Time stretching: Change speed without changing pitch
    - Pitch shifting: Change pitch without changing speed
    - Add background noise: Simulate noisy environments
    - SpecAugment: Mask time/frequency in spectrogram
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        time_stretch_rate: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_steps: Tuple[int, int] = (-2, 2),
        noise_factor: float = 0.005,
        freq_mask_param: int = 15,
        time_mask_param: int = 35
    ):
        """
        Initialize audio augmentor.

        Args:
            sample_rate: Audio sample rate
            time_stretch_rate: Range for time stretching (min, max)
            pitch_shift_steps: Range for pitch shifting in semitones
            noise_factor: Noise amplitude factor
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
        """
        self.sample_rate = sample_rate
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift_steps = pitch_shift_steps
        self.noise_factor = noise_factor

        # SpecAugment transforms
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time stretching (speed change)."""
        rate = np.random.uniform(*self.time_stretch_rate)
        stretch = T.TimeStretch(n_freq=waveform.shape[-2], fixed_rate=rate)
        return stretch(waveform)

    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting using librosa."""
        n_steps = np.random.randint(*self.pitch_shift_steps)
        waveform_np = waveform.numpy()
        shifted = librosa.effects.pitch_shift(
            waveform_np.squeeze(),
            sr=self.sample_rate,
            n_steps=n_steps
        )
        return torch.from_numpy(shifted).unsqueeze(0)

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise."""
        noise = torch.randn_like(waveform) * self.noise_factor
        return waveform + noise

    def spec_augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment (frequency and time masking).

        Reference: SpecAugment: A Simple Data Augmentation Method for ASR
        """
        # Add batch dimension if needed
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)

        # Apply frequency masking
        spectrogram = self.freq_mask(spectrogram)

        # Apply time masking
        spectrogram = self.time_mask(spectrogram)

        return spectrogram.squeeze(0)

    def augment_waveform(
        self,
        waveform: torch.Tensor,
        apply_time_stretch: bool = True,
        apply_pitch_shift: bool = True,
        apply_noise: bool = True
    ) -> torch.Tensor:
        """
        Apply random augmentation to waveform.

        Args:
            waveform: Input waveform (1, num_samples)
            apply_time_stretch: Whether to apply time stretching
            apply_pitch_shift: Whether to apply pitch shifting
            apply_noise: Whether to add noise

        Returns:
            Augmented waveform
        """
        # Randomly apply augmentations with 50% probability each
        if apply_noise and np.random.rand() < 0.5:
            waveform = self.add_noise(waveform)

        if apply_pitch_shift and np.random.rand() < 0.5:
            waveform = self.pitch_shift(waveform)

        # Time stretch is more expensive, apply less frequently
        if apply_time_stretch and np.random.rand() < 0.3:
            # Note: This works on spectrograms, skip for now
            pass

        return waveform
