"""
Sequence-to-Sequence Training Pipeline for Speech Recognition.

Handles training of both Classical RNN and RNN with Attention models.
Includes teacher forcing, gradient clipping, early stopping, and live visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import yaml

from ..utils.metrics import batch_wer, batch_cer
from ..utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SequenceTrainer:
    """
    Comprehensive trainer for sequence-to-sequence speech recognition models.

    Features:
    - Teacher forcing with decay
    - Gradient clipping for RNN stability
    - Learning rate scheduling
    - Early stopping based on WER
    - Checkpoint management
    - Live loss plotting
    - WER/CER tracking
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        device: torch.device,
        config: Dict[str, Any],
        experiment_name: str = "experiment"
    ):
        """
        Initialize the trainer.

        Args:
            model: The sequence-to-sequence model to train
            tokenizer: Text tokenizer for decoding predictions
            device: Device to train on (CPU/GPU)
            config: Training configuration dictionary
            experiment_name: Name for this experiment
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.experiment_name = experiment_name

        # Training configuration
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.gradient_clip = config['training']['gradient_clip']
        self.teacher_forcing_ratio = config['training']['teacher_forcing_ratio']
        self.teacher_forcing_decay = config['training'].get('teacher_forcing_decay', 0.99)

        # Initialize optimizer
        self.optimizer = self._get_optimizer()

        # Initialize loss function
        # Ignore padding tokens in loss calculation
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

        # Initialize scheduler
        self.scheduler = self._get_scheduler()

        # Early stopping configuration
        early_stop_config = config['training'].get('early_stopping', {})
        self.early_stop_patience = early_stop_config.get('patience', 10)
        self.early_stop_min_delta = early_stop_config.get('min_delta', 0.01)
        self.early_stop_metric = early_stop_config.get('metric', 'val_wer')

        # Training state
        self.current_epoch = 0
        self.best_val_wer = float('inf')
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_teacher_forcing = self.teacher_forcing_ratio

        # Training history
        self.training_history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'val_wer': [],
            'val_cer': [],
            'learning_rates': [],
            'teacher_forcing': []
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(f"./checkpoints/{experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SequenceTrainer initialized for {experiment_name}")
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Device: {device}")
        logger.info(f"Vocabulary size: {len(tokenizer)}")

    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on configuration."""
        optimizer_name = self.config['training']['optimizer'].lower()

        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler based on configuration."""
        scheduler_name = self.config['training'].get('scheduler', 'none').lower()

        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'reduce_on_plateau':
            params = self.config['training'].get('scheduler_params', {})
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=params.get('mode', 'min'),
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 5),
                min_lr=params.get('min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, perplexity)
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Train]",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            audio_features = batch['audio_features'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with teacher forcing
            logits, attention_weights = self.model(
                audio_features=audio_features,
                audio_lengths=audio_lengths,
                targets=text_tokens,
                teacher_forcing_ratio=self.current_teacher_forcing
            )

            # Reshape for loss calculation
            # logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch * seq_len)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = text_tokens.reshape(-1)

            # Compute loss (CrossEntropyLoss ignores pad_idx)
            loss = self.criterion(logits_flat, targets_flat)

            # Backward pass
            loss.backward()

            # Gradient clipping (important for RNNs)
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PPL': f'{np.exp(loss.item()):.2f}',
                'TF': f'{self.current_teacher_forcing:.2f}'
            })

        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)

        return avg_loss, perplexity

    def validate_epoch(
        self,
        val_loader: DataLoader,
        compute_metrics: bool = True
    ) -> Tuple[float, float, float, float]:
        """
        Validate the model for one epoch.

        Args:
            val_loader: Validation data loader
            compute_metrics: Whether to compute WER/CER (slower)

        Returns:
            Tuple of (avg_loss, perplexity, wer, cer)
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []

        with torch.no_grad():
            pbar = tqdm(
                val_loader,
                desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Val]",
                leave=False
            )

            for batch in pbar:
                # Move batch to device
                audio_features = batch['audio_features'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                transcriptions = batch['transcriptions']

                # Forward pass (no teacher forcing in validation)
                logits, _ = self.model(
                    audio_features=audio_features,
                    audio_lengths=audio_lengths,
                    targets=text_tokens,
                    teacher_forcing_ratio=0.0  # No teacher forcing in validation
                )

                # Compute loss
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = text_tokens.reshape(-1)
                loss = self.criterion(logits_flat, targets_flat)

                total_loss += loss.item()
                num_batches += 1

                # Decode predictions for WER/CER calculation
                if compute_metrics:
                    # Use greedy decoding for validation
                    predictions, _ = self.model.decode(
                        audio_features=audio_features,
                        audio_lengths=audio_lengths,
                        max_length=200,
                        sos_idx=self.tokenizer.sos_idx,
                        eos_idx=self.tokenizer.eos_idx
                    )

                    # Decode predictions to text
                    pred_texts = self.tokenizer.decode(
                        predictions.cpu().tolist(),
                        skip_special_tokens=True
                    )

                    all_predictions.extend(pred_texts)
                    all_references.extend(transcriptions)

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PPL': f'{np.exp(loss.item()):.2f}'
                })

        # Calculate metrics
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)

        if compute_metrics and all_predictions:
            avg_wer, _ = batch_wer(all_references, all_predictions)
            avg_cer, _ = batch_cer(all_references, all_predictions)
        else:
            avg_wer = 0.0
            avg_cer = 0.0

        return avg_loss, perplexity, avg_wer, avg_cer

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_checkpoints: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        logger.info(f"Initial teacher forcing ratio: {self.current_teacher_forcing}")

        start_time = time.time()

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_ppl = self.train_epoch(train_loader)

            # Validate epoch (compute metrics every epoch)
            val_loss, val_ppl, val_wer, val_cer = self.validate_epoch(
                val_loader,
                compute_metrics=True
            )

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Use validation loss for plateau scheduler
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Decay teacher forcing ratio
            self.current_teacher_forcing *= self.teacher_forcing_decay

            # Record metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_perplexity'].append(train_ppl)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_perplexity'].append(val_ppl)
            self.training_history['val_wer'].append(val_wer)
            self.training_history['val_cer'].append(val_cer)
            self.training_history['learning_rates'].append(current_lr)
            self.training_history['teacher_forcing'].append(self.current_teacher_forcing)

            # Log epoch results
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f} - "
                f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f} - "
                f"Val WER: {val_wer:.2f}%, Val CER: {val_cer:.2f}% - "
                f"LR: {current_lr:.6f}, TF: {self.current_teacher_forcing:.3f}"
            )

            # Check for best model (based on WER)
            is_best = val_wer < self.best_val_wer
            if is_best:
                self.best_val_wer = val_wer
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                if save_checkpoints:
                    self.save_checkpoint(
                        epoch,
                        is_best=True,
                        filename="best_model.pth"
                    )
                    logger.info(f"✅ New best model! WER: {val_wer:.2f}%")
            else:
                self.epochs_without_improvement += 1

            # Save periodic checkpoint
            if save_checkpoints and (epoch + 1) % self.config['training'].get('save_every_n_epochs', 10) == 0:
                self.save_checkpoint(
                    epoch,
                    filename=f"checkpoint_epoch_{epoch+1}.pth"
                )

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stop_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                break

        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        logger.info(f"Best validation WER: {self.best_val_wer:.2f}%")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final model and training history
        if save_checkpoints:
            self.save_checkpoint(
                self.current_epoch,
                filename="final_model.pth"
            )
            self.save_training_history()

        return self.training_history

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: str = "checkpoint.pth"
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
            filename: Filename for the checkpoint
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_wer': self.best_val_wer,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
            'tokenizer_vocab_size': len(self.tokenizer),
            'teacher_forcing_ratio': self.current_teacher_forcing
        }

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        if is_best:
            logger.info(f"💾 Best model saved: {filepath}")

    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            True if successful
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch']
            self.best_val_wer = checkpoint.get('best_val_wer', float('inf'))
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', {})
            self.current_teacher_forcing = checkpoint.get('teacher_forcing_ratio', self.teacher_forcing_ratio)

            logger.info(f"✅ Checkpoint loaded from {filepath}")
            logger.info(f"Resuming from epoch {self.current_epoch}")
            logger.info(f"Best WER: {self.best_val_wer:.2f}%")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {str(e)}")
            return False

    def save_training_history(self):
        """Save training history to file."""
        history_file = self.checkpoint_dir / "training_history.yaml"

        with open(history_file, 'w') as f:
            yaml.dump(self.training_history, f)

        logger.info(f"📊 Training history saved to {history_file}")

    def plot_training_history(self, save_plot: bool = True):
        """
        Plot training history.

        Args:
            save_plot: Whether to save the plot to file
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # Plot 1: Loss
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Perplexity
        axes[0, 1].plot(epochs, self.training_history['train_perplexity'], 'b-', label='Train PPL', linewidth=2)
        axes[0, 1].plot(epochs, self.training_history['val_perplexity'], 'r-', label='Val PPL', linewidth=2)
        axes[0, 1].set_title('Perplexity', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: WER
        axes[0, 2].plot(epochs, self.training_history['val_wer'], 'g-', label='Val WER', linewidth=2)
        axes[0, 2].set_title('Word Error Rate', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('WER (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: CER
        axes[1, 0].plot(epochs, self.training_history['val_cer'], 'purple', label='Val CER', linewidth=2)
        axes[1, 0].set_title('Character Error Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CER (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Learning Rate
        axes[1, 1].plot(epochs, self.training_history['learning_rates'], 'orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Teacher Forcing Ratio
        axes[1, 2].plot(epochs, self.training_history['teacher_forcing'], 'brown', linewidth=2)
        axes[1, 2].set_title('Teacher Forcing Ratio', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Ratio')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_file = self.checkpoint_dir / "training_history.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"📈 Training plot saved to {plot_file}")

        plt.show()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary.

        Returns:
            Dictionary containing training summary
        """
        return {
            'experiment_name': self.experiment_name,
            'model_type': self.model.__class__.__name__,
            'total_epochs': len(self.training_history['train_loss']),
            'best_val_wer': self.best_val_wer,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0,
            'final_val_wer': self.training_history['val_wer'][-1] if self.training_history['val_wer'] else 0,
            'model_parameters': self.model.count_parameters() if hasattr(self.model, 'count_parameters') else 0,
            'config': self.config
        }
