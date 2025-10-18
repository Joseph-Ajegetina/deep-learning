"""
Comprehensive Evaluation Module for Speech Recognition Models.

Features:
- WER/CER calculation on test set
- Per-sample evaluation
- Attention weight visualization (for attention models)
- Model comparison (Classical RNN vs RNN with Attention)
- Sample predictions display
- Detailed error analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import time

from ..utils.metrics import batch_wer, batch_cer, detailed_error_analysis
from ..utils.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SequenceEvaluator:
    """
    Comprehensive evaluator for sequence-to-sequence speech recognition models.

    Evaluates both Classical RNN and RNN with Attention models,
    with special support for attention visualization.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        device: torch.device,
        save_dir: str = "./evaluation_results"
    ):
        """
        Initialize evaluator.

        Args:
            tokenizer: Text tokenizer for decoding
            device: Device to run evaluation on
            save_dir: Directory to save evaluation results
        """
        self.tokenizer = tokenizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SequenceEvaluator initialized")
        logger.info(f"Save directory: {self.save_dir}")

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str = "Model",
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a model.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name of the model for logging
            max_samples: Maximum number of samples to evaluate (None = all)

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating {model_name}...")

        model.eval()
        model.to(self.device)

        all_predictions = []
        all_references = []
        all_attention_weights = []
        inference_times = []
        sample_results = []

        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Evaluating {model_name}")

            for batch_idx, batch in enumerate(pbar):
                # Check if we've reached max samples
                if max_samples and total_samples >= max_samples:
                    break

                # Move batch to device
                audio_features = batch['audio_features'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                transcriptions = batch['transcriptions']

                batch_size = audio_features.size(0)

                # Measure inference time
                start_time = time.time()

                # Decode predictions
                predictions, attention_weights = model.decode(
                    audio_features=audio_features,
                    audio_lengths=audio_lengths,
                    max_length=200,
                    sos_idx=self.tokenizer.sos_idx,
                    eos_idx=self.tokenizer.eos_idx
                )

                inference_time = (time.time() - start_time) / batch_size * 1000  # ms per sample
                inference_times.append(inference_time)

                # Decode predictions to text
                pred_texts = self.tokenizer.decode(
                    predictions.cpu().tolist(),
                    skip_special_tokens=True
                )

                # Store results
                all_predictions.extend(pred_texts)
                all_references.extend(transcriptions)

                # Store attention weights if available
                if attention_weights is not None:
                    all_attention_weights.append(attention_weights.cpu())

                # Store sample results for detailed analysis
                for i in range(min(batch_size, 5 if batch_idx == 0 else 0)):  # Save first 5 samples
                    sample_results.append({
                        'reference': transcriptions[i],
                        'prediction': pred_texts[i],
                        'audio_length': audio_lengths[i].item(),
                        'attention': attention_weights[i].cpu() if attention_weights is not None else None
                    })

                total_samples += batch_size

                # Update progress
                pbar.set_postfix({
                    'Samples': total_samples,
                    'Inf_ms': f'{inference_time:.1f}'
                })

        # Calculate metrics
        logger.info(f"Calculating metrics for {len(all_predictions)} samples...")

        avg_wer, individual_wers = batch_wer(all_references, all_predictions)
        avg_cer, individual_cers = batch_cer(all_references, all_predictions)

        # Inference time statistics
        mean_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)

        # Model parameters
        model_parameters = model.count_parameters() if hasattr(model, 'count_parameters') else 0
        model_size_mb = model.get_parameter_size_mb() if hasattr(model, 'get_parameter_size_mb') else 0

        results = {
            'model_name': model_name,
            'num_samples': len(all_predictions),
            'overall_metrics': {
                'wer': avg_wer,
                'cer': avg_cer,
                'accuracy': max(0, 100 - avg_wer)  # Approximate accuracy from WER
            },
            'inference_time': {
                'mean_ms': mean_inference_time,
                'std_ms': std_inference_time
            },
            'model_info': {
                'parameters': model_parameters,
                'size_mb': model_size_mb,
                'has_attention': attention_weights is not None
            },
            'individual_results': {
                'wers': individual_wers,
                'cers': individual_cers
            },
            'sample_predictions': sample_results,
            'all_predictions': all_predictions,
            'all_references': all_references
        }

        # Save results
        self._save_results(results, model_name)

        logger.info(f"✅ {model_name} Evaluation Complete:")
        logger.info(f"   WER: {avg_wer:.2f}%")
        logger.info(f"   CER: {avg_cer:.2f}%")
        logger.info(f"   Accuracy: {results['overall_metrics']['accuracy']:.2f}%")
        logger.info(f"   Inference: {mean_inference_time:.2f}ms ± {std_inference_time:.2f}ms")
        logger.info(f"   Parameters: {model_parameters:,}")

        return results

    def compare_models(
        self,
        classical_results: Dict[str, Any],
        attention_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare Classical RNN and RNN with Attention results.

        Args:
            classical_results: Results from Classical RNN evaluation
            attention_results: Results from RNN with Attention evaluation

        Returns:
            Comparison dictionary
        """
        logger.info("Comparing models...")

        classical_metrics = classical_results['overall_metrics']
        attention_metrics = attention_results['overall_metrics']

        # WER improvement
        wer_improvement = classical_metrics['wer'] - attention_metrics['wer']
        wer_relative_improvement = (wer_improvement / classical_metrics['wer']) * 100

        # CER improvement
        cer_improvement = classical_metrics['cer'] - attention_metrics['cer']
        cer_relative_improvement = (cer_improvement / classical_metrics['cer']) * 100

        # Parameter comparison
        classical_params = classical_results['model_info']['parameters']
        attention_params = attention_results['model_info']['parameters']
        param_ratio = attention_params / classical_params if classical_params > 0 else 0

        # Inference time comparison
        classical_time = classical_results['inference_time']['mean_ms']
        attention_time = attention_results['inference_time']['mean_ms']
        time_ratio = attention_time / classical_time if classical_time > 0 else 0

        comparison = {
            'wer': {
                'classical': classical_metrics['wer'],
                'attention': attention_metrics['wer'],
                'improvement': wer_improvement,
                'relative_improvement': wer_relative_improvement
            },
            'cer': {
                'classical': classical_metrics['cer'],
                'attention': attention_metrics['cer'],
                'improvement': cer_improvement,
                'relative_improvement': cer_relative_improvement
            },
            'parameters': {
                'classical': classical_params,
                'attention': attention_params,
                'ratio': param_ratio
            },
            'inference_time': {
                'classical_ms': classical_time,
                'attention_ms': attention_time,
                'ratio': time_ratio
            },
            'summary': {
                'attention_is_better': attention_metrics['wer'] < classical_metrics['wer'],
                'wer_reduction': wer_improvement,
                'parameter_overhead': param_ratio - 1.0,
                'inference_overhead': time_ratio - 1.0
            }
        }

        logger.info(f"📊 Model Comparison:")
        logger.info(f"   WER: Classical {classical_metrics['wer']:.2f}% → Attention {attention_metrics['wer']:.2f}%")
        logger.info(f"   Improvement: {wer_improvement:.2f}% absolute ({wer_relative_improvement:.1f}% relative)")
        logger.info(f"   CER: Classical {classical_metrics['cer']:.2f}% → Attention {attention_metrics['cer']:.2f}%")
        logger.info(f"   Parameters: {classical_params:,} → {attention_params:,} ({param_ratio:.2f}x)")
        logger.info(f"   Inference: {classical_time:.1f}ms → {attention_time:.1f}ms ({time_ratio:.2f}x)")

        return comparison

    def visualize_attention(
        self,
        model: nn.Module,
        audio_features: torch.Tensor,
        audio_length: torch.Tensor,
        reference_text: str,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights for a single sample.

        Args:
            model: Model with attention mechanism
            audio_features: Audio features (1, time, features)
            audio_length: Audio length (1,)
            reference_text: Reference transcription
            save_path: Optional path to save figure
        """
        model.eval()
        model.to(self.device)

        with torch.no_grad():
            # Ensure batch dimension
            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0)
            if audio_length.dim() == 0:
                audio_length = audio_length.unsqueeze(0)

            audio_features = audio_features.to(self.device)
            audio_length = audio_length.to(self.device)

            # Decode with attention
            predictions, attention_weights = model.decode(
                audio_features=audio_features,
                audio_lengths=audio_length,
                max_length=200,
                sos_idx=self.tokenizer.sos_idx,
                eos_idx=self.tokenizer.eos_idx
            )

            if attention_weights is None:
                logger.warning("Model does not have attention mechanism")
                return

            # Get prediction text
            pred_text = self.tokenizer.decode(
                predictions[0].cpu().tolist(),
                skip_special_tokens=True
            )

            # Plot attention
            attention_weights = attention_weights[0].cpu().numpy()  # (output_len, input_len)

            # Truncate to actual lengths
            output_len = len(pred_text)
            input_len = audio_length[0].item()

            attention_weights = attention_weights[:output_len, :input_len]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot heatmap
            sns.heatmap(
                attention_weights,
                cmap='viridis',
                cbar=True,
                ax=ax,
                xticklabels=False,  # Too many time steps
                yticklabels=list(pred_text) if len(pred_text) < 50 else False
            )

            ax.set_xlabel('Audio Time Steps', fontsize=12)
            ax.set_ylabel('Generated Characters', fontsize=12)
            ax.set_title(
                f'Attention Weights\nReference: "{reference_text}"\nPrediction: "{pred_text}"',
                fontsize=14,
                pad=20
            )

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Attention visualization saved to {save_path}")

            plt.show()

    def plot_comparison(
        self,
        comparison: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing Classical RNN and RNN with Attention.

        Args:
            comparison: Comparison dictionary from compare_models()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: WER Comparison
        models = ['Classical RNN', 'RNN + Attention']
        wers = [comparison['wer']['classical'], comparison['wer']['attention']]

        axes[0, 0].bar(models, wers, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_ylabel('WER (%)', fontsize=12)
        axes[0, 0].set_title('Word Error Rate Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(wers):
            axes[0, 0].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 2: CER Comparison
        cers = [comparison['cer']['classical'], comparison['cer']['attention']]

        axes[0, 1].bar(models, cers, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 1].set_ylabel('CER (%)', fontsize=12)
        axes[0, 1].set_title('Character Error Rate Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(cers):
            axes[0, 1].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 3: Parameter Count
        params = [
            comparison['parameters']['classical'] / 1e6,
            comparison['parameters']['attention'] / 1e6
        ]

        axes[1, 0].bar(models, params, color=['#FF6B6B', '#4ECDC4'])
        axes[1, 0].set_ylabel('Parameters (Millions)', fontsize=12)
        axes[1, 0].set_title('Model Complexity', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(params):
            axes[1, 0].text(i, v + 0.1, f'{v:.2f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 4: Inference Time
        times = [
            comparison['inference_time']['classical_ms'],
            comparison['inference_time']['attention_ms']
        ]

        axes[1, 1].bar(models, times, color=['#FF6B6B', '#4ECDC4'])
        axes[1, 1].set_ylabel('Inference Time (ms)', fontsize=12)
        axes[1, 1].set_title('Inference Speed', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(times):
            axes[1, 1].text(i, v + 1, f'{v:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Overall title
        improvement_pct = comparison['wer']['relative_improvement']
        fig.suptitle(
            f'Classical RNN vs RNN with Attention\n'
            f'Attention achieves {improvement_pct:.1f}% relative WER reduction',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")

        plt.show()

    def generate_report(
        self,
        classical_results: Dict[str, Any],
        attention_results: Dict[str, Any],
        comparison: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive text report.

        Args:
            classical_results: Classical RNN results
            attention_results: Attention RNN results
            comparison: Comparison results
            save_path: Optional path to save report

        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("SPEECH RECOGNITION MODEL EVALUATION REPORT")
        report.append("AfriSpeech-200 Dataset")
        report.append("=" * 80)
        report.append("")

        # Classical RNN Section
        report.append("MODEL 1: CLASSICAL RNN (ENCODER-DECODER)")
        report.append("-" * 80)
        cm = classical_results['overall_metrics']
        ci = classical_results['model_info']
        ct = classical_results['inference_time']

        report.append(f"Word Error Rate (WER):     {cm['wer']:.2f}%")
        report.append(f"Character Error Rate (CER): {cm['cer']:.2f}%")
        report.append(f"Accuracy:                   {cm['accuracy']:.2f}%")
        report.append(f"Parameters:                 {ci['parameters']:,}")
        report.append(f"Model Size:                 {ci['size_mb']:.2f} MB")
        report.append(f"Inference Time:             {ct['mean_ms']:.2f} ± {ct['std_ms']:.2f} ms")
        report.append(f"Architecture:               Bidirectional LSTM Encoder + LSTM Decoder")
        report.append(f"Context:                    Fixed vector (bottleneck)")
        report.append("")

        # Attention RNN Section
        report.append("MODEL 2: RNN WITH ATTENTION")
        report.append("-" * 80)
        am = attention_results['overall_metrics']
        ai = attention_results['model_info']
        at = attention_results['inference_time']

        report.append(f"Word Error Rate (WER):     {am['wer']:.2f}%")
        report.append(f"Character Error Rate (CER): {am['cer']:.2f}%")
        report.append(f"Accuracy:                   {am['accuracy']:.2f}%")
        report.append(f"Parameters:                 {ai['parameters']:,}")
        report.append(f"Model Size:                 {ai['size_mb']:.2f} MB")
        report.append(f"Inference Time:             {at['mean_ms']:.2f} ± {at['std_ms']:.2f} ms")
        report.append(f"Architecture:               Bidirectional LSTM Encoder + Attention + LSTM Decoder")
        report.append(f"Context:                    Dynamic (computed via attention)")
        report.append("")

        # Comparison Section
        report.append("COMPARATIVE ANALYSIS")
        report.append("-" * 80)
        wer = comparison['wer']
        cer = comparison['cer']

        report.append(f"WER Improvement:")
        report.append(f"  Absolute:  {wer['improvement']:.2f}% (Classical {wer['classical']:.2f}% → Attention {wer['attention']:.2f}%)")
        report.append(f"  Relative:  {wer['relative_improvement']:.1f}% reduction")
        report.append("")

        report.append(f"CER Improvement:")
        report.append(f"  Absolute:  {cer['improvement']:.2f}% (Classical {cer['classical']:.2f}% → Attention {cer['attention']:.2f}%)")
        report.append(f"  Relative:  {cer['relative_improvement']:.1f}% reduction")
        report.append("")

        report.append(f"Model Complexity:")
        report.append(f"  Parameter Overhead: {comparison['summary']['parameter_overhead']*100:.1f}%")
        report.append(f"  Inference Overhead: {comparison['summary']['inference_overhead']*100:.1f}%")
        report.append("")

        # Conclusion
        report.append("KEY FINDINGS")
        report.append("-" * 80)
        if comparison['summary']['attention_is_better']:
            report.append("✅ RNN with Attention outperforms Classical RNN")
            report.append(f"   - Achieves {wer['relative_improvement']:.1f}% better WER")
            report.append(f"   - With only {comparison['summary']['parameter_overhead']*100:.1f}% more parameters")
            report.append(f"   - Attention mechanism eliminates fixed context bottleneck")
            report.append(f"   - Provides interpretable alignment (attention weights)")
        else:
            report.append("⚠️ Classical RNN performs comparably to Attention model")

        report.append("")
        report.append("RECOMMENDATION")
        report.append("-" * 80)
        report.append("For Ayamra Hospital's medical dictation system:")
        report.append("→ Use RNN with Attention for production deployment")
        report.append("  Reasons:")
        report.append("  1. Superior WER performance (fewer transcription errors)")
        report.append("  2. Better handling of long medical dictations")
        report.append("  3. Attention weights provide interpretability")
        report.append("  4. Acceptable computational overhead")
        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")

        return report_text

    def _save_results(self, results: Dict[str, Any], model_name: str):
        """Save evaluation results to file."""
        import json

        # Remove non-serializable items
        save_results = results.copy()
        save_results.pop('all_predictions', None)
        save_results.pop('all_references', None)

        # Remove attention weights from samples (too large)
        for sample in save_results.get('sample_predictions', []):
            sample['attention'] = None

        save_path = self.save_dir / f"{model_name.replace(' ', '_')}_results.json"

        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)

        logger.info(f"Results saved to {save_path}")
