"""
src/probing/probes/reward.py
==============================

Probe for reward prediction (ternary classification).
"""

from .base import BaseProbe
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt


class RewardProbe(BaseProbe):
    """Probe for reward prediction (ternary classification for -1, 0, 1)."""
    
    def _build_model(self, input_dim: int) -> nn.Module:
        from src.models import MHLinearHead, MHNonLinearHead
        # return MHLinearHead((input_dim,), 3, self.cfg.trainer.probe_num_heads)  # 3 classes
        return MHNonLinearHead(input_dim, self.cfg.probing.hidden_sizes, 3, self.cfg.trainer.probe_num_heads)
    
    def _build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()
    
    def _process_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to class predictions [0, 1, 2]"""
        return torch.argmax(logits, dim=-1)
    
    def _prepare_target(self, rewards: torch.Tensor) -> torch.Tensor:
        """Convert rewards {-1, 0, 1} to class indices {0, 1, 2}"""
        targets = (rewards + 1).long()  # Maps -1→0, 0→1, 1→2
        return targets
    
    def _targets_to_rewards(self, targets: np.ndarray) -> np.ndarray:
        """Convert class indices {0, 1, 2} back to rewards {-1, 0, 1}"""
        return targets - 1
    
    def _compute_metrics(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        game_id=None
    ) -> Dict[str, float]:
        """Compute metrics for ternary classification."""
        
        # Overall accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Macro F1 (treats all classes equally - good for imbalanced data)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        
        # Use sklearn's classification report for clean per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, labels=[0, 1, 2], zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            # Per-class recall (what you called accuracy)
            'recall_neg': float(recall[0]),   # -1 rewards
            'recall_zero': float(recall[1]),  # 0 rewards  
            'recall_pos': float(recall[2]),   # +1 rewards
            # Support (sample counts)
            'support_neg': int(support[0]),
            'support_zero': int(support[1]),
            'support_pos': int(support[2]),
        }
        
        return metrics
    
    def _plot_metrics_curve(self, metrics_history):
        """Plot accuracy and F1 curves."""
        accuracies = [m['accuracy'] for m in metrics_history]
        f1_macros = [m['f1_macro'] for m in metrics_history]
        f1_weighteds = [m['f1_weighted'] for m in metrics_history]
        
        # Check for per-class accuracies
        has_neg = 'acc_reward_-1' in metrics_history[0]
        has_zero = 'acc_reward_0' in metrics_history[0]
        has_pos = 'acc_reward_+1' in metrics_history[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Overall metrics
        axes[0].plot(accuracies, label='Accuracy', linewidth=2)
        axes[0].plot(f1_macros, label='F1 Macro', linewidth=2)
        axes[0].plot(f1_weighteds, label='F1 Weighted', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Metric Value')
        axes[0].set_title('Overall Metrics')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Per-class accuracies
        if has_neg:
            acc_neg = [m.get('acc_reward_-1', 0) for m in metrics_history]
            axes[1].plot(acc_neg, label='Negative Reward', linewidth=2)
        if has_zero:
            acc_zero = [m.get('acc_reward_0', 0) for m in metrics_history]
            axes[1].plot(acc_zero, label='Zero Reward', linewidth=2)
        if has_pos:
            acc_pos = [m.get('acc_reward_+1', 0) for m in metrics_history]
            axes[1].plot(acc_pos, label='Positive Reward', linewidth=2)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Per-Class Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f'/projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}/reward_metrics_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved metrics curve to {save_path}")