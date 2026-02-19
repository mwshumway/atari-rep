"""
src/probing/probes/action.py
============================

Action prediction probing task.
"""

import torch
import torch.nn as nn
from torch.nn.modules import Module
import numpy as np
from typing import Dict
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from .base import BaseProbe


class ActionProbe(BaseProbe):
    """Probe for action prediction (classification)."""

    def __init__(self, cfg, device: torch.device, action_size: int):
        super().__init__(cfg, device)
        self.action_size = action_size
    
    def _build_model(self, input_dim: int) -> torch.nn.Module:
        """Build a linear model for action classification."""
        from src.models.heads import MHLinearHead, MHNonLinearHead
        # return MHLinearHead(input_dim, self.action_size, self.cfg.trainer.probe_num_heads) 
        print(f"Building Action Probe with Action Size: {self.action_size} and Num Heads: {self.cfg.trainer.probe_num_heads}")
        return MHNonLinearHead(input_dim, self.cfg.probing.hidden_sizes, self.action_size, self.cfg.trainer.probe_num_heads)

    def _build_criterion(self) -> Module:
        return nn.CrossEntropyLoss()
    
    def _process_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to predicted class labels."""
        return torch.argmax(logits, dim=-1)
    
    def _compute_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        game_id: np.ndarray = None
    ) -> dict:
        # Global metrics
        accuracy = np.mean(predictions == targets)
        f1 = f1_score(targets, predictions, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1),
        }
        
        # Optional: per-game breakdown
        if game_id is not None:
            print(game_id.shape, predictions.shape, targets.shape)

            unique_games = np.unique(game_id)
            per_game_metrics = {}
            for g in unique_games:
                mask = game_id == g
                acc_g = np.mean(predictions[mask] == targets[mask])
                f1_g = f1_score(targets[mask], predictions[mask], average='macro', zero_division=0)
                per_game_metrics[g] = {'accuracy': float(acc_g), 'f1_macro': float(f1_g)}
            
            metrics['by_game'] = per_game_metrics

        return metrics
    
    def _plot_metrics_curve(self, metrics_history):
        """Plot an accuracy and f1 curve."""
        accuracies = [m['accuracy'] for m in metrics_history]
        f1s = [m['f1_macro'] for m in metrics_history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(accuracies, label='Accuracy')
        plt.plot(f1s, label='F1 Macro')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'/projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}/action_metrics_curve.png')
        plt.close()
        print(f"Saved metrics curve to /projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}/action_metrics_curve.png")