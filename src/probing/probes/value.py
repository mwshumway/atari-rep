"""
src/probing/probes/value.py
============================

Value prediction probing task.
"""
import torch
import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from .base import BaseProbe

class ValueProbe(BaseProbe):
    """Probe for value prediction (regression)."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
    
    def _build_model(self, input_dim: int) -> torch.nn.Module:
        from src.models.heads import MHLinearHead, MHNonLinearHead
        # return MHLinearHead(input_dim, 1, self.cfg.trainer.probe_num_heads)
        return MHNonLinearHead(input_dim, self.cfg.probing.hidden_sizes, 1, self.cfg.trainer.probe_num_heads)

    def _build_criterion(self) -> torch.nn.Module:
        return torch.nn.MSELoss()
    
    def _process_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Process logits to get predicted values."""
        return logits

    def _compute_metrics(
            self,
            predictions: np.ndarray,
            targets: np.ndarray,
            game_id=None
    ) -> Dict[str, float]:
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        return {"mse": mse, "r2": r2}
    
    def _plot_metrics_curve(self, metrics_history):
        """Plot an MSE and R2 curve."""
        mses = [m['mse'] for m in metrics_history]
        r2s = [m['r2'] for m in metrics_history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(mses, label='MSE')
        plt.plot(r2s, label='R2')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'/projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}_value_metrics_curve.png')
        plt.close()
        print(f"Saved metrics curve to /projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}_value_metrics_curve.png")
