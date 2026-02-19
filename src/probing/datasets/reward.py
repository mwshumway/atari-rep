"""
src/probing/datasets/reward.py
==============================

Dataset for reward prediction probing task.
"""

from .base import BaseProbeDataset
import torch


class RewardProbeDataset(BaseProbeDataset):
    """Dataset for reward prediction probing task."""

    def _extract_target(self, batch) -> torch.Tensor:
        """Extract the reward labels from the batch."""
        return batch['rew']
    
    @property
    def probe_type(self) -> str:
        return 'reward'
