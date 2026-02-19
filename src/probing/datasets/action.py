"""
src/probing/datasets/action.py
==============================

Dataset for action prediction probing task.
"""

from .base import BaseProbeDataset
import torch


class ActionProbeDataset(BaseProbeDataset):
    """Dataset for action prediction probing task."""

    def _extract_target(self, batch) -> torch.Tensor:
        """Extract the action labels from the batch."""
        return batch['act']
    
    @property
    def probe_type(self) -> str:
        return 'action'
