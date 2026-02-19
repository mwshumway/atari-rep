"""
src/probing/datasets/value.py
==============================

Dataset for value prediction probing task.
"""

from .base import BaseProbeDataset
import torch


class ValueProbeDataset(BaseProbeDataset):
    """Dataset for value prediction probing task."""

    def _extract_target(self, batch) -> torch.Tensor:
        """
        Extract the value labels from the batch.
        
        the batches don't include 'value', so we need to estimate 
        them here using monte carlo returns

        as this is a regression task, we also normalize the 
        """
        return batch['rtg']

    @property
    def probe_type(self) -> str:
        return 'value'