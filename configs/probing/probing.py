from dataclasses import dataclass, field

from configs.probing.datasets import ProbingDatasetConfig
from configs.probing.probes import ProbingProbeConfig


@dataclass
class ProbingConfig:
    """End-to-end probing configuration."""

    enabled: bool = False
    dataset: ProbingDatasetConfig = field(default_factory=ProbingDatasetConfig)
    probe: ProbingProbeConfig = field(default_factory=ProbingProbeConfig)

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
