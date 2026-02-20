from dataclasses import dataclass


@dataclass
class CQLPretrainConfig:
    """CQL pretraining configuration."""

    enabled: bool = False
    alpha: float = 1.0
    temperature: float = 1.0
    loss_weight: float = 1.0
